import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .layers import *
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers import TensorBoardLogger
from ..utils.tensor import permute_dims
from .loss import kld_loss, recon_loss
from piqa import SSIM
import matplotlib.pyplot as plt
import os


class AE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, output_size, num_layers=4, bias=False, dtype=torch.float32):
        super(AE, self).__init__()
        self.encoder = LinearEncoder(
            input_size, hidden_size, latent_size, num_layers, bias, dtype)
        self.decoder = LinearDecoder(
            latent_size, hidden_size, output_size, num_layers, bias, dtype)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)


# inspired by https://medium.com/@rekalantar/variational-auto-encoder-vae-pytorch-tutorial-dce2d2fe0f5f
class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, num_layers=4, bias=False, dtype=torch.float32):
        super(VAE, self).__init__()
        self.encoder = LinearEncoder(
            input_size, hidden_size, hidden_size, num_layers, bias, dtype)
        self.mu = nn.Linear(hidden_size, latent_size)
        self.logvar = nn.Linear(latent_size, hidden_size)
        self.decoder = LinearDecoder(
            hidden_size, hidden_size, input_size, num_layers, bias, dtype)

    def encode(self, x):
        x = self.encoder(x)
        return self.mu(x), self.logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decoder(z), mean, logvar, z


class ConvVAE(nn.Module):
    def __init__(self, in_channels, hidden_size, latent_size, layers_channels=[16, 32, 64, 128, 256, 512], input_size=512):
        super(ConvVAE, self).__init__()
        self.encoder = ConvEncoder(
            in_channels, hidden_size, layers_channels, input_size)
        self.mu = nn.Linear(hidden_size, latent_size)
        self.logvar = nn.Linear(hidden_size, latent_size)
        self.decoder = ConvDecoder(
            latent_size, in_channels, layers_channels, input_size)

    def encode(self, x):
        x = self.encoder(x)
        return self.mu(x), self.logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decoder(z)
        return recon, mean, logvar, z


# used this for guidance: https://github.com/1Konny/FactorVAE/blob/master/solver.py
class PlFactorVAE(LightningModule):
    def __init__(self, args):
        super(PlFactorVAE, self).__init__()
        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        self.save_hyperparameters()

        # data params
        self.in_channels = args.in_channels
        self.input_size = args.img_size
        self.hidden_size = args.hidden_size
        self.latent_size = args.latent_size
        self.layers_channels = args.layers_channels
        self.d_hidden_size = args.d_hidden_size
        self.d_num_layers = args.d_num_layers

        # losses
        self.mse = nn.MSELoss()
        # self.bce = recon_loss
        self.ssim = SSIM(n_channels=self.in_channels)
        self.kld = kld_loss
        self.kld_weight = args.kld_weight
        self.tc_weight = args.tc_weight
        self.l1_weight = args.l1_weight
        self.onpix_weight = args.onpix_weight

        # learning rates
        self.lr_vae = args.lr_vae
        self.lr_decay_vae = args.lr_decay_vae
        self.lr_d = args.lr_d
        self.lr_decay_d = args.lr_decay_d

        # logging
        self.plot_interval = args.plot_interval
        self.args = args

        # models
        self.VAE = ConvVAE(self.in_channels, self.hidden_size,
                           self.latent_size, self.layers_channels, self.input_size)
        self.D = LinearDiscriminator(
            self.latent_size, self.d_hidden_size, 2, self.d_num_layers)

    def forward(self, x):
        return self.VAE(x)

    def encode(self, x):
        mean, logvar = self.VAE.encode(x)
        return self.VAE.reparameterize(mean, logvar)

    def decode(self, z):
        return self.VAE.decoder(z)

    def training_step(self, batch, batch_idx):
        self.VAE.train()
        self.D.train()
        # get the optimizers and schedulers
        vae_optimizer, d_optimizer = self.optimizers()
        # vae_optimizer = self.optimizers()
        vae_scheduler, d_scheduler = self.lr_schedulers()

        # get the batch
        x_1, x_2 = batch
        batch_size = x_1.shape[0]

        # create a batch of ones and zeros for the discriminator
        ones = torch.ones(batch_size, dtype=torch.long, device=self.device)
        zeros = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        # VAE forward pass
        x_recon, mean, logvar, z = self.VAE(x_1)

        # VAE reconstruction loss
        # vae_recon_loss = self.mse(x_recon, x_1 * self.onpix_weight)
        # vae_recon_loss = self.bce(x_recon, x_1)
        vae_recon_loss = 1 - self.ssim(x_recon, x_1)
        # vae_recon_loss = self.mse(x_recon, x_1) + (1 - self.ssim(x_recon, x_1))

        # VAE KLD loss
        kld_loss = self.kld(mean, logvar) * self.kld_weight

        # VAE TC loss
        d_z = self.D(z)
        vae_tc_loss = (d_z[:, 0] - d_z[:, 1]).mean() * self.tc_weight

        # L1 penalty
        l1_penalty = sum([p.abs().sum()
                         for p in self.VAE.parameters()]) * self.l1_weight

        # VAE loss
        vae_loss = vae_recon_loss + kld_loss + vae_tc_loss + l1_penalty
        # vae_loss = vae_recon_loss + kld_loss + l1_penalty

        # VAE backward pass
        vae_optimizer.zero_grad()
        self.manual_backward(vae_loss, retain_graph=True)
        # self.manual_backward(vae_loss)
        # vae_optimizer.step()

        # Discriminator forward pass
        mean_2, logvar_2 = self.VAE.encode(x_2)
        z_2 = self.VAE.reparameterize(mean_2, logvar_2)
        z_2_perm = permute_dims(z_2)
        d_z_2_perm = self.D(z_2_perm.detach())
        d_tc_loss = 0.5 * (F.cross_entropy(d_z, zeros) +
                           F.cross_entropy(d_z_2_perm, ones))

        # Discriminator backward pass
        d_optimizer.zero_grad()
        self.manual_backward(d_tc_loss)
        vae_optimizer.step()
        d_optimizer.step()

        # LR scheduler step
        vae_scheduler.step()
        d_scheduler.step()

        # log the losses
        self.log_dict({
            "vae_loss": vae_loss,
            "vae_recon_loss": vae_recon_loss,
            "vae_kld_loss": kld_loss,
            "vae_tc_loss": vae_tc_loss,
            "d_tc_loss": d_tc_loss,
            "vae_l1_penalty": l1_penalty
        }, on_step=True, on_epoch=False)

    # def validation_step(self, batch, batch_idx):
    #     # get the batch
    #     x_1, x_2 = batch
    #     batch_size = x_1.shape[0]
    #     # print('batch_size:', batch_size)

    #     # create a batch of ones and zeros for the discriminator
    #     # ones = torch.ones(batch_size, dtype=torch.long, device=self.device)
    #     # zeros = torch.zeros(batch_size, dtype=torch.long, device=self.device)

    #     # VAE forward pass
    #     x_recon, mean, logvar, z = self.VAE(x_1)

    #     # VAE reconstruction loss
    #     vae_recon_loss = self.mse(x_recon, x_1 * self.onpix_weight)
    #     # vae_recon_loss = 1 - self.ssim(x_recon, x_1)
    #     # vae_recon_loss = self.mse(x_recon, x_1) + (1 - self.ssim(x_recon, x_1))

    #     # VAE KLD loss
    #     kld_loss = self.kld(mean, logvar) * self.kld_weight

    #     # VAE TC loss
    #     # d_z = self.D(z)
    #     # vae_tc_loss = (d_z[:, 0] - d_z[:, 1]).mean() * self.tc_weight

    #     # L1 penalty
    #     l1_penalty = sum([p.abs().sum()
    #                      for p in self.VAE.parameters()]) * self.l1_weight

    #     # VAE loss
    #     # vae_loss = vae_recon_loss + kld_loss + vae_tc_loss + l1_penalty
    #     vae_loss = vae_recon_loss + kld_loss + l1_penalty

    #     # Discriminator forward pass
    #     # mean_2, logvar_2 = self.VAE.encode(x_2)
    #     # z_2 = self.VAE.reparameterize(mean_2, logvar_2)
    #     # z_2_perm = permute_dims(z_2)
    #     # d_z_2_perm = self.D(z_2_perm)
    #     # d_tc_loss = 0.5 * (F.cross_entropy(d_z, zeros) + F.cross_entropy(d_z_2_perm, ones))

    #     # log the losses
    #     self.log_dict({
    #         "val_vae_loss": vae_loss,
    #         "val_vae_recon_loss": vae_recon_loss,
    #         "val_vae_kld_loss": kld_loss,
    #         # "val_vae_tc_loss": vae_tc_loss,
    #         # "val_d_tc_loss": d_tc_loss
    #         "val_vae_l1_penalty": l1_penalty
    #     }, on_step=False, on_epoch=True)

    def on_train_epoch_end(self) -> None:
        self.VAE.eval()
        # get the epoch number from trainer
        epoch = self.trainer.current_epoch

        if epoch % self.plot_interval != 0:
            return

        self.save_recon_plot()
        self.save_latent_space_plot()

    def save_recon_plot(self, num_recons=4):
        """Save a figure of N reconstructions"""
        fig, ax = plt.subplots(2, num_recons, figsize=(20, 10))
        # get the length of the training dataset
        dataset = self.trainer.train_dataloader.dataset
        for i in range(num_recons):
            # get a random index
            idx = torch.randint(0, len(dataset), (1,)).item()
            x, _ = dataset[idx]
            x_in = x.unsqueeze(0).to(self.device)
            x_recon, mean, logvar, z = self.VAE(x_in)
            x_recon = x_recon[0, 0, ...].detach().cpu().numpy()
            # plot ground truth image
            ax[0, i].imshow(x[0, ...], cmap="gray")
            ax[0, i].set_title(f"GT_{idx}")
            # plot reconstruction
            ax[1, i].imshow(x_recon, cmap="gray")
            ax[1, i].set_title(f"Recon_{idx}")
        # save figure to checkpoint folder/recons
        save_dir = os.path.join(self.args.ckpt_path,
                                self.args.ckpt_name, "recons")
        os.makedirs(save_dir, exist_ok=True)
        fig_name = f"recons_{str(self.trainer.current_epoch).zfill(5)}.png"
        plt.savefig(os.path.join(save_dir, fig_name))
        plt.close(fig)

    def save_latent_space_plot(self, batch_size=128):
        """Save a figure of the latent space"""
        # get the length of the training dataset
        dataset = self.trainer.train_dataloader.dataset
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, drop_last=False)
        z_all = torch.zeros(
            len(dataset), self.args.latent_size).to(self.device)
        for batch_idx, data in enumerate(loader):
            x, y = data
            x_recon, mean, logvar, z = self.VAE(x.to(self.device))
            z = z.detach()
            z_all[batch_idx*batch_size: batch_idx*batch_size + batch_size] = z
        z_all = z_all.cpu().numpy()
        # create the figure
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        ax.scatter(z_all[:, 0], z_all[:, 1])
        ax.set_title(
            f"Latent space at epoch {self.trainer.current_epoch}")
        # save figure to checkpoint folder/latent
        save_dir = os.path.join(self.args.ckpt_path,
                                self.args.ckpt_name, "latent")
        os.makedirs(save_dir, exist_ok=True)
        fig_name = f"latent_{str(self.trainer.current_epoch).zfill(5)}.png"
        plt.savefig(os.path.join(save_dir, fig_name))
        plt.close(fig)

    def log_tb_images(self, viz_batch) -> None:
        # Get tensorboard logger
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                tb_logger = logger.experiment
                break

        # get the epoch number from trainer
        epoch = self.trainer.current_epoch

        if tb_logger is None:
            raise ValueError('TensorBoard Logger not found')

        # Log the images (Give them different names)
        for img_idx, pair in enumerate(viz_batch):
            x_true, x_recon = pair
            tb_logger.add_image(f"GroundTruth/{img_idx}", x_true, epoch)
            tb_logger.add_image(f"Reconstruction/{img_idx}", x_recon, epoch)

    def configure_optimizers(self):
        vae_optimizer = torch.optim.Adam(
            self.VAE.parameters(), lr=self.lr_vae, betas=(0.9, 0.999))
        d_optimizer = torch.optim.Adam(
            self.D.parameters(), lr=self.lr_d, betas=(0.5, 0.9))
        vae_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            vae_optimizer, gamma=self.lr_decay_vae)
        d_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            d_optimizer, gamma=self.lr_decay_d)
        return [vae_optimizer, d_optimizer], [vae_scheduler, d_scheduler]
