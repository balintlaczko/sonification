import torch
from torch import nn
import torch.nn.functional as F
from .layers import *
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers import TensorBoardLogger
from ..utils.tensor import permute_dims
from .loss import kld_loss
from piqa import SSIM


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
    def __init__(
            self,
            in_channels,
            hidden_size,
            latent_size,
            layers_channels,
            input_size,
            d_hidden_size,
            d_num_layers,
            lr_vae,
            lr_d,
            kld_weight,
            tc_weight,
            train_dataset,
            val_dataset,
    ):
        super(PlFactorVAE, self).__init__()
        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        self.save_hyperparameters()
        # models
        self.VAE = ConvVAE(in_channels, hidden_size,
                           latent_size, layers_channels, input_size)
        # self.D = LinearDiscriminator(latent_size, d_hidden_size, 2, d_num_layers)
        # losses
        self.mse = nn.MSELoss()
        # self.ssim = SSIM(n_channels=in_channels)
        self.kld = kld_loss
        self.kld_weight = kld_weight
        self.tc_weight = tc_weight
        # learning rates
        self.lr_vae = lr_vae
        self.lr_d = lr_d
        # datasets
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def forward(self, x):
        return self.VAE(x)

    def encode(self, x):
        mean, logvar = self.VAE.encode(x)
        return self.VAE.reparameterize(mean, logvar)

    def decode(self, z):
        return self.VAE.decoder(z)

    def training_step(self, batch, batch_idx):
        # get the optimizers
        # vae_optimizer, d_optimizer = self.optimizers()
        vae_optimizer = self.optimizers()

        # get the batch
        x_1, x_2 = batch
        batch_size = x_1.shape[0]

        # create a batch of ones and zeros for the discriminator
        # ones = torch.ones(batch_size, dtype=torch.long, device=self.device)
        # zeros = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        # VAE forward pass
        x_recon, mean, logvar, z = self.VAE(x_1)
        vae_recon_loss = self.mse(x_recon, x_1*4)
        # vae_recon_loss = 1 - self.ssim(x_recon, x_1)
        # vae_recon_loss = self.mse(x_recon, x_1) + (1 - self.ssim(x_recon, x_1))
        kld_loss = self.kld(mean, logvar)
        # VAE TC loss
        # d_z = self.D(z)
        # vae_tc_loss = (d_z[:, 0] - d_z[:, 1]).mean()
        # L1 penalty
        l1_weight = 1e-6
        l1_penalty = l1_weight * sum([p.abs().sum()
                                     for p in self.VAE.parameters()])

        # VAE loss
        # vae_loss = vae_recon_loss + self.kld_weight * \
        #     kld_loss  # + self.tc_weight * vae_tc_loss
        vae_loss = vae_recon_loss + self.kld_weight * kld_loss + l1_penalty

        # VAE backward pass
        vae_optimizer.zero_grad()
        # self.manual_backward(vae_loss, retain_graph=True)
        self.manual_backward(vae_loss)
        vae_optimizer.step()

        # Discriminator forward pass
        # mean_2, logvar_2 = self.VAE.encode(x_2)
        # z_2 = self.VAE.reparameterize(mean_2, logvar_2)
        # z_2_perm = permute_dims(z_2)
        # d_z_2_perm = self.D(z_2_perm.detach())
        # d_tc_loss = 0.5 * (F.cross_entropy(d_z, zeros) + F.cross_entropy(d_z_2_perm, ones))

        # Discriminator backward pass
        # d_optimizer.zero_grad()
        # self.manual_backward(d_tc_loss)
        # vae_optimizer.step()
        # d_optimizer.step()

        # log the losses
        self.log_dict({
            "vae_loss": vae_loss,
            "vae_recon_loss": vae_recon_loss,
            "vae_kld_loss": kld_loss,
            # "vae_tc_loss": vae_tc_loss,
            # "d_tc_loss": d_tc_loss
        }, on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        # get the batch
        x_1, x_2 = batch
        batch_size = x_1.shape[0]
        # print('batch_size:', batch_size)

        # create a batch of ones and zeros for the discriminator
        # ones = torch.ones(batch_size, dtype=torch.long, device=self.device)
        # zeros = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        # VAE forward pass
        x_recon, mean, logvar, z = self.VAE(x_1)
        vae_recon_loss = self.mse(x_recon, x_1)
        # vae_recon_loss = 1 - self.ssim(x_recon, x_1)
        # vae_recon_loss = self.mse(x_recon, x_1) + (1 - self.ssim(x_recon, x_1))
        kld_loss = self.kld(mean, logvar)
        # VAE TC loss
        # d_z = self.D(z)
        # vae_tc_loss = (d_z[:, 0] - d_z[:, 1]).mean()

        # VAE loss
        vae_loss = vae_recon_loss + self.kld_weight * \
            kld_loss  # + self.tc_weight * vae_tc_loss

        # Discriminator forward pass
        # mean_2, logvar_2 = self.VAE.encode(x_2)
        # z_2 = self.VAE.reparameterize(mean_2, logvar_2)
        # z_2_perm = permute_dims(z_2)
        # d_z_2_perm = self.D(z_2_perm)
        # d_tc_loss = 0.5 * (F.cross_entropy(d_z, zeros) + F.cross_entropy(d_z_2_perm, ones))

        # log the losses
        self.log_dict({
            "val_vae_loss": vae_loss,
            "val_vae_recon_loss": vae_recon_loss,
            "val_vae_kld_loss": kld_loss,
            # "val_vae_tc_loss": vae_tc_loss,
            # "val_d_tc_loss": d_tc_loss
        }, on_step=False, on_epoch=True)

    def on_validation_end(self) -> None:
        # get a batch of images
        num_val_samples = len(self.val_dataset)
        # get a random image
        idx = torch.randint(0, num_val_samples, (1,)).item()
        x_1, x_2 = self.val_dataset[idx]
        x_1, x_2 = x_1.unsqueeze(0).cuda(), x_2.unsqueeze(0).cuda()

        x_1_recon, mean, logvar, z = self.VAE(x_1)
        x_2_recon, mean, logvar, z = self.VAE(x_2)

        # log the images
        self.log_tb_images([[x_1.squeeze(0), x_1_recon.squeeze(0)],
                           [x_2.squeeze(0), x_2_recon.squeeze(0)]])

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
            self.VAE.parameters(), lr=self.hparams.lr_vae, betas=(0.9, 0.999))
        # d_optimizer = torch.optim.Adam(self.D.parameters(), lr=self.hparams.lr_d, betas=(0.5, 0.9))
        # return vae_optimizer, d_optimizer
        return vae_optimizer
