import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .layers import *
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers import TensorBoardLogger
from ..utils.tensor import permute_dims
from .loss import kld_loss, recon_loss, MMDloss, preserve_distance_loss, preserve_axiswise_distance_loss
from piqa import SSIM
import matplotlib.pyplot as plt
import os
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import numpy as np
# import pca from sklearn
from sklearn.decomposition import PCA


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
        self.logvar = nn.Linear(hidden_size, latent_size)
        self.decoder = LinearDecoder(
            latent_size, hidden_size, input_size, num_layers, bias, dtype)

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


class ConvVAE(nn.Module):
    def __init__(self, in_channels, latent_size, layers_channels=[16, 32, 64, 128, 256, 512], input_size=512):
        super(ConvVAE, self).__init__()
        self.encoder = ConvEncoder(
            in_channels, latent_size, layers_channels, input_size)
        self.mu = nn.Linear(latent_size, latent_size)
        self.logvar = nn.Linear(latent_size, latent_size)
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


class ConvVAE1D(nn.Module):
    def __init__(self, in_channels, latent_size, layers_channels=[16, 32, 64, 128, 256], input_size=64):
        super(ConvVAE1D, self).__init__()
        self.encoder = ConvEncoder1D(
            in_channels, latent_size, layers_channels, input_size)
        self.mu = nn.Linear(latent_size, latent_size)
        self.logvar = nn.Linear(latent_size, latent_size)
        self.decoder = ConvDecoder1D(
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


class ConvWAE(nn.Module):
    def __init__(self, in_channels, latent_size, layers_channels=[16, 32, 64, 128, 256, 512], input_size=512, z_quantization=0.0):
        super(ConvWAE, self).__init__()
        self.encoder = ConvEncoder(
            in_channels, latent_size, layers_channels, input_size)
        self.decoder = ConvDecoder(
            latent_size, in_channels, layers_channels, input_size)
        self.z_quantization = z_quantization

    def encode(self, x):
        return self.encoder(x)

    def quantize(self, z, z_quantization):
        return z_quantization * torch.round(z / z_quantization)

    def forward(self, x):
        z = self.encode(x)
        if self.z_quantization > 0:
            z = self.quantize(z, self.z_quantization)
        recon = self.decoder(z)
        return recon, z


class PlVAE(LightningModule):
    def __init__(self, args):
        super(PlVAE, self).__init__()
        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        # self.save_hyperparameters()

        # data params
        self.in_channels = args.in_channels
        self.input_size = args.img_size
        self.latent_size = args.latent_size
        self.layers_channels = args.layers_channels

        # losses
        self.mse = nn.MSELoss()
        self.kld = kld_loss
        self.recon_weight = args.recon_weight
        self.kld_weight_max = args.kld_weight_max
        self.kld_weight_min = args.kld_weight_min
        self.kld_weight_dynamic = args.kld_weight_min  # initialize to min
        self.kld_start_epoch = args.kld_start_epoch
        self.kld_warmup_epochs = args.kld_warmup_epochs

        # learning rates
        self.lr_vae = args.lr_vae
        self.lr_decay_vae = args.lr_decay_vae

        # logging
        self.plot_interval = args.plot_interval
        self.args = args

        # models
        self.VAE = ConvVAE(self.in_channels, self.latent_size,
                           self.layers_channels, self.input_size)

    def forward(self, x):
        return self.VAE(x)

    def encode(self, x):
        mean, logvar = self.VAE.encode(x)
        return self.VAE.reparameterize(mean, logvar)

    def decode(self, z):
        return self.VAE.decoder(z)

    def training_step(self, batch, batch_idx):
        self.VAE.train()
        # get the optimizers and schedulers
        vae_optimizer = self.optimizers()
        vae_scheduler = self.lr_schedulers()

        # get the batch
        x_1, x_2 = batch
        batch_size = x_1.shape[0]
        epoch_idx = self.trainer.current_epoch

        # VAE forward pass
        x_recon, mean, logvar, z = self.VAE(x_1)

        # VAE reconstruction loss
        vae_recon_loss = self.mse(x_recon, x_1)
        scaled_vae_recon_loss = vae_recon_loss * self.recon_weight

        # VAE KLD loss
        if self.args.dynamic_kld > 0:
            kld_scale = self.kld_weight_dynamic
        else:
            kld_scale = (self.kld_weight_max - self.kld_weight_min) * \
                min(1.0, (epoch_idx - self.kld_start_epoch) /
                    self.kld_warmup_epochs) + self.kld_weight_min if epoch_idx > self.kld_start_epoch else self.kld_weight_min
        kld_loss = self.kld(mean, logvar)
        scaled_kld_loss = kld_loss * kld_scale

        # VAE loss
        vae_loss = scaled_vae_recon_loss + scaled_kld_loss

        # VAE backward pass
        vae_optimizer.zero_grad()
        self.manual_backward(vae_loss)
        self.clip_gradients(vae_optimizer, gradient_clip_val=0.5,
                            gradient_clip_algorithm="norm")

        # Optimizer step
        vae_optimizer.step()

        # LR scheduler step
        vae_scheduler.step()

        # log the losses
        self.last_recon_loss = vae_recon_loss
        self.log_dict({
            "vae_loss": vae_loss,
            "vae_recon_loss": vae_recon_loss,
            "vae_kld_loss": kld_loss,
        })
        if self.args.dynamic_kld > 0:
            self.log_dict({"kld_scale": kld_scale})

    def validation_step(self, batch, batch_idx):
        self.VAE.eval()

        # get the batch
        x_1, x_2 = batch
        batch_size = x_1.shape[0]

        # VAE forward pass
        x_recon, mean, logvar, z = self.VAE(x_1)

        # VAE reconstruction loss
        vae_recon_loss = self.mse(x_recon, x_1)

        # VAE KLD loss
        kld_loss = self.kld(mean, logvar)

        # VAE loss
        vae_loss = vae_recon_loss + kld_loss

        # log the losses
        self.last_recon_loss = vae_recon_loss
        self.log_dict({
            "val_vae_loss": vae_loss,
            "val_vae_recon_loss": vae_recon_loss,
            "val_vae_kld_loss": kld_loss,
        })

    def on_validation_epoch_end(self) -> None:
        self.VAE.eval()
        # get the epoch number from trainer
        epoch = self.trainer.current_epoch

        if self.args.dynamic_kld > 0:
            if self.last_recon_loss < self.args.target_recon_loss:
                self.kld_weight_dynamic += 0.0001

        if epoch % self.plot_interval != 0 and epoch != 0:
            return

        self.save_recon_plot()
        self.save_latent_space_plot()

    def save_recon_plot(self, num_recons=4):
        """Save a figure of N reconstructions"""
        fig, ax = plt.subplots(2, num_recons, figsize=(20, 10))
        # get the length of the training dataset
        dataset = self.trainer.val_dataloaders.dataset
        for i in range(num_recons):
            # get a random index
            idx = torch.randint(0, len(dataset), (1,)).item()
            x, _ = dataset[idx]
            x_in = x.unsqueeze(0).to(self.device)
            x_recon, mean, logvar, z = self.VAE(x_in)
            # x_recon, z = self.VAE(x_in)
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
        dataset = self.trainer.val_dataloaders.dataset
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
        # if latent space is more than 2D then use PCA to reduce to 2D
        if self.args.latent_size > 2:
            pca = PCA(n_components=2)
            z_all = pca.fit_transform(z_all)
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

    def configure_optimizers(self):
        vae_optimizer = torch.optim.Adam(
            self.VAE.parameters(), lr=self.lr_vae, betas=(0.9, 0.999))
        vae_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            vae_optimizer, gamma=self.lr_decay_vae)
        return [vae_optimizer], [vae_scheduler]


# used this for guidance: https://github.com/1Konny/FactorVAE/blob/master/solver.py
class PlFactorVAE(LightningModule):
    def __init__(self, args):
        super(PlFactorVAE, self).__init__()
        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        # self.save_hyperparameters()

        # data params
        self.in_channels = args.in_channels
        self.input_size = args.img_size
        self.latent_size = args.latent_size
        self.layers_channels = args.layers_channels
        self.d_hidden_size = args.d_hidden_size
        self.d_num_layers = args.d_num_layers

        # losses
        self.mse = nn.MSELoss()
        self.kld = kld_loss
        self.recon_weight = args.recon_weight
        self.kld_weight_max = args.kld_weight_max
        self.kld_weight_min = args.kld_weight_min
        self.kld_weight_dynamic = args.kld_weight_min  # initialize to min
        self.kld_start_epoch = args.kld_start_epoch
        self.kld_warmup_epochs = args.kld_warmup_epochs
        self.tc_weight = args.tc_weight

        # learning rates
        self.lr_vae = args.lr_vae
        self.lr_decay_vae = args.lr_decay_vae
        self.lr_d = args.lr_d
        self.lr_decay_d = args.lr_decay_d

        # logging
        self.plot_interval = args.plot_interval
        self.args = args

        # dataset
        self.dataset_size = args.dataset_size

        # models
        self.VAE = ConvVAE(self.in_channels, self.latent_size,
                           self.layers_channels, self.input_size)
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
        vae_scheduler, d_scheduler = self.lr_schedulers()

        # get the batch
        x_1, x_2 = batch
        batch_size = x_1.shape[0]
        epoch_idx = self.trainer.current_epoch

        # create a batch of ones and zeros for the discriminator
        ones = torch.ones(batch_size, dtype=torch.long, device=self.device)
        zeros = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        # VAE forward pass
        x_recon, mean, logvar, z = self.VAE(x_1)

        # VAE reconstruction loss
        vae_recon_loss = self.mse(x_recon, x_1)
        scaled_vae_recon_loss = vae_recon_loss * self.recon_weight

        # VAE KLD loss
        if self.args.dynamic_kld > 0:
            kld_scale = self.kld_weight_dynamic
        else:
            kld_scale = (self.kld_weight_max - self.kld_weight_min) * \
                min(1.0, (epoch_idx - self.kld_start_epoch) /
                    self.kld_warmup_epochs) + self.kld_weight_min if epoch_idx > self.kld_start_epoch else self.kld_weight_min
        kld_loss = self.kld(mean, logvar)
        scaled_kld_loss = kld_loss * kld_scale

        # VAE TC loss
        d_z = self.D(z)
        vae_tc_loss = F.cross_entropy(d_z, ones)
        scaled_vae_tc_loss = vae_tc_loss * self.tc_weight

        # VAE loss
        vae_loss = scaled_vae_recon_loss + scaled_kld_loss + scaled_vae_tc_loss

        # VAE backward pass
        vae_optimizer.zero_grad()
        self.manual_backward(vae_loss, retain_graph=True)
        self.clip_gradients(vae_optimizer, gradient_clip_val=0.5,
                            gradient_clip_algorithm="norm")

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

        # Optimizer step
        vae_optimizer.step()
        d_optimizer.step()

        # LR scheduler step
        vae_scheduler.step()
        d_scheduler.step()

        # log the losses
        self.last_recon_loss = vae_recon_loss
        self.log_dict({
            "vae_loss": vae_loss,
            "vae_recon_loss": vae_recon_loss,
            "vae_kld_loss": kld_loss,
            "vae_tc_loss": vae_tc_loss,
            "d_tc_loss": d_tc_loss,
        })
        if self.args.dynamic_kld > 0:
            self.log_dict({"kld_scale": kld_scale})

    def on_train_epoch_end(self) -> None:
        self.VAE.eval()
        # get the epoch number from trainer
        epoch = self.trainer.current_epoch

        if self.args.dynamic_kld > 0:
            if self.last_recon_loss < self.args.target_recon_loss:
                self.kld_weight_dynamic *= 1.01

        if epoch % self.plot_interval != 0 and epoch != 0:
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
            # x_recon, z = self.VAE(x_in)
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


class PlFactorVAE1D(LightningModule):
    def __init__(self, args):
        super(PlFactorVAE1D, self).__init__()
        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        # self.save_hyperparameters()

        # data params
        self.in_channels = args.in_channels
        self.input_size = args.img_size
        self.latent_size = args.latent_size
        self.layers_channels = args.layers_channels
        self.d_hidden_size = args.d_hidden_size
        self.d_num_layers = args.d_num_layers

        # losses
        self.mse = nn.MSELoss()
        self.bce = recon_loss
        self.kld = kld_loss
        self.recon_weight = args.recon_weight
        self.last_recon_loss = float("inf")  # initialize to infinity
        self.kld_weight_max = args.kld_weight_max
        self.kld_weight_min = args.kld_weight_min
        self.kld_weight_dynamic = args.kld_weight_min  # initialize to min
        self.kld_start_epoch = args.kld_start_epoch
        self.kld_warmup_epochs = args.kld_warmup_epochs
        self.tc_weight = args.tc_weight
        self.tc_start = args.tc_start
        self.tc_warmup_epochs = args.tc_warmup_epochs

        # learning rates
        self.lr_vae = args.lr_vae
        self.lr_decay_vae = args.lr_decay_vae
        self.lr_d = args.lr_d
        self.lr_decay_d = args.lr_decay_d

        # logging
        self.plot_interval = args.plot_interval
        self.args = args

        # models
        self.VAE = ConvVAE1D(self.in_channels, self.latent_size,
                             self.layers_channels, self.input_size)
        self.D = LinearDiscriminator(
            self.latent_size, self.d_hidden_size, 2, self.d_num_layers)

        # train dataset scaler
        self.train_scaler = args.train_scaler

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
        epoch_idx = self.trainer.current_epoch
        # get the optimizers and schedulers
        vae_optimizer, d_optimizer = self.optimizers()
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
        vae_recon_loss = self.mse(x_recon, x_1)
        scaled_vae_recon_loss = vae_recon_loss * self.recon_weight

        # VAE KLD loss
        if self.args.dynamic_kld > 0:
            kld_scale = self.kld_weight_dynamic
        else:
            kld_scale = (self.kld_weight_max - self.kld_weight_min) * \
                min(1.0, (epoch_idx - self.kld_start_epoch) /
                    self.kld_warmup_epochs) + self.kld_weight_min if epoch_idx > self.kld_start_epoch else self.kld_weight_min
        kld_loss = self.kld(mean, logvar)
        scaled_kld_loss = kld_loss * kld_scale

        # VAE TC loss
        d_z = self.D(z)
        vae_tc_scale = self.tc_weight * \
            min(1.0, (epoch_idx - self.tc_start) /
                self.tc_warmup_epochs) if epoch_idx > self.tc_start else 0
        vae_tc_loss = F.cross_entropy(d_z, ones)
        scaled_vae_tc_loss = vae_tc_loss * vae_tc_scale

        # VAE loss
        vae_loss = scaled_vae_recon_loss + scaled_kld_loss + scaled_vae_tc_loss

        # VAE backward pass
        vae_optimizer.zero_grad()
        self.manual_backward(vae_loss, retain_graph=True)
        self.clip_gradients(vae_optimizer, gradient_clip_val=0.5,
                            gradient_clip_algorithm="norm")

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

        # Optimizer step
        vae_optimizer.step()
        d_optimizer.step()

        # LR scheduler step
        vae_scheduler.step()
        d_scheduler.step()

        # log the losses
        self.last_recon_loss = vae_recon_loss
        self.log_dict({
            "vae_loss": vae_loss,
            "vae_recon_loss": vae_recon_loss,
            "vae_kld_loss": kld_loss,
            "vae_tc_loss": vae_tc_loss,
            "d_tc_loss": d_tc_loss,
        })
        if self.args.dynamic_kld > 0:
            self.log_dict({"kld_scale": kld_scale})

    def validation_step(self, batch, batch_idx):
        self.VAE.eval()
        self.D.eval()

        # get the batch
        x_1, x_2 = batch
        batch_size = x_1.shape[0]

        # create a batch of ones and zeros for the discriminator
        ones = torch.ones(batch_size, dtype=torch.long, device=self.device)
        zeros = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        # VAE forward pass
        x_recon, mean, logvar, z = self.VAE(x_1)

        # VAE reconstruction loss
        vae_recon_loss = self.mse(x_recon, x_1)

        # VAE KLD loss
        kld_loss = self.kld(mean, logvar)

        # VAE TC loss
        d_z = self.D(z)
        vae_tc_loss = F.cross_entropy(d_z, ones)

        # VAE loss
        vae_loss = vae_recon_loss + kld_loss + vae_tc_loss

        # Discriminator forward pass
        mean_2, logvar_2 = self.VAE.encode(x_2)
        z_2 = self.VAE.reparameterize(mean_2, logvar_2)
        z_2_perm = permute_dims(z_2)
        d_z_2_perm = self.D(z_2_perm.detach())
        d_tc_loss = 0.5 * (F.cross_entropy(d_z, zeros) +
                           F.cross_entropy(d_z_2_perm, ones))

        # log the losses
        self.log_dict({
            "val_vae_loss": vae_loss,
            "val_vae_recon_loss": vae_recon_loss,
            "val_vae_kld_loss": kld_loss,
            "val_vae_tc_loss": vae_tc_loss,
            "val_d_tc_loss": d_tc_loss,
        })

    def on_validation_epoch_end(self) -> None:
        self.VAE.eval()
        # get the epoch number from trainer
        epoch = self.trainer.current_epoch

        if self.args.dynamic_kld > 0:
            if self.last_recon_loss < self.args.target_recon_loss:
                self.kld_weight_dynamic += 0.0001

        if epoch % self.plot_interval != 0 and epoch != 0:
            return

        self.save_recon_plot()
        self.save_latent_space_plot()

    def save_recon_plot(self, num_recons=64):
        """Save a figure of N reconstructions"""
        # get the length of the training dataset
        dataset = self.trainer.val_dataloaders.dataset
        # get the train dataset's scaler
        scaler = self.train_scaler
        # get a random indices
        indices = torch.randint(0, len(dataset), (num_recons,))
        x, _ = dataset[indices]
        x_recon, mean, logvar, z = self.VAE(x.to(self.device))
        x = x.squeeze(1).cpu().numpy()
        x_recon = x_recon.detach().squeeze(1).cpu().numpy()
        x_scaled = scaler.inverse_transform(x).T
        x_recon_scaled = scaler.inverse_transform(x_recon).T
        # Create a normalization object for colormap
        vmin = min(np.min(x_scaled), np.min(x_recon_scaled))
        vmax = max(np.max(x_scaled), np.max(x_recon_scaled))
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        # create figure
        fig = plt.figure(figsize=(10, 4))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05])
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])
        ax2 = plt.subplot(gs[2])
        # Create the first subplot
        cax0 = ax0.matshow(x_scaled, interpolation='nearest',
                           cmap='viridis', norm=norm)
        ax0.set_title('Ground Truth')
        # Create the second subplot
        cax1 = ax1.matshow(
            x_recon_scaled, interpolation='nearest', cmap='viridis', norm=norm)
        ax1.set_title('Reconstruction')
        # Create a colorbar that is shared between the two subplots
        fig.colorbar(cax1, cax=ax2)
        # save figure to checkpoint folder/recons
        save_dir = os.path.join(self.args.ckpt_path,
                                self.args.ckpt_name, "recons")
        os.makedirs(save_dir, exist_ok=True)
        fig_name = f"recons_{str(self.trainer.current_epoch).zfill(5)}.png"
        plt.savefig(os.path.join(save_dir, fig_name))
        plt.close(fig)

    def save_latent_space_plot(self, batch_size=256):
        """Save a figure of the latent space"""
        # get the length of the training dataset
        dataset = self.trainer.val_dataloaders.dataset
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


class PlMapper(LightningModule):
    def __init__(self, args):
        super(PlMapper, self).__init__()
        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        # self.save_hyperparameters()

        # data params
        self.in_features = args.in_features
        self.out_features = args.out_features
        self.hidden_layers_features = args.hidden_layers_features

        # losses
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.locality_weight = args.locality_weight
        self.cycle_consistency_weight = args.cycle_consistency_weight
        self.cycle_consistency_start = args.cycle_consistency_start
        self.cycle_consistency_warmup_epochs = args.cycle_consistency_warmup_epochs

        # learning rate
        self.lr = args.lr
        self.lr_decay = args.lr_decay

        # logging
        self.plot_interval = args.plot_interval
        self.args = args

        # models
        self.model = LinearProjector(
            in_features=self.in_features, out_features=self.out_features, hidden_layers_features=self.hidden_layers_features)
        self.in_model = args.in_model
        self.out_model = args.out_model
        self.in_model.eval()
        self.out_model.eval()
        self.out_latent_space = args.out_latent_space

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        self.model.train()
        self.in_model.eval()
        self.out_model.eval()
        epoch_idx = self.trainer.current_epoch

        # get the optimizer and scheduler
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()

        # get the batch
        x, _ = batch
        batch_size = x.shape[0]

        # encode with input model
        z_1 = self.in_model.encode(x)

        # project to output space
        z_2 = self.model(z_1)

        # decode with output model
        x_hat = self.out_model.decode(z_2)

        # re-encode with output model
        z_3 = self.out_model.encode(x_hat.detach())

        # LOSSES
        # locality loss
        # TODO: have to do it per axis, since the axes have different meanings
        # TODO: use l1 or mse?
        # TODO: this assumes that z_1 and z_2 have the same dimensions
        num_dims = z_1.shape[1]
        locality_loss = 0
        for dim in range(num_dims):
            locality_loss += self.l1(z_2[:, dim], z_1[:, dim])
        scaled_locality_loss = self.locality_weight * locality_loss

        # cycle consistency loss
        cycle_consistency_loss = self.mse(z_3, z_2)
        cycle_consistency_scale = self.cycle_consistency_weight * \
            min(1.0, (epoch_idx - self.cycle_consistency_start) /
                self.cycle_consistency_warmup_epochs) if epoch_idx > self.cycle_consistency_start else 0
        scaled_cycle_consistency_loss = cycle_consistency_scale * cycle_consistency_loss

        # total loss
        loss = scaled_locality_loss + scaled_cycle_consistency_loss

        # backward pass
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        scheduler.step()

        # log losses
        self.log_dict({
            "locality_loss": locality_loss,
            "cycle_consistency_loss": cycle_consistency_loss,
            "train_loss": loss,
        })

    def on_train_epoch_end(self) -> None:
        self.model.eval()
        # get the epoch number from trainer
        epoch = self.trainer.current_epoch

        if epoch % self.plot_interval != 0 and epoch != 0:
            return

        self.save_latent_space_plot()

    def save_latent_space_plot(self, batch_size=64):
        """Save a figure of the latent space"""
        # get the length of the training dataset
        dataset = self.trainer.train_dataloader.dataset
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, drop_last=False)
        z_all = torch.zeros(
            len(dataset), self.out_features).to(self.device)
        for batch_idx, data in enumerate(loader):
            x, y = data
            x_recon, mean, logvar, z = self.in_model(x.to(self.device))
            z = self.model(z)
            z = z.detach()
            z_all[batch_idx*batch_size: batch_idx*batch_size + batch_size] = z
        z_all = z_all.cpu().numpy()
        # create the figure
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        out_latent_space = self.out_latent_space.cpu().numpy()
        ax.scatter(out_latent_space[:, 0],
                   out_latent_space[:, 1], c="blue")
        ax.scatter(z_all[:, 0], z_all[:, 1], c="red")
        ax.set_title(
            f"Latent space at epoch {self.trainer.current_epoch}")
        # save figure to checkpoint folder/latent
        save_dir = os.path.join(self.args.ckpt_path,
                                self.args.ckpt_name, "latent")
        os.makedirs(save_dir, exist_ok=True)
        fig_name = f"latent_{str(self.trainer.current_epoch).zfill(5)}.png"
        plt.savefig(os.path.join(save_dir, fig_name))
        plt.close(fig)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.lr_decay)
        return [optimizer], [scheduler]
