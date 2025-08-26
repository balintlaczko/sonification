import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .layers import LinearEncoder, LinearDecoder, ResBlock, LinearResBlock, ConvEncoder, ConvDecoder, ConvEncoder1D, ConvDecoder1D, ConvEncoder1DRes, ConvDecoder1DRes, LinearDiscriminator, LinearProjector, LinearDiscriminator_w_dropout, MultiScaleEncoder
from .ddsp import FMSynth
from torchaudio.transforms import MelSpectrogram
from torchaudio.functional import resample
from lightning.pytorch import LightningModule
from ..utils.tensor import permute_dims, midi2frequency, scale
from ..utils.misc import kl_scheduler, ema
from ..utils.dsp import transposition2duration
from .loss import kld_loss, latent_consistency_loss
import matplotlib.pyplot as plt
import os
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import numpy as np
# import pca from sklearn
from sklearn.decomposition import PCA
from tqdm import tqdm
from auraloss.freq import MultiResolutionSTFTLoss
from copy import deepcopy
from collections import OrderedDict
from sys import stderr
import math


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
            in_channels, input_size, layers_channels, input_size)
        self.mu = nn.Linear(input_size, latent_size)
        self.logvar = nn.Linear(input_size, latent_size)
        self.decoder = ConvDecoder(
            latent_size, in_channels, layers_channels[::-1], input_size)

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
    def __init__(self, in_channels, latent_size, kernel_size=3, layers_channels=[16, 32, 64, 128, 256], input_size=64, dropout=0.0):
        super(ConvVAE1D, self).__init__()
        self.encoder = ConvEncoder1D(
            in_channels, input_size, kernel_size, layers_channels, input_size, dropout)
        self.mu = nn.Linear(input_size, latent_size)
        self.logvar = nn.Linear(input_size, latent_size)
        self.decoder = ConvDecoder1D(
            latent_size, in_channels, kernel_size, layers_channels[::-1], input_size, dropout)

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


class ConvVAE1DRes(nn.Module):
    def __init__(self, in_channels, latent_size, layers_channels=[16, 32, 64, 128, 256], input_size=64):
        super(ConvVAE1DRes, self).__init__()
        self.encoder = ConvEncoder1DRes(
            in_channels, input_size, layers_channels, input_size)
        self.mu = nn.Linear(input_size, latent_size)
        self.logvar = nn.Linear(input_size, latent_size)
        self.decoder = ConvDecoder1DRes(
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
        self.save_hyperparameters()

        # data params
        self.in_channels = args.in_channels
        self.input_size = args.img_size
        self.latent_size = args.latent_size
        self.layers_channels = args.layers_channels

        # losses
        self.mse = nn.MSELoss()
        self.kld = kld_loss
        self.recon_weight = args.recon_weight
        self.red_ch_weight = args.red_ch_weight
        self.target_recon_loss = args.target_recon_loss
        self.kld_weight_max = args.kld_weight_max
        self.kld_weight_min = args.kld_weight_min
        self.kld_weight_dynamic = args.kld_weight_min  # initialize to min
        self.dynamic_kld_increment = args.dynamic_kld_increment
        self.kld_start_epoch = args.kld_start_epoch
        self.kld_warmup_epochs = args.kld_warmup_epochs
        self.last_recon_loss = float("inf")  # initialize to infinity
        self.kld_scale = args.kld_weight_min  # just init for sanity check

        self.max_patches_per_batch = args.max_patches_per_batch
        self.n_train_batches = args.n_train_batches

        # learning rates
        self.lr = args.lr
        self.lr_decay = args.lr_decay
        self.lr_scheduler_patience = args.lr_scheduler_patience

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
        # x_1, x_2 = batch
        x_1 = batch
        epoch_idx = self.trainer.current_epoch
        # get training dataset
        dataset = self.trainer.train_dataloader.dataset

        # loop through all patches in the images
        vae_loss = None
        # shuffle dataset.idx2yx
        yx_shuffled = dataset.idx2yx.copy()
        np.random.shuffle(yx_shuffled)
        # cap it if necessary
        yx_shuffled = yx_shuffled[:self.max_patches_per_batch]
        for idx, yx in enumerate(tqdm(yx_shuffled)):
            y, x = yx
            # get the patch
            x_1_patch = x_1[:, :, y:y+self.input_size,
                            x:x+self.input_size]

            # VAE forward pass
            x_recon, mean, logvar, z = self.VAE(x_1_patch)

            # VAE reconstruction loss
            x_recon_r, x_recon_g = x_recon[:, 0, ...], x_recon[:, 1, ...]
            # add back removed axis
            x_recon_r = x_recon_r.unsqueeze(1)
            x_recon_g = x_recon_g.unsqueeze(1)
            x_1_patch_r, x_1_patch_g = x_1_patch[:, 0, ...], x_1_patch[:, 1, ...]
            x_1_patch_r = x_1_patch_r.unsqueeze(1)
            x_1_patch_g = x_1_patch_g.unsqueeze(1)

            vae_recon_loss_r = self.mse(x_recon_r, x_1_patch_r)
            vae_recon_loss_g = self.mse(x_recon_g, x_1_patch_g)
            # vae_recon_loss = self.mse(x_recon, x_1_patch)
            vae_recon_loss = vae_recon_loss_r * self.red_ch_weight + vae_recon_loss_g
            scaled_vae_recon_loss = vae_recon_loss * self.recon_weight

            # VAE KLD loss
            if self.args.dynamic_kld > 0:
                kld_scale = self.kld_weight_dynamic
            else:
                kld_scale = (self.kld_weight_max - self.kld_weight_min) * \
                    min(1.0, (epoch_idx - self.kld_start_epoch) /
                        self.kld_warmup_epochs) + self.kld_weight_min if epoch_idx > self.kld_start_epoch else self.kld_weight_min
            # calculate beta-norm according to beta-vae paper
            kld_scale = kld_scale * self.latent_size / self.input_size
            kld_loss = self.kld(mean, logvar)
            self.kld_scale = kld_scale
            scaled_kld_loss = kld_loss * self.kld_scale

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
            vae_scheduler.step(vae_loss)

        # log the losses
        self.last_recon_loss = vae_recon_loss.item()
        self.log_dict({
            "vae_loss": vae_loss,
            "vae_recon_loss": vae_recon_loss,
            "vae_kld_loss": kld_loss,
            "kld_scale": self.kld_scale,
            "lr": vae_optimizer.param_groups[0]["lr"],
        })
        if self.args.dynamic_kld > 0:
            self.log_dict({"kld_scale": kld_scale})

    def validation_step(self, batch, batch_idx):
        self.VAE.eval()

        # get the batch
        # x_1, x_2 = batch
        x_1 = batch
        # get validation dataset
        dataset = self.trainer.val_dataloaders.dataset

        # loop through all patches in the images
        vae_loss = None
        # shuffle dataset.idx2yx
        yx_shuffled = dataset.idx2yx.copy()
        np.random.shuffle(yx_shuffled)
        # cap it if necessary
        yx_shuffled = yx_shuffled[:self.max_patches_per_batch]
        for idx, yx in enumerate(tqdm(yx_shuffled)):
            y, x = yx
            # get the patch
            x_1_patch = x_1[:, :, y:y+self.input_size,
                            x:x+self.input_size]

            # VAE forward pass
            x_recon, mean, logvar, z = self.VAE(x_1_patch)

            # VAE reconstruction loss
            # print(x_recon.shape, x_recon[:, 0, ...].shape)
            x_recon_r, x_recon_g = x_recon[:, 0, ...], x_recon[:, 1, ...]
            # add back removed axis
            x_recon_r = x_recon_r.unsqueeze(1)
            x_recon_g = x_recon_g.unsqueeze(1)
            x_1_patch_r, x_1_patch_g = x_1_patch[:, 0, ...], x_1_patch[:, 1, ...]
            x_1_patch_r = x_1_patch_r.unsqueeze(1)
            x_1_patch_g = x_1_patch_g.unsqueeze(1)

            vae_recon_loss_r = self.mse(x_recon_r, x_1_patch_r)
            vae_recon_loss_g = self.mse(x_recon_g, x_1_patch_g)
            # vae_recon_loss = self.mse(x_recon, x_1_patch)
            vae_recon_loss = vae_recon_loss_r * self.red_ch_weight + vae_recon_loss_g
            scaled_vae_recon_loss = vae_recon_loss * self.recon_weight

            # VAE KLD loss
            kld_loss = self.kld(mean, logvar)
            scaled_kld_loss = kld_loss * self.kld_scale

            # VAE loss
            if idx == 0:
                vae_loss = scaled_vae_recon_loss + scaled_kld_loss
            else:
                vae_loss += scaled_vae_recon_loss + scaled_kld_loss

        # log the losses
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
                self.kld_weight_dynamic += self.dynamic_kld_increment

        if epoch % self.plot_interval != 0 and epoch != 0:
            return

        self.save_recon_plot()
        # self.save_latent_space_plot()

    def save_recon_plot(self, num_recons=4):
        """Save a figure of N reconstructions"""
        fig, ax = plt.subplots(2, num_recons, figsize=(20, 10))
        # get the length of the training dataset
        dataset = self.trainer.val_dataloaders.dataset
        for i in range(num_recons):
            # get a random index
            idx = torch.randint(0, len(dataset), (1,)).item()
            # x, _ = dataset[idx]
            x = dataset[idx]
            patch_idx = torch.randint(0, len(dataset.idx2yx), (1,)).item()
            _y, _x = dataset.idx2yx[patch_idx]
            x = x[:, _y:_y+self.input_size, _x:_x+self.input_size]
            x_in = x.unsqueeze(0).to(self.device)
            x_recon, mean, logvar, z = self.VAE(x_in)
            # x_recon, z = self.VAE(x_in)
            # x_recon: (B, C, H, W)
            x_recon = x_recon[0, ...].detach().cpu().numpy()
            # x_recon: (C, H, W)
            x = x.detach().cpu().numpy()

            # # split r and g channels
            # x_recon_r, x_recon_g = x_recon[:2, ...]
            # x_r, x_g = x[:2, ...]
            # # inverse scale the images
            # x_recon_r, x_recon_g = dataset.scale_inv(
            #     x_recon_r[None, ...], x_recon_g[None, ...])
            # x_r, x_g = dataset.scale_inv(x_r[None, ...], x_g[None, ...])
            # # merge the images
            # x_recon = np.concatenate((x_recon_r, x_recon_g), axis=0)
            # x = np.concatenate((x_r, x_g), axis=0)

            # append zeros for blue channel
            zeros = np.zeros_like(x_recon[0, ...][None, ...])
            x_recon = np.concatenate(
                (x_recon, zeros), axis=0)
            x = np.concatenate((x, zeros), axis=0)

            # from channels-first to channels-last
            x_recon = np.moveaxis(x_recon, 0, -1)
            x = np.moveaxis(x, 0, -1)

            # normalize the images
            x_recon = (x_recon - x_recon.min()) / \
                (x_recon.max() - x_recon.min())
            x = (x - x.min()) / (x.max() - x.min())

            # plot ground truth image
            ax[0, i].imshow(x)
            ax[0, i].set_title(f"GT_{idx}")
            # plot reconstruction
            ax[1, i].imshow(x_recon)
            ax[1, i].set_title(f"Recon_{idx}")
        # save figure to checkpoint folder/recons
        save_dir = os.path.join(self.args.ckpt_path,
                                self.args.ckpt_name, "recons")
        os.makedirs(save_dir, exist_ok=True)
        fig_name = f"recons_{str(self.trainer.current_epoch).zfill(5)}.png"
        plt.savefig(os.path.join(save_dir, fig_name))
        plt.close(fig)

    def save_latent_space_plot(self, batch_size=1024):
        """Save a figure of the latent space"""
        # get the length of the training dataset
        dataset = self.trainer.val_dataloaders.dataset
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, drop_last=False)
        z_all = torch.zeros(
            len(dataset), self.args.latent_size).to(self.device)
        for batch_idx, data in enumerate(loader):
            # x, y = data
            x = data
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
            self.VAE.parameters(), lr=self.lr, betas=(0.9, 0.999))
        vae_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            vae_optimizer, mode="min", factor=self.lr_decay, patience=self.lr_scheduler_patience * self.max_patches_per_batch * self.n_train_batches, verbose=True)
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
        self.save_hyperparameters()  # this will save args in the checkpoint

        # data params
        self.in_channels = args.in_channels
        self.input_size = args.img_size

        # vae params
        self.latent_size = args.latent_size
        self.kernel_size = args.kernel_size
        self.layers_channels = args.layers_channels
        # self.layers_channels = [args.vae_channels] * args.vae_num_layers
        self.vae_dropout = args.vae_dropout

        # d params (for tc loss)
        self.d_hidden_size = args.d_hidden_size
        self.d_num_layers = args.d_num_layers
        self.d_dropout = args.d_dropout

        # d2 params (for reconstruction loss)
        self.d2_hidden_size = args.d_hidden_size
        self.d2_num_layers = args.d_num_layers
        self.d2_dropout = args.d_dropout

        # losses
        # recon loss
        # self.recon_loss = nn.MSELoss()
        self.recon_loss = nn.L1Loss()
        self.recon_weight = args.recon_weight
        # self.mmd_loss = MMDloss()

        # kld loss
        self.kld = kld_loss
        self.last_recon_loss = float("inf")  # initialize to infinity
        self.kld_weight_max = args.kld_weight_max
        self.kld_weight_min = args.kld_weight_min
        self.kld_weight_dynamic = args.kld_weight_min  # initialize to min
        self.dynamic_kld_increment = args.dynamic_kld_increment
        self.auto_dkld_scale = args.auto_dkld_scale
        self.kld_start_epoch = args.kld_start_epoch
        self.kld_warmup_epochs = args.kld_warmup_epochs
        self.cycling_kld = args.cycling_kld
        self.cycling_kld_period = args.cycling_kld_period
        self.cycling_kld_ramp_up_phase = args.cycling_kld_ramp_up_phase
        self.kld_scale = args.kld_weight_min  # just init for sanity check
        self.kld_decay = args.kld_decay

        # tc loss
        self.tc_weight = args.tc_weight
        self.tc_start_epoch = args.tc_start_epoch
        self.tc_warmup_epochs = args.tc_warmup_epochs
        self.tc_weight_dynamic = args.tc_weight  # initialize to tc_weight
        self.dynamic_tc_increment = args.dynamic_tc_increment
        self.auto_dtc_scale = args.auto_dtc_scale
        self.tc_scale = args.tc_weight  # just init for sanity check
        self.d_criterion = nn.BCELoss()

        # latent consistency loss
        self.latent_consistency_weight = args.latent_consistency_weight

        self.ema_alpha = args.ema_alpha

        # learning rates
        self.lr_vae = args.lr_vae
        self.lr_decay_vae = args.lr_decay_vae
        self.lr_d = args.lr_d
        self.lr_decay_d = args.lr_decay_d

        # logging
        self.plot_interval = args.plot_interval
        self.args = args

        # models
        self.VAE = ConvVAE1D(self.in_channels, self.latent_size, self.kernel_size,
                             self.layers_channels, self.input_size, self.vae_dropout)
        # self.VAE = ConvVAE1DRes(self.in_channels, self.latent_size,
        #                      self.layers_channels, self.input_size)

        self.D = LinearDiscriminator_w_dropout(
            self.latent_size, self.d_hidden_size, 1, self.d_num_layers, self.d_dropout)

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
        # batch_size = x_1.shape[0]

        # add small amount of noise to the input
        x_1 = x_1 + torch.randn_like(x_1) * 0.001
        x_2 = x_2 + torch.randn_like(x_2) * 0.001

        # VAE forward pass
        x_recon, mean, logvar, z = self.VAE(x_1)
        mean_2, logvar_2 = self.VAE.encode(x_2)
        z_2 = self.VAE.reparameterize(mean_2, logvar_2)
        z_2_perm = permute_dims(z_2)

        # Discriminator forward pass (for TC loss)
        z_judgement = self.D(z.detach())
        z_2_perm_judgement = self.D(z_2_perm.detach())

        # Compute discriminator loss
        original_z_loss = self.d_criterion(
            z_judgement, torch.ones_like(z_judgement))  # label 1 = the original z
        shuffled_z_loss = self.d_criterion(z_2_perm_judgement, torch.zeros_like(
            z_2_perm_judgement))  # label 0 = the shuffled z
        d_tc_loss = (original_z_loss + shuffled_z_loss) / 2

        # Discriminator backward pass
        d_optimizer.zero_grad()
        self.manual_backward(d_tc_loss, retain_graph=True)

        # VAE reconstruction loss
        vae_recon_loss = self.recon_loss(x_recon, x_1)
        scaled_vae_recon_loss = vae_recon_loss * self.recon_weight

        # VAE KLD loss
        if self.args.dynamic_kld > 0:
            kld_scale = self.kld_weight_dynamic
        elif self.cycling_kld > 0:
            kld_scale = kl_scheduler(epoch=epoch_idx, cycle_period=self.cycling_kld_period,
                                     ramp_up_phase=self.cycling_kld_ramp_up_phase) * self.kld_weight_max
        else:
            kld_scale = (self.kld_weight_max - self.kld_weight_min) * \
                min(1.0, (epoch_idx - self.kld_start_epoch) /
                    self.kld_warmup_epochs) + self.kld_weight_min if epoch_idx > self.kld_start_epoch else self.kld_weight_min
        # calculate beta-norm according to beta-vae paper
        kld_scale = kld_scale * self.latent_size / self.input_size
        # use kld decay after kld warmup
        if epoch_idx > self.kld_start_epoch + self.kld_warmup_epochs:
            kld_scale = kld_scale * \
                (self.kld_decay ** (epoch_idx -
                 self.kld_start_epoch - self.kld_warmup_epochs))
        kld_scale = max(kld_scale, self.kld_weight_min *
                        self.latent_size / self.input_size)
        kld_loss = self.kld(mean, logvar)
        self.kld_scale = kld_scale
        scaled_kld_loss = kld_loss * self.kld_scale

        # VAE TC loss
        if self.args.dynamic_tc > 0:
            vae_tc_scale = self.tc_weight_dynamic
        else:
            vae_tc_scale = self.tc_weight * \
                min(1.0, (epoch_idx - self.tc_start_epoch) /
                    self.tc_warmup_epochs) if epoch_idx > self.tc_start_epoch else 0

        vae_tc_loss = self.d_criterion(z_judgement, torch.zeros_like(
            z_judgement))  # label 0 = what we used for the shuffled z
        self.tc_scale = vae_tc_scale
        scaled_vae_tc_loss = vae_tc_loss * self.tc_scale

        # VAE latent consistency loss
        vae_lc_loss = 0
        scaled_vae_lc_loss = 0
        if self.latent_consistency_weight > 0:
            shift_vector = torch.randn(
                self.latent_size, device=self.device) * 0.1
            vae_lc_loss = latent_consistency_loss(
                self.encode,
                self.decode,
                x_1,
                shift_vector)
            scaled_vae_lc_loss = vae_lc_loss * self.latent_consistency_weight

        # VAE loss
        vae_loss = scaled_vae_recon_loss + scaled_kld_loss + \
            scaled_vae_tc_loss + scaled_vae_lc_loss

        # VAE backward pass
        vae_optimizer.zero_grad()
        self.manual_backward(vae_loss)

        # Discriminator Optimizer & Scheduler step
        d_optimizer.step()
        d_scheduler.step()

        # VAE Optimizer & Scheduler step
        vae_optimizer.step()
        vae_scheduler.step()

        # log the losses
        self.last_recon_loss = vae_recon_loss.item()
        self.log_dict({
            "vae_loss": vae_loss,
            "vae_recon_loss": vae_recon_loss,
            "vae_kld_loss": kld_loss,
            "vae_tc_loss": vae_tc_loss,
            "d_tc_loss": d_tc_loss,
            "kld_scale": self.kld_scale,
            "tc_scale": self.tc_scale,
            "vae_lc_loss": vae_lc_loss,
        })

    def validation_step(self, batch, batch_idx):
        self.VAE.eval()
        self.D.eval()

        # get the batch
        x_1, x_2 = batch
        # batch_size = x_1.shape[0]

        # add small amount of noise to the input
        x_1 = x_1 + torch.randn_like(x_1) * 0.001
        x_2 = x_2 + torch.randn_like(x_2) * 0.001

        # VAE forward pass
        x_recon, mean, logvar, z = self.VAE(x_1)
        mean_2, logvar_2 = self.VAE.encode(x_2)
        z_2 = self.VAE.reparameterize(mean_2, logvar_2)
        z_2_perm = permute_dims(z_2)

        # Discriminator forward pass
        z_judgement = self.D(z.detach())
        z_2_perm_judgement = self.D(z_2_perm.detach())

        # Compute discriminator loss
        original_z_loss = self.d_criterion(
            z_judgement, torch.ones_like(z_judgement))  # label 1 = the original z
        shuffled_z_loss = self.d_criterion(z_2_perm_judgement, torch.zeros_like(
            z_2_perm_judgement))  # label 0 = the shuffled z
        d_tc_loss = original_z_loss + shuffled_z_loss

        # VAE reconstruction loss
        vae_recon_loss = self.recon_loss(x_recon, x_1)
        scaled_vae_recon_loss = vae_recon_loss * self.recon_weight

        # VAE KLD loss
        kld_loss = self.kld(mean, logvar)
        scaled_kld_loss = kld_loss * self.kld_scale

        # VAE TC loss
        vae_tc_loss = self.d_criterion(z_judgement, torch.zeros_like(
            z_judgement))  # label 0 = what we used for the shuffled z
        scaled_vae_tc_loss = vae_tc_loss * self.tc_scale

        # VAE latent consistency loss
        vae_lc_loss = 0
        scaled_vae_lc_loss = 0
        if self.latent_consistency_weight > 0:
            shift_vector = torch.randn(
                self.latent_size, device=self.device) * 0.1
            vae_lc_loss = latent_consistency_loss(
                self.encode,
                self.decode,
                x_1,
                shift_vector)
            scaled_vae_lc_loss = vae_lc_loss * self.latent_consistency_weight

        # VAE loss
        vae_loss = scaled_vae_recon_loss + scaled_kld_loss + \
            scaled_vae_tc_loss + scaled_vae_lc_loss

        # log the losses
        self.log_dict({
            "val_vae_loss": vae_loss,
            "val_vae_recon_loss": vae_recon_loss,
            "val_vae_kld_loss": kld_loss,
            "val_vae_tc_loss": vae_tc_loss,
            "val_d_tc_loss": d_tc_loss,
            "val_vae_lc_loss": vae_lc_loss,
        })

    def on_validation_epoch_end(self) -> None:
        self.VAE.eval()
        # get the epoch number from trainer
        epoch = self.trainer.current_epoch

        # adjust dynamic kld increment or auto kld scale
        if self.args.dynamic_kld > 0:
            if self.auto_dkld_scale > 0:  # in auto scale mode we scale in both directions
                last_val = self.kld_weight_dynamic
                new_val = self.kld_weight_dynamic * \
                    (1 + self.args.target_recon_loss - self.last_recon_loss)
                self.kld_weight_dynamic = np.clip(ema(
                    last_val, new_val, alpha=self.ema_alpha), self.kld_weight_min, self.kld_weight_max)
            # in manual mode we increment the kld weight when the recon loss is below the target
            elif self.last_recon_loss < self.args.target_recon_loss:
                self.kld_weight_dynamic += self.dynamic_kld_increment

        # adjust dynamic tc increment or auto tc scale
        if self.args.dynamic_tc > 0:
            if self.auto_dtc_scale > 0:
                last_val = self.tc_weight_dynamic
                new_val = self.tc_weight_dynamic * \
                    (1 + self.args.target_recon_loss - self.last_recon_loss)
                self.tc_weight_dynamic = np.clip(
                    ema(last_val, new_val, alpha=self.ema_alpha), self.tc_weight, self.tc_weight*10)
            # in manual mode we increment the tc weight when the recon loss is below the target
            elif self.last_recon_loss < self.args.target_recon_loss:
                self.tc_weight_dynamic += self.dynamic_tc_increment

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
        ax0.matshow(x_scaled, interpolation='nearest',
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

        # if z dim > 2 then PCA to 2D
        if self.args.latent_size > 2:
            pca = PCA(n_components=2)
            z_all = pca.fit_transform(z_all)

        # create the figure
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        ax.scatter(z_all[:, 0], z_all[:, 1], c="blue")

        percentiles = [5, 10, 15, 20, 25, 30]
        color_labels = ["red", "green", "yellow", "purple", "orange", "cyan"]
        df = dataset.df
        # get all unique pitch values sorter from low to high
        pitch_values = np.sort(df['pitch'].unique())
        # get the percentiles for the pitch values
        n_pitches = len(pitch_values)
        # get all unique loudness values sorter from low to high
        loudness_values = np.sort(df['loudness'].unique())
        # get the percentiles for the loudness values
        n_loudnesses = len(loudness_values)

        for idx, percentile in enumerate(percentiles):
            percentile_low = percentile
            percentile_high = 100 - percentile

            pitch_low = pitch_values[int(n_pitches * percentile_low / 100)]
            pitch_high = pitch_values[int(n_pitches * percentile_high / 100)]

            loudness_low = loudness_values[int(
                n_loudnesses * percentile_low / 100)]
            loudness_high = loudness_values[int(
                n_loudnesses * percentile_high / 100)]

            # get the sample with the lowest pitch and lowest loudness
            df_lowest_pitch = df[df['pitch'] == pitch_low]
            # get the closest loudness to the lowest loudness
            df_lowest_pitch_lowest_loudness = df_lowest_pitch.iloc[(
                df_lowest_pitch['loudness'] - loudness_low).abs().argsort()[:1]]
            # now highest pitch lowest loudness
            df_highest_pitch = df[df['pitch'] == pitch_high]
            # get the closest loudness to the lowest loudness
            df_highest_pitch_lowest_loudness = df_highest_pitch.iloc[(
                df_highest_pitch['loudness'] - loudness_low).abs().argsort()[:1]]
            # now lowest pitch highest loudness
            df_lowest_pitch = df[df['pitch'] == pitch_low]
            # get the closest loudness to the highest loudness
            df_lowest_pitch_highest_loudness = df_lowest_pitch.iloc[(
                df_lowest_pitch['loudness'] - loudness_high).abs().argsort()[:1]]
            # now highest pitch highest loudness
            df_highest_pitch = df[df['pitch'] == pitch_high]
            # get the closest loudness to the highest loudness
            df_highest_pitch_highest_loudness = df_highest_pitch.iloc[(
                df_highest_pitch['loudness'] - loudness_high).abs().argsort()[:1]]

            idx_top_left = df_highest_pitch_highest_loudness.index[0]
            idx_bottom_left = df_lowest_pitch_highest_loudness.index[0]
            idx_top_right = df_highest_pitch_lowest_loudness.index[0]
            idx_bottom_right = df_lowest_pitch_lowest_loudness.index[0]
            # get the row number of the sample
            idx_top_left = np.where(df.index == idx_top_left)[0][0]
            idx_bottom_left = np.where(df.index == idx_bottom_left)[0][0]
            idx_top_right = np.where(df.index == idx_top_right)[0][0]
            idx_bottom_right = np.where(df.index == idx_bottom_right)[0][0]

            ax.scatter(z_all[idx_top_left, 0],
                       z_all[idx_top_left, 1], c=color_labels[idx], s=100)
            ax.scatter(z_all[idx_bottom_left, 0],
                       z_all[idx_bottom_left, 1], c=color_labels[idx], s=100)
            ax.scatter(z_all[idx_top_right, 0],
                       z_all[idx_top_right, 1], c=color_labels[idx], s=100)
            ax.scatter(z_all[idx_bottom_right, 0],
                       z_all[idx_bottom_right, 1], c=color_labels[idx], s=100)

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
        vae_optimizer = torch.optim.AdamW(
            self.VAE.parameters(), lr=self.lr_vae, betas=(0.9, 0.999))
        d_optimizer = torch.optim.AdamW(
            self.D.parameters(), lr=self.lr_d, betas=(0.5, 0.999))
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


class FMParamEstimator(nn.Module):
    def __init__(
            self,
            latent_size=128,
            encoder_kernels=[4, 4],
            input_dim_h=64,
            input_dim_w=29,
            n_res_block=2,
            n_res_channel=32,
            stride=4,
            ):
        super(FMParamEstimator, self).__init__()

        self.chans_per_group = 16

        self.encoder = MultiScaleEncoder(
            in_channel=1,
            channel=latent_size,
            n_res_block=n_res_block,
            n_res_channel=n_res_channel,
            stride=stride,
            kernels=encoder_kernels,
            input_dim_h=input_dim_h,
            input_dim_w=input_dim_w,
        )
        self.post_encoder = nn.Sequential(
            nn.Conv2d(latent_size, latent_size, 3, 2, 1),
            nn.GroupNorm(latent_size // self.chans_per_group, latent_size),
            nn.LeakyReLU(0.2),
            nn.Conv2d(latent_size, latent_size, 3, 2, 1),
            nn.GroupNorm(latent_size // self.chans_per_group, latent_size),
            nn.LeakyReLU(0.2),
            nn.Conv2d(latent_size, latent_size, 3, 2, 1),
            nn.GroupNorm(latent_size // self.chans_per_group, latent_size),
            nn.LeakyReLU(0.2),
            nn.Conv2d(latent_size, latent_size, 3, 2, 1),
            nn.GroupNorm(latent_size // self.chans_per_group, latent_size),
            nn.LeakyReLU(0.2),
            nn.Conv2d(latent_size, latent_size, 3, 2, 1),
            nn.GroupNorm(latent_size // self.chans_per_group, latent_size),
            nn.LeakyReLU(0.2),
            nn.Conv2d(latent_size, latent_size, 3, 2, 1),
            nn.GroupNorm(latent_size // self.chans_per_group, latent_size),
            nn.LeakyReLU(0.2),
        )

        self.mlp = nn.Sequential(
            nn.Linear(latent_size * (latent_size // 64), 128),
            nn.GroupNorm(128 // self.chans_per_group, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.GroupNorm(64 // self.chans_per_group, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.GroupNorm(32 // self.chans_per_group, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 16),
            nn.GroupNorm(16 // self.chans_per_group, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print("x shape: ", x.shape)
        x = self.encoder(x)
        # print("encoder x shape: ", x.shape)
        x = self.post_encoder(x)
        # print("post_encoder x shape: ", x.shape)
        x = x.view(x.size(0), -1)
        # print("post_encoder x view shape: ", x.shape)
        x = self.mlp(x)
        # print("mlp x shape: ", x.shape)
        return x



class PlFMParamEstimator(LightningModule):
    def __init__(self, args):
        super(PlFMParamEstimator, self).__init__()
        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        self.save_hyperparameters()

        self.sr = args.sr
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.lr_decay = args.lr_decay
        self.warmup_epochs = args.warmup_epochs
        self.param_loss_weight_start = args.param_loss_weight_start
        self.param_loss_weight = args.param_loss_weight_start # initialize to start
        self.param_loss_weight_end = args.param_loss_weight_end
        self.param_loss_weight_ramp_start_epoch = args.param_loss_weight_ramp_start_epoch
        self.param_loss_weight_ramp_end_epoch = args.param_loss_weight_ramp_end_epoch
        self.n_samples = args.length_samps
        self.n_fft = args.n_fft
        self.spectrogram_w = self.n_samples // (self.n_fft // 2) + 1
        self.max_harm_ratio = args.max_harm_ratio
        self.max_mod_idx = args.max_mod_idx
        self.logdir = args.logdir

        # models
        self.input_synth = FMSynth(sr=self.sr)
        self.input_synth.eval()
        self.mel_spectrogram = MelSpectrogram(
            sample_rate=self.sr,
            n_fft=self.n_fft,
            f_min=args.f_min,
            f_max=args.f_max,
            n_mels=args.n_mels,
            power=args.power,
            normalized=args.normalized > 0,
        )
        self.model = FMParamEstimator(
            latent_size=args.latent_size,
            encoder_kernels=args.encoder_kernels,
            input_dim_h=args.n_mels,
            input_dim_w=self.spectrogram_w,
            n_res_block=args.n_res_block,
            n_res_channel=args.n_res_channel,
            stride=4,
        )
        def init_weights_kaiming(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):  # Apply to conv and linear layers
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        self.model.apply(init_weights_kaiming)

        self.output_synth = FMSynth(sr=self.sr)

        # mss loss
        self.mss_loss = MultiResolutionSTFTLoss(
            fft_sizes=[1024, 2048],
            hop_sizes=[256, 512],
            win_lengths=[1024, 2048],
            scale="mel",
            n_bins=128,
            sample_rate=self.sr,
            perceptual_weighting=True,
        )
        self.mss_loss.eval()


    def sample_fm_params(self, batch_size):
        pitches_norm = torch.rand(batch_size, requires_grad=False, device=self.device)
        pitches = scale(pitches_norm, 0, 1, 38, 86)
        freqs = midi2frequency(pitches)
        ratios_norm = torch.rand(batch_size, requires_grad=False, device=self.device)
        ratios = ratios_norm * self.max_harm_ratio
        indices_norm = torch.rand(batch_size, requires_grad=False, device=self.device)
        indices = indices_norm * self.max_mod_idx
        # stack norm params together
        norm_params = torch.stack([pitches_norm, ratios_norm, indices_norm], dim=1)
        # now repeat on the samples dimension
        freqs = freqs.unsqueeze(1).repeat(1, self.sr)
        ratios = ratios.unsqueeze(1).repeat(1, self.sr)
        indices = indices.unsqueeze(1).repeat(1, self.sr)
        return norm_params, freqs, ratios, indices
    

    def scale_predicted_params(self, predicted_params):
        # separate the param dim
        pitches = predicted_params[:, 0]
        ratios = predicted_params[:, 1]
        indices = predicted_params[:, 2]
        # clamp everything to 0..1
        pitches = torch.clamp(pitches, 0, 1)
        ratios = torch.clamp(ratios, 0, 1)
        indices = torch.clamp(indices, 0, 1)
        # scale to the correct range
        pitches = scale(pitches, 0, 1, 38, 86)
        freqs = midi2frequency(pitches)
        ratios = scale(ratios, 0, 1, 0, self.max_harm_ratio)
        indices = scale(indices, 0, 1, 0, self.max_mod_idx)
        # re-stack them
        out = torch.cat([freqs.unsqueeze(1), ratios.unsqueeze(1), indices.unsqueeze(1)], dim=1)
        return out


    def forward(self, x):
        # # scale input to -1 1
        # x = (x - x.min()) / (x.max() - x.min())
        # x = x * 2 - 1
        in_wf = x.unsqueeze(1)
        # get the mel spectrogram
        in_spec = self.mel_spectrogram(in_wf)
        # normalize it
        in_spec = scale(in_spec, in_spec.min(), in_spec.max(), 0, 1)
        # predict the params
        norm_predicted_params = self.model(in_spec)
        # scale the predicted params
        predicted_params = self.scale_predicted_params(norm_predicted_params)
        return predicted_params, norm_predicted_params


    def training_step(self, batch, batch_idx):
        # torch.autograd.set_detect_anomaly(True)
        self.model.train()

        # get the optimizer and scheduler
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()

        # get the batch
        norm_params, freqs, ratios, indices = self.sample_fm_params(self.batch_size)
        x = self.input_synth(freqs, ratios, indices).detach()
        in_wf = x.unsqueeze(1)
        # select a random slice of self.n_samples
        start_idx = torch.randint(0, self.sr - self.n_samples, (1,))
        x = x[:, start_idx:start_idx + self.n_samples]
        # add random phase flip
        phase_flip = torch.rand(self.batch_size, 1, device=self.device)
        phase_flip = torch.where(phase_flip > 0.5, 1, -1)
        x = x * phase_flip
        # add random noise
        noise = torch.randn_like(x, device=self.device)
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        noise = noise * 2 - 1
        noise_coeff = torch.rand(self.batch_size, 1, device=self.device) * 0.001
        noise = noise * noise_coeff
        x = x + noise
        # # rescale to -1 1
        # x = (x - x.min()) / (x.max() - x.min())
        # x = x * 2 - 1
        in_wf_slice = x.unsqueeze(1)

        # forward pass
        # get the mel spectrogram
        in_spec = self.mel_spectrogram(in_wf_slice.detach())
        # normalize per sample (and channel) over spatial dims (H, W)
        reduce_dims = (2, 3)
        mins = in_spec.amin(dim=reduce_dims, keepdim=True)
        maxs = in_spec.amax(dim=reduce_dims, keepdim=True)
        den = (maxs - mins).clamp_min(torch.finfo(in_spec.dtype).eps)
        in_spec = (in_spec - mins) / den
        # predict the params
        norm_predicted_params = self.model(in_spec)
        # scale the predicted params
        predicted_params = self.scale_predicted_params(norm_predicted_params)
        # now repeat on the samples dimension
        predicted_freqs = predicted_params[:, 0].unsqueeze(1).repeat(1, self.sr)
        predicted_ratios = predicted_params[:, 1].unsqueeze(1).repeat(1, self.sr)
        predicted_indices = predicted_params[:, 2].unsqueeze(1).repeat(1, self.sr)

        # generate the output
        y = self.output_synth(predicted_freqs, predicted_ratios, predicted_indices)
        out_wf = y.unsqueeze(1)

        # loss: MSS + param loss
        # param loss
        # param_loss = F.mse_loss(norm_predicted_params, norm_params.detach())
        # param_loss = F.huber_loss(norm_predicted_params, norm_params.detach(), delta=0.01)
        param_loss = F.l1_loss(norm_predicted_params, norm_params.detach())
        # mss loss
        mss_loss = self.mss_loss(out_wf, in_wf)
        # calculate current param loss weight
        current_epoch = self.trainer.current_epoch
        if current_epoch < self.param_loss_weight_ramp_start_epoch:
            param_loss_weight = self.param_loss_weight_start
        elif current_epoch > self.param_loss_weight_ramp_end_epoch:
            param_loss_weight = self.param_loss_weight_end
        else:
            param_loss_weight = self.param_loss_weight_start + (self.param_loss_weight_end - self.param_loss_weight_start) * \
                (current_epoch - self.param_loss_weight_ramp_start_epoch) / \
                (self.param_loss_weight_ramp_end_epoch - self.param_loss_weight_ramp_start_epoch)
        self.param_loss_weight = param_loss_weight
        loss = (param_loss * self.param_loss_weight) + mss_loss

        # backward pass
        optimizer.zero_grad()
        self.manual_backward(loss)
        # clip gradients
        self.clip_gradients(optimizer, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        optimizer.step()
        scheduler.step(loss.item())

        # log losses
        self.log_dict({
            "train_loss": loss,
            "param_loss": param_loss,
            "mss_loss": mss_loss,
            "lr": scheduler.get_last_lr()[0],
            "param_loss_weight": self.param_loss_weight,
        },
        prog_bar=True)

    def on_train_epoch_end(self):
        epoch = self.trainer.current_epoch
        scheduler = self.lr_schedulers()
        # Check if this is a ReduceLROnPlateau scheduler
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if epoch == 500:
                # Dynamically increase patience at epoch 500
                scheduler.patience = 60 * 1000
                scheduler.num_bad_epochs = 0
                print(f"Patience changed to {scheduler.patience} at epoch {epoch}")
            if epoch == 700:
                scheduler.patience = 70 * 1000
                scheduler.num_bad_epochs = 0
                print(f"Patience changed to {scheduler.patience} at epoch {epoch}")
            if epoch == 900:
                scheduler.patience = 80 * 1000
                scheduler.num_bad_epochs = 0
                print(f"Patience changed to {scheduler.patience} at epoch {epoch}")

    def on_train_batch_start(self, batch, batch_idx):
        epoch = self.trainer.current_epoch
        if epoch < self.warmup_epochs:
            lr_scale = min(1.0, (epoch + 1) / self.warmup_epochs)
            for pg in self.trainer.optimizers[0].param_groups:
                pg["lr"] = self.lr * lr_scale

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=self.lr_decay, patience=50000)
        return [optimizer], [scheduler]
    


class MelSpecEncoder(nn.Module):
    def __init__(
            self,
            input_width=512,
            encoder_channels=128,
            encoder_kernels=[4, 4],
            n_res_block=2,
            n_res_channel=32,
            stride=4,
            latent_size=8,
            dropout=0.2,
            ):
        super().__init__()

        self.chans_per_group = 16

        self.encoder = MultiScaleEncoder(
            in_channel=1,
            channel=encoder_channels,
            n_res_block=n_res_block,
            n_res_channel=n_res_channel,
            stride=stride,
            kernels=encoder_kernels,
            input_dim_h=0,
            input_dim_w=0,
        )

        encoded_width = input_width // stride # 128
        target_width = 2
        num_blocks = int(np.log2(encoded_width)) - int(np.log2(target_width)) # 7 - 1 = 6

        post_encoder_blocks = []
        post_encoder_block = [
            nn.Conv2d(encoder_channels, encoder_channels, 3, 2, 1),
            nn.GroupNorm(encoder_channels // self.chans_per_group, encoder_channels),
            nn.LeakyReLU(0.2),
        ]
        for _ in range(num_blocks):
            post_encoder_blocks.extend(post_encoder_block)

        self.post_encoder = nn.Sequential(*post_encoder_blocks)

        post_encoder_n_features = encoder_channels * target_width # 256
        target_n_features = latent_size * 2 # 16
        mlp_layers = []
        num_mlp_blocks = int(np.log2(post_encoder_n_features) - np.log2(target_n_features)) # 256 -> 16 = 4 blocks
        mlp_layers_features = [post_encoder_n_features // (2 ** i) for i in range(num_mlp_blocks + 1)]
        for i in range(num_mlp_blocks):
            block = [
                nn.Linear(mlp_layers_features[i], mlp_layers_features[i + 1]),
                nn.GroupNorm(mlp_layers_features[i + 1] // self.chans_per_group, mlp_layers_features[i + 1]),
                nn.LeakyReLU(0.2),
            ]
            mlp_layers.extend(block)

        self.mlp = nn.Sequential(*mlp_layers)

        self.mu = nn.Linear(target_n_features, latent_size)
        self.logvar = nn.Linear(target_n_features, latent_size)

    def forward(self, x):
        # print("x shape: ", x.shape)
        x = self.encoder(x)
        # print("encoder x shape: ", x.shape)
        x = self.post_encoder(x)
        # print("post_encoder x shape: ", x.shape)
        x = x.view(x.size(0), -1)
        # print("post_encoder x view shape: ", x.shape)
        x = self.mlp(x)
        # print("mlp x shape: ", x.shape)
        mu = self.mu(x)
        logvar = self.logvar(x)
        # print("mu shape: ", mu.shape)
        # print("logvar shape: ", logvar.shape)
        return mu, logvar
    

class ParamDecoder(nn.Module):
    def __init__(
            self,
            decoder_features=128,
            n_res_block=2,
            n_res_features=32,
            latent_size=8,
            n_params=3,
            dropout=0.2,
            ):
        super().__init__()

        self.chans_per_group = 16

        pre_decoder_num_blocks = int(np.log2(decoder_features) - np.log2(latent_size))
        pre_decoder_blocks = []
        pre_decoder_layers_features = [latent_size * (2 ** i) for i in range(pre_decoder_num_blocks + 1)]
        for i in range(pre_decoder_num_blocks):
            block = [
                nn.Linear(pre_decoder_layers_features[i], pre_decoder_layers_features[i + 1]),
                nn.GroupNorm(pre_decoder_layers_features[i + 1] // self.chans_per_group, pre_decoder_layers_features[i + 1]),
                nn.LeakyReLU(0.2),
            ]
            pre_decoder_blocks.extend(block)

        self.pre_decoder = nn.Sequential(*pre_decoder_blocks)

        res_blocks = [LinearResBlock(decoder_features, n_res_features) for _ in range(n_res_block)]
        self.res_blocks = nn.Sequential(*res_blocks)

        post_decoder_target_n_features = 16
        post_decoder_num_blocks = int(np.log2(decoder_features) - np.log2(post_decoder_target_n_features))
        post_decoder_blocks = []
        post_decoder_layers_features = [decoder_features // (2 ** i) for i in range(post_decoder_num_blocks + 1)]
        for i in range(post_decoder_num_blocks):
            block = [
                nn.Linear(post_decoder_layers_features[i], post_decoder_layers_features[i + 1]),
                nn.GroupNorm(post_decoder_layers_features[i + 1] // self.chans_per_group, post_decoder_layers_features[i + 1]),
                nn.LeakyReLU(0.2),
            ]
            post_decoder_blocks.extend(block)
        post_decoder_blocks.extend([
            nn.Linear(post_decoder_target_n_features, n_params),
            # nn.Sigmoid()
            # nn.ReLU()
            nn.Identity()
        ])

        self.post_decoder = nn.Sequential(*post_decoder_blocks)

    def forward(self, x):
        # print("x shape: ", x.shape)
        x = self.pre_decoder(x)
        # print("pre_decoder x shape: ", x.shape)
        x = self.res_blocks(x)
        # print("res_blocks x shape: ", x.shape)
        x = self.post_decoder(x)
        # print("post_decoder x shape: ", x.shape)
        return x


class ImageDecoder(nn.Module):
    def __init__(
            self,
            output_width=512,
            output_channels=1,
            decoder_channels=128,
            n_res_block=2,
            n_res_channel=64,
            latent_size=16,
            ):
        super().__init__()

        self.chans_per_group = 16 if decoder_channels >= 32 else decoder_channels // 2

        # mlp it up to 64, then reshape to 8x8, convtranspose to output_width, do resblocks, predict
        target_n_features = 64
        num_mlp_blocks = int(np.log2(target_n_features) - np.log2(latent_size))  # 64 -> 16 = 3 blocks
        mlp_layers = []
        mlp_layers_features = [latent_size * (2 ** i) for i in range(num_mlp_blocks + 1)]
        for i in range(num_mlp_blocks):
            block = [
                nn.Linear(mlp_layers_features[i], mlp_layers_features[i + 1]),
                nn.GroupNorm(mlp_layers_features[i + 1] // self.chans_per_group, mlp_layers_features[i + 1]),
                nn.LeakyReLU(0.2),
            ]
            mlp_layers.extend(block)

        self.mlp = nn.Sequential(*mlp_layers)

        reshaped_width = int(np.sqrt(target_n_features))  # 8
        reshape_layers = [nn.Unflatten(1, (1, reshaped_width, reshaped_width))]
        reshape_num_blocks = int(np.log2(decoder_channels))
        for i in range(reshape_num_blocks):
            in_channels = 2 ** i
            out_channels = 2 ** (i + 1)
            num_groups = out_channels // self.chans_per_group if out_channels // self.chans_per_group >= 2 else out_channels // 2
            reshape_layers.extend([
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.GroupNorm(num_groups, out_channels),
                nn.LeakyReLU(0.2),
            ])
        self.reshape = nn.Sequential(*reshape_layers)
        # now we have a 8x8 feature map with decoder_channels channels

        reshaped_width = int(np.sqrt(target_n_features))  # 8
        num_convtranspose_blocks = int(np.log2(output_width) - np.log2(reshaped_width))  # 512 -> 8 = 6 blocks
        convtranspose_blocks = []
        convtranspose_block = [
            nn.ConvTranspose2d(decoder_channels, decoder_channels, 3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(decoder_channels // self.chans_per_group, decoder_channels),
            nn.LeakyReLU(0.2),
        ]
        for _ in range(num_convtranspose_blocks - 1):
            convtranspose_blocks.extend(convtranspose_block)
        # last block has no activation
        convtranspose_blocks.extend([
            nn.ConvTranspose2d(decoder_channels, decoder_channels, 3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(decoder_channels // self.chans_per_group, decoder_channels),
        ])
        self.upscaler = nn.Sequential(*convtranspose_blocks)
        # now we have a 512x512 feature map with decoder_channels channels

        resblocks = [ResBlock(decoder_channels, n_res_channel, self.chans_per_group) for _ in range(n_res_block)]
        resblocks.append(
            nn.LeakyReLU(0.2)  # last resblock has no activation
        )
        self.res_blocks = nn.Sequential(*resblocks)

        # round up output_channels to next power of 2
        output_channels_p2 = 2 ** int(max(1, np.ceil(np.log2(output_channels))))
        # print(f"\nOutput channels: {output_channels}, output_channels_p2: {output_channels_p2}")
        num_head_blocks = int(np.log2(decoder_channels) - np.log2(output_channels_p2))
        # print("\nNum_head_blocks:", num_head_blocks)
        head_layers = []
        for i in range(num_head_blocks):
            in_channels = 2 ** int(np.log2(decoder_channels) - i)
            out_channels = 2 ** int(np.log2(decoder_channels) - i - 1)
            if i < num_head_blocks - 1:
                num_groups = out_channels // self.chans_per_group if out_channels // self.chans_per_group >= 2 else out_channels // 2
                head_layers.extend([
                    nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                    nn.GroupNorm(num_groups, out_channels),
                    nn.LeakyReLU(0.2)
                ])
            else:
                out_channels = output_channels
                head_layers.extend([
                    nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                    # nn.Tanh()  # output is in range [-1, 1]
                    nn.Sigmoid()  # output is in range [0, 1]
                ])
            # print(f"block {i+1}/{num_head_blocks}: {in_channels} --> {out_channels}")

        self.head = nn.Sequential(*head_layers)

    def forward(self, x):
        # print("x shape: ", x.shape)
        x = self.mlp(x)
        # print("mlp x shape: ", x.shape)
        x = self.reshape(x)
        # print("reshape x shape: ", x.shape)
        x = self.upscaler(x)
        # print("upscaler x shape: ", x.shape)
        x = self.res_blocks(x)
        # print("res_blocks x shape: ", x.shape)
        x = self.head(x)
        # print("head x shape: ", x.shape)
        return x


class ImageVAE(nn.Module):
    def __init__(self,
                 input_width=512,
                 output_channels=1,
                 encoder_channels=128,
                 encoder_kernels=[4, 4],
                 encoder_n_res_block=2,
                 encoder_n_res_channel=64,
                 decoder_channels=128,
                 decoder_n_res_block=2,
                 decoder_n_res_channel=64,
                 latent_size=16,
                 ):
        super().__init__()
        self.encoder = MelSpecEncoder(
            input_width=input_width,
            encoder_channels=encoder_channels,
            encoder_kernels=encoder_kernels,
            n_res_block=encoder_n_res_block,
            n_res_channel=encoder_n_res_channel,
            stride=4,
            latent_size=latent_size,
        )
        self.decoder = ImageDecoder(
            output_width=input_width,
            output_channels=output_channels,
            decoder_channels=decoder_channels,
            n_res_block=decoder_n_res_block,
            n_res_channel=decoder_n_res_channel,
            latent_size=latent_size
        )
    
    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoded_img = self.decoder(z)
        return decoded_img, mu, logvar, z


class FMVAE(nn.Module):
    def __init__(self,
                 input_width=512,
                 encoder_channels=128,
                 encoder_kernels=[4, 4],
                 encoder_n_res_block=2,
                 encoder_n_res_channel=32,
                 decoder_features=128,
                 decoder_n_res_block=2,
                 decoder_n_res_features=32,
                 latent_size=8,
                 dropout=0.2
                 ):
        super().__init__()
        self.encoder = MelSpecEncoder(
            input_width=input_width,
            encoder_channels=encoder_channels,
            encoder_kernels=encoder_kernels,
            n_res_block=encoder_n_res_block,
            n_res_channel=encoder_n_res_channel,
            stride=4,
            latent_size=latent_size,
            dropout=dropout
        )
        self.decoder = ParamDecoder(
            decoder_features=decoder_features,
            n_res_block=decoder_n_res_block,
            n_res_features=decoder_n_res_features,
            latent_size=latent_size,
            n_params=3,
            dropout=dropout
        )
    
    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        params = self.decoder(z)
        return params, mu, logvar, z
    

class PlFMFactorVAE(LightningModule):
    def __init__(self, args):
        super().__init__()
        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.args = args

        self.sr = args.sr
        self.batch_size = args.batch_size
        self.warmup_epochs = args.warmup_epochs
        self.param_loss_weight_start = args.param_loss_weight_start
        self.param_loss_weight = args.param_loss_weight_start # initialize to start
        self.param_loss_weight_end = args.param_loss_weight_end
        self.param_loss_weight_ramp_start_epoch = args.param_loss_weight_ramp_start_epoch
        self.param_loss_weight_ramp_end_epoch = args.param_loss_weight_ramp_end_epoch
        self.n_samples = args.length_samps
        self.n_fft = args.n_fft
        self.spectrogram_w = self.n_samples // (self.n_fft // 2) + 1
        self.use_curriculum = args.use_curriculum > 0
        self.min_harm_ratio = args.min_harm_ratio
        self.max_harm_ratio = max(6, self.min_harm_ratio * 2) if self.use_curriculum else args.max_harm_ratio
        self.min_mod_idx = args.min_mod_idx
        self.max_mod_idx = max(6, self.min_mod_idx * 2) if self.use_curriculum else args.max_mod_idx
        self.latent_size = args.latent_size
        self.logdir = args.logdir
        self.d_hidden_size = args.d_hidden_size
        self.d_num_layers = args.d_num_layers

        # losses
        self.mss_loss = MultiResolutionSTFTLoss(
            fft_sizes=[1024, 2048],
            hop_sizes=[256, 512],
            win_lengths=[1024, 2048],
            scale="mel",
            n_bins=128,
            sample_rate=self.sr,
            perceptual_weighting=True,
        )
        self.mss_loss.eval()
        self.kld = kld_loss
        self.recon_weight = args.recon_weight
        self.kld_weight_max = args.kld_weight_max
        self.kld_weight_min = args.kld_weight_min
        self.kld_weight_dynamic = args.kld_weight_min  # initialize to min
        self.kld_start_epoch = args.kld_start_epoch
        self.kld_warmup_epochs = args.kld_warmup_epochs
        self.tc_weight_max = args.tc_weight_max
        self.tc_weight_min = args.tc_weight_min
        self.tc_start_epoch = args.tc_start_epoch
        self.tc_warmup_epochs = args.tc_warmup_epochs
        self.contrastive_regularization = args.contrastive_regularization > 0
        self.contrastive_weight_max = args.contrastive_weight_max
        self.contrastive_weight_min = args.contrastive_weight_min
        self.contrastive_start_epoch = args.contrastive_start_epoch
        self.contrastive_warmup_epochs = args.contrastive_warmup_epochs

        # learning rates
        self.lr_vae = args.lr_vae
        self.lr_decay_vae = args.lr_decay_vae
        self.lr_d = args.lr_d
        self.lr_decay_d = args.lr_decay_d

        # models
        self.input_synth = FMSynth(sr=self.sr)
        self.output_synth = FMSynth(sr=self.sr)
        self.input_synth.eval()
        self.mel_spectrogram = MelSpectrogram(
            sample_rate=self.sr,
            n_fft=self.n_fft,
            f_min=args.f_min,
            f_max=args.f_max,
            n_mels=args.n_mels,
            power=args.power,
            normalized=args.normalized > 0,
        )
        self.model = FMVAE(
            input_width=self.args.n_mels,
            encoder_channels=args.encoder_channels,
            encoder_kernels=args.encoder_kernels,
            encoder_n_res_block=args.encoder_n_res_block,
            encoder_n_res_channel=args.encoder_n_res_channel,
            decoder_features=args.decoder_features,
            decoder_n_res_block=args.decoder_n_res_block,
            decoder_n_res_features=args.decoder_n_res_features,
            latent_size=self.latent_size,
            dropout=self.args.dropout,
        )
        self.D = LinearDiscriminator(
            self.latent_size, self.d_hidden_size, 2, self.d_num_layers)


        def init_weights_kaiming(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):  # Apply to conv and linear layers
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        self.model.apply(init_weights_kaiming)
        self.D.apply(init_weights_kaiming)


    def sample_fm_params(self, batch_size):
        pitches_norm = torch.rand(batch_size, requires_grad=False, device=self.device)
        pitches = scale(pitches_norm, 0, 1, 38, 86)
        freqs = midi2frequency(pitches)
        ratios_norm = torch.rand(batch_size, requires_grad=False, device=self.device)
        ratios = scale(ratios_norm, 0, 1, self.min_harm_ratio, self.max_harm_ratio)
        ratios_norm = scale(ratios, self.min_harm_ratio, self.args.max_harm_ratio, 0, 1)
        indices_norm = torch.rand(batch_size, requires_grad=False, device=self.device)
        indices = scale(indices_norm, 0, 1, self.min_mod_idx, self.max_mod_idx)
        indices_norm = scale(indices, self.min_mod_idx, self.args.max_mod_idx, 0, 1)
        # stack norm params together
        norm_params = torch.stack([pitches_norm, ratios_norm, indices_norm], dim=1)
        # now repeat on the samples dimension
        freqs = freqs.unsqueeze(1).repeat(1, self.sr)
        ratios = ratios.unsqueeze(1).repeat(1, self.sr)
        indices = indices.unsqueeze(1).repeat(1, self.sr)
        return norm_params, freqs, ratios, indices
    

    def scale_predicted_params(self, predicted_params):
        # separate the param dim
        pitches = predicted_params[:, 0]
        ratios = predicted_params[:, 1]
        indices = predicted_params[:, 2]
        # clamp everything to 0..1
        pitches = torch.clamp(pitches, 0, 1)
        ratios = torch.clamp(ratios, 0, 1)
        indices = torch.clamp(indices, 0, 1)
        # scale to the correct range
        pitches = scale(pitches, 0, 1, 38, 86)
        freqs = midi2frequency(pitches)
        ratios = scale(ratios, 0, 1, self.min_harm_ratio, self.args.max_harm_ratio)
        indices = scale(indices, 0, 1, self.min_mod_idx, self.args.max_mod_idx)
        # re-stack them
        out = torch.cat([freqs.unsqueeze(1), ratios.unsqueeze(1), indices.unsqueeze(1)], dim=1)
        return out
    

    def forward(self, x):
        in_wf = x.unsqueeze(1)
        # get the mel spectrogram
        in_spec = self.mel_spectrogram(in_wf)
        # # normalize per sample (and channel) over spatial dims (H, W)
        # reduce_dims = (2, 3)
        # mins = in_spec.amin(dim=reduce_dims, keepdim=True)
        # maxs = in_spec.amax(dim=reduce_dims, keepdim=True)
        # den = (maxs - mins).clamp_min(torch.finfo(in_spec.dtype).eps)
        # in_spec = (in_spec - mins) / den
        # normalize it
        in_spec = scale(in_spec, in_spec.min(), in_spec.max(), 0, 1)
        # predict the params
        norm_predicted_params, mu, logvar, z = self.model(in_spec)
        # scale the predicted params
        predicted_params = self.scale_predicted_params(norm_predicted_params)
        return predicted_params, norm_predicted_params, mu, logvar, z
    

    def training_step(self, batch, batch_idx):
        self.model.train()
        self.D.train()

        # get the optimizers and schedulers
        vae_optimizer, d_optimizer = self.optimizers()
        vae_scheduler, d_scheduler = self.lr_schedulers()

        # create a batch of ones and zeros for the discriminator
        ones = torch.ones(self.batch_size, dtype=torch.long, device=self.device)
        zeros = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)

        # get the batch
        epoch_idx = self.trainer.current_epoch
        triplet_batch = None
        if self.contrastive_regularization:
            _, triplet_batch = batch

        norm_params, freqs, ratios, indices = self.sample_fm_params(self.batch_size)
        x = self.input_synth(freqs, ratios, indices).detach()
        in_wf = x.unsqueeze(1)
        # select a random slice of self.n_samples
        start_idx = torch.randint(0, self.sr - self.n_samples, (1,))
        x = x[:, start_idx:start_idx + self.n_samples]
        # add random phase flip
        phase_flip = torch.rand(self.batch_size, 1, device=self.device)
        phase_flip = torch.where(phase_flip > 0.5, 1, -1)
        x = x * phase_flip
        # add random noise
        noise = torch.randn_like(x, device=self.device)
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        noise = noise * 2 - 1
        noise_coeff = torch.rand(self.batch_size, 1, device=self.device) * 0.001
        noise = noise * noise_coeff
        x = x + noise
        in_wf_slice = x.unsqueeze(1)

        # forward pass
        # get the mel spectrogram
        in_spec = self.mel_spectrogram(in_wf_slice.detach())
        # # normalize per sample (and channel) over spatial dims (H, W)
        # reduce_dims = (2, 3)
        # mins = in_spec.amin(dim=reduce_dims, keepdim=True)
        # maxs = in_spec.amax(dim=reduce_dims, keepdim=True)
        # den = (maxs - mins).clamp_min(torch.finfo(in_spec.dtype).eps)
        # in_spec = (in_spec - mins) / den
        # normalize it
        in_spec = scale(in_spec, in_spec.min(), in_spec.max(), 0, 1)
        # predict the params
        norm_predicted_params, mu, logvar, z = self.model(in_spec)
        # scale the predicted params
        predicted_params = self.scale_predicted_params(norm_predicted_params)
        # now repeat on the samples dimension
        predicted_freqs = predicted_params[:, 0].unsqueeze(1).repeat(1, self.sr)
        predicted_ratios = predicted_params[:, 1].unsqueeze(1).repeat(1, self.sr)
        predicted_indices = predicted_params[:, 2].unsqueeze(1).repeat(1, self.sr)
        # generate the output
        y = self.output_synth(predicted_freqs, predicted_ratios, predicted_indices)
        out_wf = y.unsqueeze(1)

        # VAE recon_loss: MSS + param loss
        # param loss
        param_loss = F.l1_loss(norm_predicted_params, norm_params.detach())
        # mss loss
        mss_loss = self.mss_loss(out_wf, in_wf)
        # calculate current param loss weight
        current_epoch = self.trainer.current_epoch
        if current_epoch < self.param_loss_weight_ramp_start_epoch:
            param_loss_weight = self.param_loss_weight_start
        elif current_epoch > self.param_loss_weight_ramp_end_epoch:
            param_loss_weight = self.param_loss_weight_end
        else:
            param_loss_weight = self.param_loss_weight_start + (self.param_loss_weight_end - self.param_loss_weight_start) * \
                (current_epoch - self.param_loss_weight_ramp_start_epoch) / \
                (self.param_loss_weight_ramp_end_epoch - self.param_loss_weight_ramp_start_epoch)
        self.param_loss_weight = param_loss_weight
        vae_recon_loss = (param_loss * self.param_loss_weight) + mss_loss
        scaled_vae_recon_loss = vae_recon_loss * self.recon_weight

        # VAE KLD loss
        if self.args.dynamic_kld > 0:
            kld_scale = self.kld_weight_dynamic
        else:
            kld_scale = (self.kld_weight_max - self.kld_weight_min) * \
                min(1.0, (epoch_idx - self.kld_start_epoch) /
                    self.kld_warmup_epochs) + self.kld_weight_min if epoch_idx > self.kld_start_epoch else self.kld_weight_min
        kld_loss = self.kld(mu, logvar)
        scaled_kld_loss = kld_loss * kld_scale

        # VAE TC loss
        d_z = self.D(z)
        vae_tc_loss = F.cross_entropy(d_z, ones)
        tc_scale = (self.tc_weight_max - self.tc_weight_min) * \
            min(1.0, (epoch_idx - self.tc_start_epoch) /
            self.tc_warmup_epochs) + self.tc_weight_min if epoch_idx > self.tc_start_epoch else self.tc_weight_min
        scaled_vae_tc_loss = vae_tc_loss * tc_scale

        # VAE contrastive loss
        triplet_loss, scaled_triplet_loss, triplet_loss_scale = 0, 0, 0
        if self.contrastive_regularization:
            # get triplet data
            anchor_params = triplet_batch[:, 0].unsqueeze(-1).repeat(1, 1, self.sr) # (B, 3, ds_n_samps)
            positive_params = triplet_batch[:, 1].unsqueeze(-1).repeat(1, 1, self.sr)
            negative_params = triplet_batch[:, 2].unsqueeze(-1).repeat(1, 1, self.sr)
            # create waveforms with the usual augmentations
            triplet_specs = [None, None, None]
            for idx, params in enumerate([anchor_params, positive_params, negative_params]):
                freqs, ratios, indices = params[:, 0], params[:, 1], params[:, 2] # (B, 1, ds_n_samps) (B, 1, ds_n_samps) (B, 1, ds_n_samps)
                # generate the waveform
                _x = self.input_synth(freqs, ratios, indices).detach() # (B, 1, ds_n_samps)
                _x = _x.squeeze(1)  # remove the channel dimension
                # select a random slice of self.n_samples
                start_idx = torch.randint(0, _x.shape[-1] - self.n_samples, (1,))
                _x = _x[:, start_idx:start_idx + self.n_samples]
                # add random phase flip
                phase_flip = torch.rand(self.batch_size, 1, device=self.device)
                phase_flip = torch.where(phase_flip > 0.5, 1, -1)
                _x = _x * phase_flip
                # add random noise
                noise = torch.randn_like(_x, device=self.device)
                noise = (noise - noise.min()) / (noise.max() - noise.min())
                noise = noise * 2 - 1
                noise_coeff = torch.rand(self.batch_size, 1, device=self.device) * 0.001
                noise = noise * noise_coeff
                _x = _x + noise
                spectrogram = self.mel_spectrogram(_x.unsqueeze(1).detach())
                # normalize it
                spectrogram = scale(spectrogram, spectrogram.min(), spectrogram.max(), 0, 1)
                triplet_specs[idx] = spectrogram
            # encode with the VAE
            anchor_spec, positive_spec, negative_spec = triplet_specs
            anchor_z = self.model.reparameterize(*self.model.encode(anchor_spec))
            positive_z = self.model.reparameterize(*self.model.encode(positive_spec))
            negative_z = self.model.reparameterize(*self.model.encode(negative_spec))
            # calculate triplet loss
            triplet_loss = F.triplet_margin_loss(anchor_z, positive_z, negative_z, margin=1.0, p=2)
            triplet_loss_scale = (self.contrastive_weight_max - self.contrastive_weight_min) * \
                min(1.0, (epoch_idx - self.contrastive_start_epoch) /
                    self.contrastive_warmup_epochs) + self.contrastive_weight_min if epoch_idx > self.contrastive_start_epoch else self.contrastive_weight_min
            scaled_triplet_loss = triplet_loss * triplet_loss_scale

        # VAE loss
        vae_loss = scaled_vae_recon_loss + scaled_kld_loss + scaled_vae_tc_loss + scaled_triplet_loss

        # VAE backward pass
        vae_optimizer.zero_grad()
        self.manual_backward(vae_loss, retain_graph=True)
        self.clip_gradients(vae_optimizer, gradient_clip_val=0.5, gradient_clip_algorithm="norm")

        vae_optimizer.step()
        vae_scheduler.step(vae_loss.item())

        # get another batch for D
        norm_params, freqs, ratios, indices = self.sample_fm_params(self.batch_size)
        x = self.input_synth(freqs, ratios, indices).detach()
        in_wf = x.unsqueeze(1)
        # select a random slice of self.n_samples
        start_idx = torch.randint(0, self.sr - self.n_samples, (1,))
        x = x[:, start_idx:start_idx + self.n_samples]
        # add random phase flip
        phase_flip = torch.rand(self.batch_size, 1, device=self.device)
        phase_flip = torch.where(phase_flip > 0.5, 1, -1)
        x = x * phase_flip
        # add random noise
        noise = torch.randn_like(x, device=self.device)
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        noise = noise * 2 - 1
        noise_coeff = torch.rand(self.batch_size, 1, device=self.device) * 0.001
        noise = noise * noise_coeff
        x = x + noise
        in_wf_slice = x.unsqueeze(1)
        # get the mel spectrogram
        in_spec = self.mel_spectrogram(in_wf_slice.detach())
        # # normalize per sample (and channel) over spatial dims (H, W)
        # reduce_dims = (2, 3)
        # mins = in_spec.amin(dim=reduce_dims, keepdim=True)
        # maxs = in_spec.amax(dim=reduce_dims, keepdim=True)
        # den = (maxs - mins).clamp_min(torch.finfo(in_spec.dtype).eps)
        # in_spec = (in_spec - mins) / den
        # normalize it
        in_spec = scale(in_spec, in_spec.min(), in_spec.max(), 0, 1)
        # encode with the VAE
        self.model.eval()
        mu_2, logvar_2 = self.model.encode(in_spec)
        # reparameterize
        z_2 = self.model.reparameterize(mu_2, logvar_2)
        z_2_perm = permute_dims(z_2)
        # get the discriminator output
        d_z_detached = self.D(z.detach())
        d_z_2_perm = self.D(z_2_perm.detach())
        d_tc_loss = 0.5 * (F.cross_entropy(d_z_detached, zeros) +
                           F.cross_entropy(d_z_2_perm, ones))

        # Discriminator backward pass
        d_optimizer.zero_grad()
        self.manual_backward(d_tc_loss)
        self.clip_gradients(d_optimizer, gradient_clip_val=0.5, gradient_clip_algorithm="norm")

        # VAE step
        # vae_optimizer.step()
        # vae_scheduler.step(vae_loss.item())
        # D step
        d_optimizer.step()
        d_scheduler.step(d_tc_loss.item())

        # log losses
        # self.last_recon_loss = vae_recon_loss
        # self.last_recon_loss = param_loss # using it for dynamic kld threshold
        self.last_recon_loss = mss_loss # using it for dynamic kld threshold
        self.log_dict({
            "vae_loss": vae_loss,
            "vae_param_loss": param_loss,
            "vae_mss_loss": mss_loss,
            "vae_kld_loss": kld_loss,
            "vae_tc_loss": vae_tc_loss,
            "vae_triplet_loss": triplet_loss,
            "d_tc_loss": d_tc_loss,
            "lr_vae": vae_scheduler.get_last_lr()[0],
            "lr_d": d_scheduler.get_last_lr()[0],
            "vae_param_loss_weight": self.param_loss_weight,
            "vae_kld_scale": kld_scale,
            "vae_tc_scale": tc_scale,
            "vae_triplet_scale": triplet_loss_scale,
            "max_harm_ratio": self.max_harm_ratio,
            "max_mod_idx": self.max_mod_idx
        },
        prog_bar=True)


    def on_train_epoch_end(self):
        change_patience = False
        if change_patience:
            epoch = self.trainer.current_epoch
            vae_scheduler, d_scheduler = self.lr_schedulers()
            for scheduler in [vae_scheduler, d_scheduler]:
                # Check if this is a ReduceLROnPlateau scheduler
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if epoch == 500:
                        # Dynamically increase patience at epoch 500
                        scheduler.patience = 60 * 1000
                        scheduler.num_bad_epochs = 0
                        print(f"Patience changed to {scheduler.patience} at epoch {epoch}")
                    if epoch == 700:
                        scheduler.patience = 70 * 1000
                        scheduler.num_bad_epochs = 0
                        print(f"Patience changed to {scheduler.patience} at epoch {epoch}")
                    if epoch == 900:
                        scheduler.patience = 80 * 1000
                        scheduler.num_bad_epochs = 0
                        print(f"Patience changed to {scheduler.patience} at epoch {epoch}")
        # update the kld weight
        changed_kld, changed_param = False, False
        if self.args.dynamic_kld > 0 and not self.use_curriculum: # only start dynamic kld when curriculum is completed
            if self.last_recon_loss < self.args.target_recon_loss:
                self.kld_weight_dynamic *= 1.01
                changed_kld = True
        if self.use_curriculum:
            if self.last_recon_loss < self.args.target_recon_loss:
                # increase the max_harm_ratio and max_mod_idx
                if self.max_harm_ratio < self.args.max_harm_ratio:
                    self.max_harm_ratio += 0.2
                    self.max_harm_ratio = min(self.max_harm_ratio, self.args.max_harm_ratio)
                    changed_param = True
                if self.max_mod_idx < self.args.max_mod_idx:
                    self.max_mod_idx += 0.2
                    self.max_mod_idx = min(self.max_mod_idx, self.args.max_mod_idx)
                    changed_param = True
                # switch off if finished
                if self.max_harm_ratio >= self.args.max_harm_ratio and self.max_mod_idx >= self.args.max_mod_idx:
                    self.use_curriculum = False
                    print("Curriculum learning completed!")
        # if changed param or kld, reset scheduler bad epochs
        if changed_param or changed_kld:
            epoch = self.trainer.current_epoch
            vae_scheduler, d_scheduler = self.lr_schedulers()
            for scheduler in [vae_scheduler, d_scheduler]:
                # Check if this is a ReduceLROnPlateau scheduler
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.num_bad_epochs = 0
                    print(f"Resetting patience for {scheduler} at epoch {epoch}")


    def on_train_batch_start(self, batch, batch_idx):
        epoch = self.trainer.current_epoch
        if epoch < self.warmup_epochs:
            lr_scale = min(1.0, (epoch + 1) / self.warmup_epochs)
            for pg in self.trainer.optimizers[0].param_groups:
                pg["lr"] = self.lr_vae * lr_scale
            for pg in self.trainer.optimizers[1].param_groups:
                pg["lr"] = self.lr_d * lr_scale


    def configure_optimizers(self):
        vae_optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr_vae)
        d_optimizer = torch.optim.AdamW(
            self.D.parameters(), lr=self.lr_d)
        vae_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            vae_optimizer, mode='min', factor=self.lr_decay_vae, patience=10000)
        d_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            d_optimizer, mode='min', factor=self.lr_decay_d, patience=10000)
        # return the optimizers and schedulers
        return [vae_optimizer, d_optimizer], [vae_scheduler, d_scheduler]
    

class PlImgFactorVAE(LightningModule):
    def __init__(self, args):
        super().__init__()
        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.args = args

        self.batch_size = args.batch_size
        self.warmup_epochs = args.warmup_epochs
        self.latent_size = args.latent_size
        self.logdir = args.logdir
        self.d_hidden_size = args.d_hidden_size
        self.d_num_layers = args.d_num_layers

        # losses
        self.recon_weight = args.recon_weight
        self.kld = kld_loss
        self.kld_weight_max = args.kld_weight_max
        self.kld_weight_min = args.kld_weight_min
        self.kld_weight_dynamic = args.kld_weight_min  # initialize to min
        self.kld_start_epoch = args.kld_start_epoch
        self.kld_warmup_epochs = args.kld_warmup_epochs
        self.tc_weight_max = args.tc_weight_max
        self.tc_weight_min = args.tc_weight_min
        self.tc_start_epoch = args.tc_start_epoch
        self.tc_warmup_epochs = args.tc_warmup_epochs

        # learning rates
        self.lr_vae = args.lr_vae
        self.lr_decay_vae = args.lr_decay_vae
        self.lr_d = args.lr_d
        self.lr_decay_d = args.lr_decay_d

        # models
        self.model = ImageVAE(
            input_width=args.input_width,
            output_channels=args.output_channels,
            encoder_channels=args.encoder_channels,
            encoder_kernels=args.encoder_kernels,
            encoder_n_res_block=args.encoder_n_res_block,
            encoder_n_res_channel=args.encoder_n_res_channel,
            decoder_channels=args.decoder_channels,
            decoder_n_res_block=args.decoder_n_res_block,
            decoder_n_res_channel=args.decoder_n_res_channel,
            latent_size=self.latent_size
        )
        self.D = LinearDiscriminator(
            self.latent_size, self.d_hidden_size, 2, self.d_num_layers)


        def init_weights_kaiming(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):  # Apply to conv and linear layers
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        self.model.apply(init_weights_kaiming)
        self.D.apply(init_weights_kaiming)
    

    def forward(self, x):
        # predict the image
        predicted_img, mu, logvar, z = self.model(x)
        return predicted_img, mu, logvar, z
    

    def on_train_epoch_start(self):
        # Reinitialize the Discriminator batch iterator at the start of every epoch
        train_loader = self.trainer.train_dataloader
        self.disc_iter = iter(train_loader)
    

    def training_step(self, batch, batch_idx):
        self.model.train()
        self.D.train()

        # get the optimizers and schedulers
        vae_optimizer, d_optimizer = self.optimizers()
        vae_scheduler, d_scheduler = self.lr_schedulers()

        # create a batch of ones and zeros for the discriminator
        ones = torch.ones(self.batch_size, dtype=torch.long, device=self.device)
        zeros = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)

        # get the batch
        epoch_idx = self.trainer.current_epoch
        # batch is the main VAE batch
        x_vae, _ = batch  # assuming MNIST (x, label)

        # Get discriminator batch
        try:
            x_disc, _ = next(self.disc_iter)
        except StopIteration:
            self.disc_iter = iter(self.trainer.train_dataloader)
            x_disc, _ = next(self.disc_iter)

        x_vae = x_vae.to(self.device)
        x_disc = x_disc.to(self.device)

        # forward pass
        # predict the image
        predicted_img, mu, logvar, z = self.model(x_vae)

        # VAE recon_loss
        vae_recon_loss = F.l1_loss(predicted_img, x_vae)
        scaled_vae_recon_loss = vae_recon_loss * self.recon_weight

        # VAE KLD loss
        if self.args.dynamic_kld > 0:
            kld_scale = self.kld_weight_dynamic
        else:
            kld_scale = (self.kld_weight_max - self.kld_weight_min) * \
                min(1.0, (epoch_idx - self.kld_start_epoch) /
                    self.kld_warmup_epochs) + self.kld_weight_min if epoch_idx > self.kld_start_epoch else self.kld_weight_min
        kld_loss = self.kld(mu, logvar)
        scaled_kld_loss = kld_loss * kld_scale

        # VAE TC loss
        d_z = self.D(z)
        vae_tc_loss = F.cross_entropy(d_z, ones)
        tc_scale = (self.tc_weight_max - self.tc_weight_min) * \
            min(1.0, (epoch_idx - self.tc_start_epoch) /
            self.tc_warmup_epochs) + self.tc_weight_min if epoch_idx > self.tc_start_epoch else self.tc_weight_min
        scaled_vae_tc_loss = vae_tc_loss * tc_scale

        # VAE loss
        vae_loss = scaled_vae_recon_loss + scaled_kld_loss + scaled_vae_tc_loss

        # VAE backward pass
        vae_optimizer.zero_grad()
        self.manual_backward(vae_loss, retain_graph=True)
        self.clip_gradients(vae_optimizer, gradient_clip_val=0.5, gradient_clip_algorithm="norm")

        vae_optimizer.step()
        vae_scheduler.step(vae_loss.item())

        # Discriminator forward pass
        # encode with the VAE
        self.model.eval()
        mu_2, logvar_2 = self.model.encode(x_disc)
        # reparameterize
        z_2 = self.model.reparameterize(mu_2, logvar_2)
        z_2_perm = permute_dims(z_2)
        # get the discriminator output
        d_z_detached = self.D(z.detach())
        d_z_2_perm = self.D(z_2_perm.detach())
        d_tc_loss = 0.5 * (F.cross_entropy(d_z_detached, zeros) +
                           F.cross_entropy(d_z_2_perm, ones))

        # Discriminator backward pass
        d_optimizer.zero_grad()
        self.manual_backward(d_tc_loss)
        self.clip_gradients(d_optimizer, gradient_clip_val=0.5, gradient_clip_algorithm="norm")

        # D step
        d_optimizer.step()
        d_scheduler.step(d_tc_loss.item())

        # log losses
        self.last_recon_loss = vae_recon_loss
        self.log_dict({
            "vae_loss": vae_loss,
            "vae_recon_loss": vae_recon_loss,
            "vae_kld_loss": kld_loss,
            "vae_tc_loss": vae_tc_loss,
            "d_tc_loss": d_tc_loss,
            "lr_vae": vae_scheduler.get_last_lr()[0],
            "lr_d": d_scheduler.get_last_lr()[0],
            "vae_kld_scale": kld_scale,
            "vae_tc_scale": tc_scale,
        },
        prog_bar=True)


    def on_validation_epoch_start(self):
        # Reinitialize the Discriminator batch iterator at the start of every validation epoch
        val_loader = self.trainer.val_dataloaders
        self.disc_iter = iter(val_loader)

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        self.D.eval()

        # create a batch of ones and zeros for the discriminator
        ones = torch.ones(self.batch_size, dtype=torch.long, device=self.device)
        zeros = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)

        # get the batch
        # epoch_idx = self.trainer.current_epoch
        # batch is the main VAE batch
        x_vae, _ = batch  # assuming MNIST (x, label)

        # Get discriminator batch
        try:
            x_disc, _ = next(self.disc_iter)
        except StopIteration:
            self.disc_iter = iter(self.trainer.val_dataloaders)
            x_disc, _ = next(self.disc_iter)

        x_vae = x_vae.to(self.device)
        x_disc = x_disc.to(self.device)

        # forward pass
        # predict the image
        predicted_img, mu, logvar, z = self.model(x_vae)

        # VAE recon_loss
        vae_recon_loss = F.l1_loss(predicted_img, x_vae)

        # VAE KLD loss
        kld_loss = self.kld(mu, logvar)

        # VAE TC loss
        d_z = self.D(z)
        vae_tc_loss = F.cross_entropy(d_z, ones)

        # Discriminator forward pass
        # encode with the VAE
        mu_2, logvar_2 = self.model.encode(x_disc)
        # reparameterize
        z_2 = self.model.reparameterize(mu_2, logvar_2)
        z_2_perm = permute_dims(z_2)
        # get the discriminator output
        d_z_detached = self.D(z.detach())
        d_z_2_perm = self.D(z_2_perm.detach())
        d_tc_loss = 0.5 * (F.cross_entropy(d_z_detached, zeros) +
                           F.cross_entropy(d_z_2_perm, ones))
        
        # log losses
        self.log_dict({
            "val_vae_loss": vae_recon_loss + kld_loss + vae_tc_loss,
            "val_vae_recon_loss": vae_recon_loss,
            "val_vae_kld_loss": kld_loss,
            "val_vae_tc_loss": vae_tc_loss,
            "val_d_tc_loss": d_tc_loss,
        },
        prog_bar=True)


    def on_train_epoch_end(self):
        change_patience = False
        if change_patience:
            epoch = self.trainer.current_epoch
            vae_scheduler, d_scheduler = self.lr_schedulers()
            for scheduler in [vae_scheduler, d_scheduler]:
                # Check if this is a ReduceLROnPlateau scheduler
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if epoch == 500:
                        # Dynamically increase patience at epoch 500
                        scheduler.patience = 60 * 1000
                        scheduler.num_bad_epochs = 0
                        print(f"Patience changed to {scheduler.patience} at epoch {epoch}")
                    if epoch == 700:
                        scheduler.patience = 70 * 1000
                        scheduler.num_bad_epochs = 0
                        print(f"Patience changed to {scheduler.patience} at epoch {epoch}")
                    if epoch == 900:
                        scheduler.patience = 80 * 1000
                        scheduler.num_bad_epochs = 0
                        print(f"Patience changed to {scheduler.patience} at epoch {epoch}")
        # update the kld weight
        if self.args.dynamic_kld > 0:
            if self.last_recon_loss < self.args.target_recon_loss:
                self.kld_weight_dynamic *= 1.01


    def on_train_batch_start(self, batch, batch_idx):
        epoch = self.trainer.current_epoch
        if epoch < self.warmup_epochs:
            lr_scale = min(1.0, (epoch + 1) / self.warmup_epochs)
            for pg in self.trainer.optimizers[0].param_groups:
                pg["lr"] = self.lr_vae * lr_scale
            for pg in self.trainer.optimizers[1].param_groups:
                pg["lr"] = self.lr_d * lr_scale


    def configure_optimizers(self):
        vae_optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr_vae)
        d_optimizer = torch.optim.AdamW(
            self.D.parameters(), lr=self.lr_d)
        vae_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            vae_optimizer, mode='min', factor=self.lr_decay_vae, patience=50000)
        d_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            d_optimizer, mode='min', factor=self.lr_decay_d, patience=50000)
        # return the optimizers and schedulers
        return [vae_optimizer, d_optimizer], [vae_scheduler, d_scheduler]
    

class PlFMEmbedder(LightningModule):
    def __init__(self, args):
        super().__init__()
        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.args = args

        self.sr = args.sr
        self.resample_base = getattr(args, "resample_base", 960)  # divides 48k well
        if self.sr % self.resample_base != 0:
            self.resample_base = math.gcd(self.sr, self.resample_base) or 1
        self.batch_size = args.batch_size
        self.warmup_epochs = args.warmup_epochs
        self.n_samples = args.length_samps
        self.n_fft = args.n_fft
        self.spectrogram_w = self.n_samples // (self.n_fft // 2) + 1
        self.max_harm_ratio = args.max_harm_ratio
        self.max_mod_idx = args.max_mod_idx
        self.num_views = args.num_views
        self.apply_transposition = args.apply_transposition > 0
        self.transposition_range = args.transposition_range
        self.noise_max_amp = args.noise_max_amp
        self.latent_size = args.latent_size
        self.center_momentum = args.center_momentum
        self.register_buffer("center", torch.zeros(1, self.latent_size, device=self.device))
        self.ema_decay_min = args.ema_decay_min
        self.ema_decay_max = args.ema_decay_max
        self.ema_decay_ramp_start_epoch = args.ema_decay_ramp_start_epoch
        self.ema_decay_ramp_num_epochs = args.ema_decay_ramp_num_epochs
        self.student_temperature = args.student_temperature
        self.teacher_temperature_min = args.teacher_temperature_min
        self.teacher_temperature_max = args.teacher_temperature_max
        self.teacher_temperature_ramp_start_epoch = args.teacher_temperature_ramp_start_epoch
        self.teacher_temperature_ramp_num_epochs = args.teacher_temperature_ramp_num_epochs
        self.logdir = args.logdir

        # learning rate
        self.lr = args.lr
        self.lr_decay = args.lr_decay

        # models
        self.input_synth = FMSynth(sr=self.sr)
        self.input_synth.eval()
        self.mel_spectrogram = MelSpectrogram(
            sample_rate=self.sr,
            n_fft=self.n_fft,
            f_min=args.f_min,
            f_max=args.f_max,
            n_mels=args.n_mels,
            power=args.power,
            normalized=args.normalized > 0,
        )
        self.model = MelSpecEncoder(
            input_width=args.n_mels,
            encoder_channels=args.encoder_channels,
            encoder_kernels=args.encoder_kernels,
            n_res_block=args.encoder_n_res_block,
            n_res_channel=args.encoder_n_res_channel,
            stride=4,
            latent_size=self.latent_size,
            dropout=args.dropout
        )

        def init_weights_kaiming(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):  # Apply to conv and linear layers
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        self.model.apply(init_weights_kaiming)


    def create_shadow_model(self):
        self.shadow = deepcopy(self.model)
        for param in self.shadow.parameters():
            param.detach_()
        self.shadow.eval()


    # following this source: https://www.zijianhu.com/post/pytorch/ema/
    @torch.no_grad()
    def ema_update(self):
        if not self.training:
            print("EMA update should only be called during training",
                  file=stderr, flush=True)
            return

        model_params = OrderedDict(self.model.named_parameters())
        shadow_params = OrderedDict(self.shadow.named_parameters())

        # check if both model contains the same set of keys
        assert model_params.keys() == shadow_params.keys()

        # calculate decay
        decay = (self.ema_decay_max - self.ema_decay_min) * \
            min(1.0, (self.trainer.current_epoch - self.ema_decay_ramp_start_epoch) /
            self.ema_decay_ramp_num_epochs) + self.ema_decay_min if self.trainer.current_epoch > self.ema_decay_ramp_start_epoch else self.ema_decay_min
        self.log("ema_decay", decay)

        for name, param in model_params.items():
            # see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
            # shadow_variable -= (1 - decay) * (shadow_variable - variable)
            shadow_params[name].sub_(
                (1. - decay) * (shadow_params[name] - param))

        model_buffers = OrderedDict(self.model.named_buffers())
        shadow_buffers = OrderedDict(self.shadow.named_buffers())

        # check if both model contains the same set of keys
        assert model_buffers.keys() == shadow_buffers.keys()

        for name, buffer in model_buffers.items():
            # buffers are copied
            shadow_buffers[name].copy_(buffer)

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output on a single GPU.
        """
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


    def sample_fm_params(self, batch_size):
        pitches_norm = torch.rand(batch_size, requires_grad=False, device=self.device)
        pitches = scale(pitches_norm, 0, 1, 38, 86)
        freqs = midi2frequency(pitches)
        ratios_norm = torch.rand(batch_size, requires_grad=False, device=self.device)
        ratios = ratios_norm * self.max_harm_ratio
        ratios = torch.clip(ratios, 0.1, self.max_harm_ratio) # avoid zero ratios
        indices_norm = torch.rand(batch_size, requires_grad=False, device=self.device)
        indices = indices_norm * self.max_mod_idx
        indices = torch.clip(indices, 0.1, self.max_mod_idx) # avoid zero indices
        # stack norm params together
        norm_params = torch.stack([pitches_norm, ratios_norm, indices_norm], dim=1)
        # now repeat on the samples dimension
        freqs = freqs.unsqueeze(1).repeat(1, self.sr)
        ratios = ratios.unsqueeze(1).repeat(1, self.sr)
        indices = indices.unsqueeze(1).repeat(1, self.sr)
        return norm_params, freqs, ratios, indices
    

    def forward(self, x):
        # print("entering PlFMEmbedder forward")
        in_wf = x.unsqueeze(1)
        # get the mel spectrogram
        in_spec = self.mel_spectrogram(in_wf)
        # normalize per sample (and channel) over spatial dims (H, W)
        reduce_dims = (2, 3)
        mins = in_spec.amin(dim=reduce_dims, keepdim=True)
        maxs = in_spec.amax(dim=reduce_dims, keepdim=True)
        den = (maxs - mins).clamp_min(torch.finfo(in_spec.dtype).eps)
        in_spec = (in_spec - mins) / den
        # predict the embedding
        if self.training:
            # print("training, using the main model")
            mu, logvar = self.model(in_spec)
        else:
            # print("not training, using the shadow model")
            # use the shadow model for inference
            mu, logvar = self.shadow(in_spec)
        return mu #+ logvar


    def training_step(self, batch, batch_idx):
        self.model.train()
        self.shadow.eval()

        # get the optimizers and schedulers
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()

        # get the batch
        # epoch_idx = self.trainer.current_epoch

        norm_params, freqs, ratios, indices = self.sample_fm_params(self.batch_size)
        x = self.input_synth(freqs, ratios, indices).detach() # (batch_size, n_samples)
        views = []
        for _ in range(self.num_views):
            # copy to x_a
            x_a = x.clone()

            # apply transposition augmentation for x_a
            if self.apply_transposition:
                # get a random number between -2 and 2 for pitch transposition
                transposition = torch.rand(1) * (2 * self.transposition_range) - self.transposition_range
                target_dur = transposition2duration(transposition.float())[0]
                # Quantize new_sr to keep gcd(orig,new) large
                new_sr = int(round(self.sr * float(target_dur) / self.resample_base) * self.resample_base)
                new_sr = max(1, new_sr)
                # apply the transposition
                # x_a = x_a.unsqueeze(1) # (batch_size, 1, n_samples)
                # x_a = torch.nn.functional.interpolate(x_a, scale_factor=target_dur, mode='linear')
                # print(x_a.shape, self.sr, target_dur, new_sr, new_sr / self.sr)
                x_a = resample(x_a, self.sr, new_sr, lowpass_filter_width=128)
                # x_a = x_a.squeeze(1) # (batch_size, n_samples)

            # apply common augmentations: random slice, random phase flip, random noise
            # select a random slice of self.n_samples
            start_idx = torch.randint(0, x_a.shape[-1] - self.n_samples, (1,))
            x_a = x_a[:, start_idx:start_idx + self.n_samples]
            # add random phase flip
            phase_flip = torch.rand(self.batch_size, 1, device=self.device)
            phase_flip = torch.where(phase_flip > 0.5, 1, -1)
            x_a = x_a * phase_flip
            # add random noise
            noise = torch.randn_like(x_a, device=self.device)
            noise = (noise - noise.min()) / (noise.max() - noise.min())
            noise = noise * 2 - 1
            noise_coeff = torch.rand(self.batch_size, 1, device=self.device) * self.noise_max_amp
            noise = noise * noise_coeff
            x_a = x_a + noise
            # x_a = x_a.unsqueeze(1) # (batch_size, 1, n_samples)
            # add to the views
            views.append(x_a)

        # forward pass
        # TEACHER
        # get the mel spectrograms
        in_spec_x = self.mel_spectrogram(x.unsqueeze(1).detach())
        # normalize per sample (and channel) over spatial dims (H, W)
        reduce_dims = (2, 3)
        mins = in_spec.amin(dim=reduce_dims, keepdim=True)
        maxs = in_spec.amax(dim=reduce_dims, keepdim=True)
        den = (maxs - mins).clamp_min(torch.finfo(in_spec.dtype).eps)
        in_spec = (in_spec - mins) / den
        # predict the embeddings
        teacher_x, _ = self.shadow(in_spec_x)  # teacher output
        teacher_x = teacher_x.detach()  # detach the teacher output to avoid gradients flowing back to the shadow model
        current_teacher_temperature = (self.teacher_temperature_max - self.teacher_temperature_min) * \
            min(1.0, (self.trainer.current_epoch - self.teacher_temperature_ramp_start_epoch) /
            self.teacher_temperature_ramp_num_epochs) + self.teacher_temperature_min if self.trainer.current_epoch > self.teacher_temperature_ramp_start_epoch else self.teacher_temperature_min
        # center and sharpen DINO-style
        teacher_x_centered = F.softmax((teacher_x - self.center) / current_teacher_temperature, dim=-1).detach()

        # STUDENT
        total_loss = 0.0
        for x_a in views:
            # get the mel spectrogram
            in_spec_x_a = self.mel_spectrogram(x_a.unsqueeze(1).detach())
            in_spec_x_a = scale(in_spec_x_a, in_spec_x_a.min(), in_spec_x_a.max(), 0, 1)

            student_x_a, _ = self.model(in_spec_x_a)
            student_x_a = student_x_a / self.student_temperature

            # calculate the loss
            # cross entropy loss between teacher and student (DINO-style)
            loss = torch.sum(-teacher_x_centered * F.log_softmax(student_x_a, dim=-1), dim=-1)
            total_loss += loss.mean()
        total_loss /= len(views)  # average over the views

        # backward pass
        optimizer.zero_grad()
        self.manual_backward(total_loss)
        self.clip_gradients(optimizer, gradient_clip_val=0.5, gradient_clip_algorithm="norm")

        optimizer.step()
        scheduler.step(total_loss.item())

        # update the EMA
        self.ema_update()
        # update the center
        self.update_center(teacher_x)

        # log losses
        self.log_dict({
            "loss": total_loss,
            "lr": scheduler.get_last_lr()[0],
            "teacher_temperature": current_teacher_temperature,
        }, prog_bar=True)
        

    def on_train_batch_start(self, batch, batch_idx):
        epoch = self.trainer.current_epoch
        if epoch < self.warmup_epochs:
            lr_scale = min(1.0, (epoch + 1) / self.warmup_epochs)
            for pg in self.trainer.optimizers[0].param_groups:
                pg["lr"] = self.lr * lr_scale


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=self.lr_decay, patience=100000)
        # return the optimizers and schedulers
        return [optimizer], [scheduler]