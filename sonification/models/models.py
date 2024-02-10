import torch
from torch import nn
import torch.nn.functional as F
from .layers import *
from lightning.pytorch import LightningModule
from ..utils.tensor import permute_dims
from .loss import kld_loss


class AE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, output_size):
        super(AE, self).__init__()
        self.encoder = LinearEncoder(input_size, hidden_size, latent_size)
        self.decoder = LinearDecoder(latent_size, hidden_size, output_size)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)


# inspired by https://medium.com/@rekalantar/variational-auto-encoder-vae-pytorch-tutorial-dce2d2fe0f5f
class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(VAE, self).__init__()
        self.encoder = LinearEncoder(input_size, hidden_size, latent_size)
        self.mu = nn.Linear(latent_size, latent_size)
        self.logvar = nn.Linear(latent_size, latent_size)
        self.decoder = LinearDecoder(latent_size, hidden_size, input_size)

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
    def __init__(self, in_channels, latent_size, layers_channels=[16, 32, 64, 128, 256, 512], input_size=512):
        super(ConvVAE, self).__init__()
        self.encoder = ConvEncoder(in_channels, latent_size, layers_channels, input_size)
        self.mu = nn.Linear(latent_size, latent_size)
        self.logvar = nn.Linear(latent_size, latent_size)
        self.decoder = ConvDecoder(latent_size, in_channels, layers_channels, input_size)
    
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
    

# used this for guidance: https://github.com/1Konny/FactorVAE/blob/master/solver.py
class PlFactorVAE(LightningModule):
    def __init__(
            self, 
            in_channels, 
            latent_size,  
            layers_channels, 
            input_size,
            d_hidden_size,
            d_num_layers,
            lr_vae,
            lr_d,
            kld_weight,
            tc_weight
            ):
        super(PlFactorVAE, self).__init__()
        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        self.save_hyperparameters()
        # models
        self.VAE = ConvVAE(in_channels, latent_size, layers_channels, input_size)
        self.D = LinearDiscriminator(latent_size, d_hidden_size, 2, d_num_layers)
        # losses
        self.mse = nn.MSELoss()
        self.kld = kld_loss
        self.kld_weight = kld_weight
        self.tc_weight = tc_weight
        # learning rates
        self.lr_vae = lr_vae
        self.lr_d = lr_d


    def forward(self, x):
        return self.VAE(x)
    

    def encode(self, x):
        mean, logvar = self.VAE.encode(x)
        return self.VAE.reparameterize(mean, logvar)
    

    def decode(self, z):
        return self.VAE.decoder(z)


    def training_step(self, batch, batch_idx):
        # get the optimizers
        vae_optimizer, d_optimizer = self.optimizers()

        # get the batch
        x_1, x_2 = batch
        batch_size = x_1.shape[0]

        # create a batch of ones and zeros for the discriminator
        ones = torch.ones(batch_size, dtype=torch.long, device=self.device)
        zeros = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        # VAE forward pass
        x_recon, mean, logvar, z = self.VAE(x_1)
        vae_recon_loss = self.mse(x_recon, x_1)
        kld_loss = self.kld(mean, logvar)
        # VAE TC loss
        d_z = self.D(z)
        vae_tc_loss = (d_z[:, 0] - d_z[:, 1]).mean()
        # VAE loss
        vae_loss = vae_recon_loss + self.kld_weight * kld_loss + self.tc_weight * vae_tc_loss

        # VAE backward pass
        vae_optimizer.zero_grad()
        self.manual_backward(vae_loss, retain_graph=True)
        # vae_optimizer.step()

        # Discriminator forward pass
        mean_2, logvar_2 = self.VAE.encode(x_2)
        z_2 = self.VAE.reparameterize(mean_2, logvar_2)
        z_2_perm = permute_dims(z_2)
        d_z_2_perm = self.D(z_2_perm.detach())
        # print(d_z.shape, zeros.shape, d_z_2_perm.shape, ones.shape)
        d_tc_loss = 0.5 * (F.cross_entropy(d_z, zeros) + F.cross_entropy(d_z_2_perm, ones))

        # Discriminator backward pass
        d_optimizer.zero_grad()
        self.manual_backward(d_tc_loss)
        vae_optimizer.step()
        d_optimizer.step()

        # log the losses
        self.log_dict({
            "vae_loss": vae_loss,
            "vae_recon_loss": vae_recon_loss,
            "vae_kld_loss": kld_loss,
            "vae_tc_loss": vae_tc_loss,
            "d_tc_loss": d_tc_loss
        }, on_step=False, on_epoch=True)

    
    def validation_step(self, batch, batch_idx):
        # get the batch
        x_1, x_2 = batch
        batch_size = x_1.shape[0]
        # print('batch_size:', batch_size)

        # create a batch of ones and zeros for the discriminator
        ones = torch.ones(batch_size, dtype=torch.long, device=self.device)
        zeros = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        # VAE forward pass
        x_recon, mean, logvar, z = self.VAE(x_1)
        # print('x_recon:', x_recon.shape)
        # print('mean:', mean.shape)
        # print('logvar:', logvar.shape)
        # print('z:', z.shape)
        vae_recon_loss = self.mse(x_recon, x_1)
        # print('vae_recon_loss:', vae_recon_loss)
        kld_loss = self.kld(mean, logvar)
        # print('kld_loss:', kld_loss)
        # VAE TC loss
        d_z = self.D(z)
        # print('d_z:', d_z.shape)
        vae_tc_loss = (d_z[:, 0] - d_z[:, 1]).mean()
        # print('vae_tc_loss:', vae_tc_loss)
        # VAE loss
        vae_loss = vae_recon_loss + self.kld_weight * kld_loss + self.tc_weight * vae_tc_loss
        # print('vae_loss:', vae_loss)

        # Discriminator forward pass
        mean_2, logvar_2 = self.VAE.encode(x_2)
        # print('mean_2:', mean_2.shape)
        # print('logvar_2:', logvar_2.shape)
        z_2 = self.VAE.reparameterize(mean_2, logvar_2)
        # print('z_2:', z_2.shape)
        z_2_perm = permute_dims(z_2)
        # print('z_2_perm:', z_2_perm.shape)
        d_z_2_perm = self.D(z_2_perm)
        # print('d_z_2_perm:', d_z_2_perm.shape)
        # print(d_z.shape, zeros.shape, d_z_2_perm.shape, ones.shape)
        d_tc_loss = 0.5 * (F.cross_entropy(d_z, zeros) + F.cross_entropy(d_z_2_perm, ones))

        # log the losses
        self.log_dict({
            "val_vae_loss": vae_loss,
            "val_vae_recon_loss": vae_recon_loss,
            "val_vae_kld_loss": kld_loss,
            "val_vae_tc_loss": vae_tc_loss,
            "val_d_tc_loss": d_tc_loss
        }, on_step=False, on_epoch=True)


    def configure_optimizers(self):
        vae_optimizer = torch.optim.Adam(self.VAE.parameters(), lr=self.hparams.lr_vae)
        d_optimizer = torch.optim.Adam(self.D.parameters(), lr=self.hparams.lr_d)
        return vae_optimizer, d_optimizer