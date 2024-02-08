import torch
from torch import nn
from layers import LinearEncoder, LinearDecoder, ConvEncoder, ConvDecoder


class AE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, output_size):
        super().__init__()
        self.encoder = LinearEncoder(input_size, hidden_size, latent_size)
        self.decoder = LinearDecoder(latent_size, hidden_size, output_size)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)


# inspired by https://medium.com/@rekalantar/variational-auto-encoder-vae-pytorch-tutorial-dce2d2fe0f5f
class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super().__init__()
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
    def __init__(self, in_channels, latent_size):
        super().__init__()
        self.encoder = ConvEncoder(in_channels, latent_size)
        self.mu = nn.Linear(latent_size, latent_size)
        self.logvar = nn.Linear(latent_size, latent_size)
        self.decoder = ConvDecoder(latent_size, in_channels)
    
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