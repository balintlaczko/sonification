import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size, dtype=torch.float32),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, latent_size),
            nn.BatchNorm1d(latent_size),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.layers(x)
    

class ImgEncoder(nn.Module):
    def __init__(self, in_channels, latent_size):
        super().__init__()
        self.in_channels = in_channels # 2 for red and green
        self.latent_size = latent_size
        self.layers = nn.Sequential(
            nn.Conv2d(self.in_channels, 16, 3, stride=2, padding=1), # in shape: (512, 512, 2) out shape: (256, 256, 16)
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # in shape: (256, 256, 16) out shape: (128, 128, 32)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # in shape: (128, 128, 32) out shape: (64, 64, 64)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # in shape: (64, 64, 64) out shape: (32, 32, 128)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), # in shape: (32, 32, 128) out shape: (16, 16, 256)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 3, stride=2, padding=1), # in shape: (16, 16, 256) out shape: (8, 8, 512)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(512*8*8, latent_size),
            nn.BatchNorm1d(latent_size),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, latent_size, hidden_size, output_size):
        super().__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layers = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


class ImgDecoder(nn.Module):
    def __init__(self, latent_size, out_channels):
        super().__init__()
        self.latent_size = latent_size
        self.out_channels = out_channels # 2 for red and green
        self.layers = nn.Sequential(
            nn.Linear(latent_size, 512*8*8),
            nn.BatchNorm1d(512*8*8),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (512, 8, 8)),
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1), # in shape: (8, 8, 512) out shape: (16, 16, 256)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), # in shape: (16, 16, 256) out shape: (32, 32, 128)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), # in shape: (32, 32, 128) out shape: (64, 64, 64)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), # in shape: (64, 64, 64) out shape: (128, 128, 32)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # in shape: (128, 128, 32) out shape: (256, 256, 16)
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(16, out_channels, 3, stride=2, padding=1, output_padding=1), # in shape: (256, 256, 16) out shape: (512, 512, 2)
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.layers(x)


class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, output_size):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size, latent_size)
        self.decoder = Decoder(latent_size, hidden_size, output_size)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)

# inspired by https://medium.com/@rekalantar/variational-auto-encoder-vae-pytorch-tutorial-dce2d2fe0f5f


class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size, latent_size)
        self.mu = nn.Linear(latent_size, latent_size)
        self.logvar = nn.Linear(latent_size, latent_size)
        self.decoder = Decoder(latent_size, hidden_size, input_size)

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


class ImgVAE(nn.Module):
    def __init__(self, in_channels, latent_size):
        super().__init__()
        self.encoder = ImgEncoder(in_channels, latent_size)
        self.mu = nn.Linear(latent_size, latent_size)
        self.logvar = nn.Linear(latent_size, latent_size)
        self.decoder = ImgDecoder(latent_size, in_channels)
    
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