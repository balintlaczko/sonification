import torch
from torch import nn
from torch.nn import functional as F


class LinearEncoder(nn.Module):
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


class LinearDecoder(nn.Module):
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


class ConvEncoder(nn.Module):
    def __init__(self, in_channels, latent_size, layers_channels=[16, 32, 64, 128, 256, 512], input_size=512):
        super().__init__()
        self.in_channels = in_channels # 2 for red and green
        self.latent_size = latent_size

        layers = []
        in_channel = self.in_channels
        for out_channel in layers_channels:
            layers.extend([
                nn.Conv2d(in_channel, out_channel, 3, stride=2, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.LeakyReLU(0.2)
            ])
            in_channel = out_channel

        # Calculate the size of the feature map when it reaches the linear layer
        feature_map_size = input_size // (2 ** len(layers_channels))
        
        layers.extend([
            nn.Flatten(),
            nn.Linear(layers_channels[-1] * feature_map_size * feature_map_size, self.latent_size),
            nn.BatchNorm1d(self.latent_size),
            nn.LeakyReLU(0.2),
        ])

        self.layers = nn.Sequential(*layers)
    

    def forward(self, x):
        return self.layers(x)
    

class ConvDecoder(nn.Module):
    def __init__(self, latent_size, out_channels, layers_channels=[512, 256, 128, 64, 32, 16], output_size=512):
        super().__init__()
        self.latent_size = latent_size
        self.out_channels = out_channels

        # Calculate the size of the feature map when it reaches the linear layer
        feature_map_size = output_size // (2 ** len(layers_channels))

        layers = [
            nn.Linear(latent_size, layers_channels[0] * feature_map_size * feature_map_size),
            nn.BatchNorm1d(layers_channels[0] * feature_map_size * feature_map_size),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (layers_channels[0], feature_map_size, feature_map_size)),
        ]

        in_channel = layers_channels[0]
        for out_channel in layers_channels[1:]:
            layers.extend([
                nn.ConvTranspose2d(in_channel, out_channel, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_channel),
                nn.LeakyReLU(0.2),
            ])
            in_channel = out_channel

        layers.extend([
            nn.ConvTranspose2d(layers_channels[-1], out_channels, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid(),
        ])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)