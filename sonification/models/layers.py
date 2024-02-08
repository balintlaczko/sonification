import torch
from torch import nn
from functools import reduce


class LinearEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(LinearEncoder, self).__init__()
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
        super(LinearDecoder, self).__init__()
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
        super(ConvEncoder, self).__init__()
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
        super(ConvDecoder, self).__init__()
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
    

class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super(ResBlock, self).__init__()

        # this is the residual block
        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            # nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
            # nn.BatchNorm2d(in_channel),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input  # skip connection

        return out
    

class MultiScaleEncoder(nn.Module):
    def __init__(
            self,
            in_channel=1,
            channel=128,
            n_res_block=1,
            n_res_channel=32,
            stride=4,
            kernels=[4, 4],
            input_dim_h=80,
            input_dim_w=188,
    ):
        super(MultiScaleEncoder, self).__init__()

        # check that the stride is valid
        assert stride in [2, 4]

        # check that kernels is a list with even number of elements
        assert len(kernels) % 2 == 0

        # group kernels into pairs
        kernels = [kernels[i:i + 2] for i in range(0, len(kernels), 2)]

        # save input dimension for later use
        self.input_dim_h = input_dim_h
        self.input_dim_w = input_dim_w

        # create a list of lanes
        self.lanes = nn.ModuleList()

        # create a lane for each kernel size
        for kernel in kernels:
            padding = [kernel_side // 2 - 1 for kernel_side in kernel]
            lane = None

            if stride == 4:
                # base block: in -> out/2 -> out -> out
                lane = [
                    nn.Conv2d(in_channel, channel // 2, kernel,
                              stride=2, padding=padding),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channel // 2, channel, kernel,
                              stride=2, padding=padding),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channel, channel, 3, padding=1),
                ]

            elif stride == 2:
                # base block: in -> out/2 -> out
                lane = [
                    nn.Conv2d(in_channel, channel // 2, kernel,
                              stride=2, padding=padding),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channel // 2, channel, 3, padding=1),
                ]

            # add residual blocks
            lane.extend([ResBlock(channel, n_res_channel)
                         for _ in range(n_res_block)])

            # add final ReLU
            lane.append(nn.ReLU(inplace=True))

            # add to list of blocks
            self.lanes.append(nn.Sequential(*lane))

    def forward(self, input):
        # reducing with this so the "+" still means whatever it should
        def add_lane(x, y):
            return x + y

        # apply each block to the input, then sum the results
        return reduce(add_lane, [lane(input) for lane in self.lanes])
    

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP, self).__init__()

        # create a chain of ints for the dimensions of the MLP
        self.dims = [hidden_dim, ] * (num_layers + 1)
        # mark first and last as the input and output dimensions
        self.dims[0], self.dims[-1] = input_dim, output_dim
        # create a flat list of layers reading the dims pairwise
        layers = []
        for i in range(num_layers):
            layers.extend(self.mlp_layer(self.dims[i], self.dims[i+1]))

        self.layers = nn.Sequential(*layers)

    def mlp_layer(self, input_dim, output_dim) -> list:
        return [
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.LeakyReLU(),
        ]

    def forward(self, x):
        return self.layers(x)
    
