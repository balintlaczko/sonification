import torch
from torch import nn
from functools import reduce


class LinearEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, num_layers=4, bias=False, dtype=torch.float32):
        super(LinearEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        layers = []
        for i in range(num_layers-1):
            layers.extend([
                nn.Linear(input_size, hidden_size, bias=bias, dtype=dtype),
                nn.BatchNorm1d(hidden_size),
                nn.LeakyReLU(0.2),
            ])
            input_size = hidden_size

        layers.extend([
            nn.Linear(hidden_size, latent_size, bias=bias, dtype=dtype),
            nn.BatchNorm1d(latent_size),
            nn.LeakyReLU(0.2),
        ])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class LinearDecoder(nn.Module):
    def __init__(self, latent_size, hidden_size, output_size, num_layers=4, bias=False, dtype=torch.float32):
        super(LinearDecoder, self).__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        layers = []
        for i in range(num_layers-1):
            layers.extend([
                nn.Linear(latent_size, hidden_size, bias=bias, dtype=dtype),
                nn.BatchNorm1d(hidden_size),
                nn.LeakyReLU(0.2),
            ])
            latent_size = hidden_size

        layers.extend([
            nn.Linear(hidden_size, output_size, bias=bias, dtype=dtype),
            nn.BatchNorm1d(output_size),
            nn.Sigmoid(),
        ])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ConvEncoder(nn.Module):
    def __init__(self, in_channels, output_size, layers_channels=[16, 32, 64, 128, 256, 512], input_size=512):
        super(ConvEncoder, self).__init__()
        self.in_channels = in_channels  # 2 for red and green
        self.output_size = output_size

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
            nn.Linear(
                layers_channels[-1] * feature_map_size * feature_map_size, self.output_size),
            nn.BatchNorm1d(self.output_size),
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
            nn.Linear(
                latent_size, layers_channels[0] * feature_map_size * feature_map_size),
            nn.BatchNorm1d(layers_channels[0] *
                           feature_map_size * feature_map_size),
            nn.LeakyReLU(0.2),
            nn.Unflatten(
                1, (layers_channels[0], feature_map_size, feature_map_size)),
        ]

        in_channel = layers_channels[0]
        for out_channel in layers_channels[1:]:
            layers.extend([
                nn.ConvTranspose2d(in_channel, out_channel,
                                   3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_channel),
                nn.LeakyReLU(0.2),
            ])
            in_channel = out_channel

        layers.extend([
            nn.ConvTranspose2d(
                layers_channels[-1], out_channels, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid(),
        ])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ConvEncoder1D(nn.Module):
    def __init__(self, in_channels, output_size, kernel_size=3, layers_channels=[16, 32, 64, 128, 256, 512], input_size=512, dropout=0.2):
        super(ConvEncoder1D, self).__init__()
        self.in_channels = in_channels
        self.output_size = output_size

        layers = []
        in_channel = self.in_channels
        if type(kernel_size) is int:
            kernel_size = [kernel_size] * len(layers_channels)
        for idx, out_channel in enumerate(layers_channels):
            k_size = kernel_size[idx]
            padding = k_size // 2
            layers.extend([
                nn.Conv1d(in_channel, out_channel, kernel_size=k_size, stride=2, padding=padding),
                nn.BatchNorm1d(out_channel),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
            ])
            in_channel = out_channel

        # Calculate the size of the feature map when it reaches the linear layer
        feature_map_size = input_size // (2 ** len(layers_channels))

        layers.extend([
            nn.Flatten(),
            nn.Linear(
                layers_channels[-1] * feature_map_size, self.output_size),
            nn.BatchNorm1d(self.output_size),
            nn.LeakyReLU(0.2),
        ])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    

class ConvEncoder1DRes(nn.Module):
    def __init__(self, in_channels, output_size, layers_channels=[16, 32, 64, 128, 256, 512], input_size=512):
        super(ConvEncoder1DRes, self).__init__()
        self.in_channels = in_channels
        self.output_size = output_size

        layers = []
        in_channel = self.in_channels
        for idx, hidden_channel in enumerate(layers_channels):
            layers.extend([
                ResBlock1D(in_channel, hidden_channel),
            ])

        layers.extend([
            nn.Flatten(),
            # nn.Linear(
            #     input_size, self.output_size),
            # nn.BatchNorm1d(self.output_size),
            # nn.LeakyReLU(0.2),
        ])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ConvDecoder1D(nn.Module):
    def __init__(self, latent_size, out_channels, kernel_size=3, layers_channels=[512, 256, 128, 64, 32, 16], output_size=512, dropout=0.2):
        super(ConvDecoder1D, self).__init__()
        self.latent_size = latent_size
        self.out_channels = out_channels

        # Calculate the size of the feature map when it reaches the linear layer
        feature_map_size = output_size // (2 ** len(layers_channels))

        layers = [
            nn.Linear(
                latent_size, layers_channels[0] * feature_map_size),
            nn.BatchNorm1d(layers_channels[0] *
                           feature_map_size),
            nn.LeakyReLU(0.2),
            nn.Unflatten(
                1, (layers_channels[0], feature_map_size)),
        ]

        in_channel = layers_channels[0]
        if type(kernel_size) is int:
            kernel_size = [kernel_size] * len(layers_channels)
        # reverse kernel size list
        kernel_size = kernel_size[::-1]
        for idx, out_channel in enumerate(layers_channels[1:]):
            k_size = kernel_size[idx]
            padding = k_size // 2
            layers.extend([
                nn.ConvTranspose1d(in_channel, out_channel,
                                   kernel_size=k_size, stride=2, padding=padding, output_padding=1),
                nn.BatchNorm1d(out_channel),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
            ])
            in_channel = out_channel

        k_size = kernel_size[-1]
        padding = k_size // 2
        layers.extend([
            nn.ConvTranspose1d(
                layers_channels[-1], out_channels, kernel_size=k_size, stride=2, padding=padding, output_padding=1),
            nn.BatchNorm1d(out_channels),
            nn.Sigmoid(),
        ])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    

class ConvDecoder1DRes(nn.Module):
    def __init__(self, latent_size, out_channels, layers_channels=[512, 256, 128, 64, 32, 16], output_size=512):
        super(ConvDecoder1DRes, self).__init__()
        self.latent_size = latent_size
        self.out_channels = out_channels

        layers = [
            nn.Linear(
                latent_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.LeakyReLU(0.2),
            nn.Unflatten(
                1, (1, output_size)),
        ]

        for idx, hidden_channel in enumerate(layers_channels):
            layers.extend([
                ResBlock1DTr(out_channels, hidden_channel),
            ])

        layers.extend([
            nn.ConvTranspose1d(
                out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.Sigmoid(),
        ])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel, chans_per_group=16):
        super(ResBlock, self).__init__()

        # this is the residual block
        self.conv = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            # nn.BatchNorm2d(channel),
            nn.GroupNorm(channel // chans_per_group, channel),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel, in_channel, 1),
            # nn.BatchNorm2d(in_channel),
            nn.GroupNorm(in_channel // chans_per_group, in_channel),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input  # skip connection
        return out


class LinearResBlock(nn.Module):
    def __init__(self, in_features, features, feats_per_group=16):
        super().__init__()

        # this is the residual block
        self.block = nn.Sequential(
            nn.Linear(in_features, features),
            nn.GroupNorm(features // feats_per_group, features),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(features, in_features),
            nn.GroupNorm(in_features // feats_per_group, in_features),
            nn.LeakyReLU(0.2, inplace=False),
        )
    
    def forward(self, input):
        out = self.block(input)
        out += input
        return out
    

class ResBlock1D(nn.Module):
    def __init__(self, in_channel, channel, chans_per_group=16):
        super(ResBlock1D, self).__init__()

        # this is the residual block
        self.conv = nn.Sequential(
            nn.Conv1d(in_channel, channel, 3, padding=1),
            # nn.BatchNorm1d(channel),
            nn.GroupNorm(channel // chans_per_group, channel),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channel, in_channel, 1),
            # nn.BatchNorm1d(in_channel),
            nn.GroupNorm(in_channel // chans_per_group, in_channel),
            nn.LeakyReLU(0.2),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input  # skip connection
        return out
    

class ResBlock1DTr(nn.Module):
    def __init__(self, in_channel, channel):
        super(ResBlock1DTr, self).__init__()

        # this is the residual block
        self.conv = nn.Sequential(
            nn.ConvTranspose1d(in_channel, channel, 3, padding=1),
            nn.BatchNorm1d(channel),
            nn.ReLU(),
            nn.ConvTranspose1d(channel, in_channel, 1),
            nn.BatchNorm1d(in_channel),
            nn.ReLU(),
        )

    def forward(self, input):
        out = self.conv(input)
        out = out + input  # skip connection

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
            padding = [max(1, (kernel_side - 1) // 2) for kernel_side in kernel]
            lane = None

            if stride == 4:
                # base block: in -> out/2 -> out -> out
                lane = [
                    nn.Conv2d(in_channel, channel // 2, kernel, stride=2, padding=padding),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(channel // 2, channel, kernel, stride=2, padding=padding),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(channel, channel, 3, stride=[1, 2], padding=1),
                ]

            elif stride == 2:
                # base block: in -> out/2 -> out
                lane = [
                    nn.Conv2d(in_channel, channel // 2, kernel,
                              stride=2, padding=padding),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(channel // 2, channel, 3, padding=1),
                ]

            # add residual blocks
            lane.extend([ResBlock(channel, n_res_channel)
                         for _ in range(n_res_block)])

            # add final ReLU
            lane.append(nn.LeakyReLU(inplace=True))

            # add to list of blocks
            self.lanes.append(nn.Sequential(*lane))
    
    def forward(self, input):
        # Process all lanes and stack the results
        outputs = [lane(input) for lane in self.lanes]
        
        # Use torch's built-in sum function to add all outputs together
        if len(outputs) == 1:
            return outputs[0]
        else:
            return torch.stack(outputs).sum(dim=0)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, relu_inplace=False):
        super(MLP, self).__init__()
        self.relu_inplace = relu_inplace
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
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(0.2, inplace=self.relu_inplace),
            nn.Dropout(0.2)
        ]

    def forward(self, x):
        return self.layers(x)


class LinearDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LinearDiscriminator, self).__init__()

        layers = []
        for i in range(num_layers-1):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            input_dim = hidden_dim
        layers.extend([
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        ])

        self.discriminator = nn.Sequential(*layers)

    def forward(self, x):
        return self.discriminator(x)
    

class LinearDiscriminator_w_dropout(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.0):
        super(LinearDiscriminator_w_dropout, self).__init__()

        layers = []
        for i in range(num_layers-1):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout),
            ])
            input_dim = hidden_dim
        layers.extend([
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        ])

        self.discriminator = nn.Sequential(*layers)

    def forward(self, x):
        return self.discriminator(x)
    

class LinearCritique(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LinearCritique, self).__init__()

        layers = []
        for i in range(num_layers-1):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            input_dim = hidden_dim
        self.critique = nn.Sequential(*layers)
        self.discriminator = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.critique(x)
        return x, self.discriminator(x)
    
class LinearCritique_w_dropout(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.0):
        super(LinearCritique_w_dropout, self).__init__()

        layers = []
        for i in range(num_layers-1):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout),
            ])
            input_dim = hidden_dim
        self.critique = nn.Sequential(*layers)
        self.discriminator = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.critique(x)
        return x, self.discriminator(x)


class LinearProjector(nn.Module):
    def __init__(self, in_features, out_features, hidden_layers_features=[64, 128, 256, 128, 64]):
        super(LinearProjector, self).__init__()

        layers = []

        # add first layer
        layers.extend([
            nn.Linear(in_features, hidden_layers_features[0]),
            nn.LeakyReLU(0.2, inplace=True),
        ])

        # add hidden layers
        for i in range(len(hidden_layers_features)-1):
            layers.extend([
                nn.Linear(hidden_layers_features[i],
                          hidden_layers_features[i+1]),
                nn.LeakyReLU(0.2, inplace=True),
            ])

        # add output layer
        layers.append(nn.Linear(hidden_layers_features[-1], out_features))

        self.linear_projector = nn.Sequential(*layers)

    def forward(self, x):
        return self.linear_projector(x)
