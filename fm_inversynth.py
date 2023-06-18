# imports
import torch
from torch import tensor, nn
import torchaudio
import torchyin
from torchsynth.module import (
    SineVCO,
    FmVCO,
)
from utils import *
from functools import reduce
import ddsp_utils


class FmSynth(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.fm_vco = FmVCO(
            tuning=tensor([0.0] * config.batch_size),
            mod_depth=tensor([0.0] * config.batch_size),
            synthconfig=config,
            device=device,
        )
        self.modulator = SineVCO(
            tuning=tensor([0.0] * config.batch_size),
            mod_depth=tensor([0.0] * config.batch_size),
            initial_phase=tensor([torch.pi / 2] * config.batch_size),
            synthconfig=config,
            device=device,
        )

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=80,
            normalized=True,
            f_min=20.0,
            f_max=8000.0,
        ).to(device)

    def forward(
            self,
            carrier_frequency: torch.Tensor,
            modulator_frequency: torch.Tensor,
            modulation_index: torch.Tensor
    ):
        # generate modulator signal
        modulator_signal = self.modulator(
            frequency2midi_tensor(modulator_frequency))

        # set modulation index
        self.fm_vco.set_parameter(
            "mod_depth", torch.clip(modulation_index, -96, 96))

        # generate fm signal
        fm_signal = self.fm_vco(frequency2midi_tensor(
            carrier_frequency), modulator_signal)

        # generate mel spectrogram
        mel_spec = self.mel_transform(fm_signal)

        return mel_spec


class FmSynth2Wave(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.fm_vco = FmVCO(
            tuning=tensor([0.0] * config.batch_size),
            mod_depth=tensor([0.0] * config.batch_size),
            synthconfig=config,
            device=device,
        )
        self.modulator = SineVCO(
            tuning=tensor([0.0] * config.batch_size),
            mod_depth=tensor([0.0] * config.batch_size),
            initial_phase=tensor([torch.pi / 2] * config.batch_size),
            synthconfig=config,
            device=device,
        )

    def forward(
            self,
            carrier_frequency: torch.Tensor,
            modulator_frequency: torch.Tensor,
            modulation_index: torch.Tensor
    ):
        # generate modulator signal
        modulator_signal = self.modulator(
            frequency2midi_tensor(modulator_frequency))

        # set modulation index
        self.fm_vco.set_parameter(
            "mod_depth", torch.clip(modulation_index, -96, 96))

        # generate fm signal
        fm_signal = self.fm_vco(frequency2midi_tensor(
            carrier_frequency), modulator_signal)

        return fm_signal


class Mel2Params(nn.Module):
    def __init__(self, num_mels, num_hops, num_params, dim=512):
        super().__init__()

        self.num_mels = num_mels
        self.num_hops = num_hops
        self.num_params = num_params

        self.layers = nn.Sequential(
            nn.Linear(self.num_mels * self.num_hops, dim),
            nn.ReLU(),
            nn.Linear(dim, num_params),
            # nn.ReLU(),
        )

    def forward(self, mel_spec):
        mel_spec = mel_spec.reshape(-1, self.num_mels * self.num_hops)
        params = self.layers(mel_spec)
        return params


class Mel2Params2(nn.Module):
    def __init__(self, num_mels, num_hops, num_params, dim=256):
        super().__init__()

        self.num_mels = num_mels
        self.num_hops = num_hops
        self.num_params = num_params

        self.mel_encoder = Encoder(input_dim_h=num_mels, input_dim_w=num_hops)

        self.layers = nn.Sequential(
            nn.Linear(128 * 20 * 47, dim),
            nn.ReLU(),
            nn.Linear(dim, num_params),
            # nn.ReLU(),
        )

    def forward(self, mel_spec):
        # print("mel_spec.shape", mel_spec.shape)
        mel_encoded = self.mel_encoder(mel_spec.unsqueeze(1))
        # print("mel_encoded.shape", mel_encoded.shape)
        mel_flattened = mel_encoded.view(mel_encoded.shape[0], -1)
        # print("mel_flattened.shape", mel_flattened.shape)
        params = self.layers(mel_flattened)
        return params


class Wave2MFCCEncoder(nn.Module):
    def __init__(
            self,
            sr=48000,
            n_mels=160,
            n_mfcc=40,
            n_fft=2048,
            hop_length=512,
            normalized=True,
            f_min=20.0,
            f_max=8000.0,
            z_dim=32
    ):
        super().__init__()

        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=sr,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "hop_length": hop_length,
                "n_mels": n_mels,
                "normalized": normalized,
                "f_min": f_min,
                "f_max": f_max,
            })

        # self.norm = nn.LayerNorm(n_mfcc)

        self.fc = nn.Linear(n_mfcc, z_dim)

    def forward(self, x):
        mfcc = self.mfcc(x)
        mfcc_mean = mfcc.mean(dim=-1)
        # mfcc_norm = self.norm(mfcc_mean)
        z = self.fc(mfcc_mean)
        return z


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()

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


class MFCCEncoder(nn.Module):
    def __init__(
            self,
            sr=48000,
            n_mels=80,
            n_mfcc=30,
            n_fft=2048,
            hop_length=512,
            normalized=True,
            f_min=20.0,
            f_max=20000.0,
            gru_hidden_dim=512,
            mlp_in_dim=16,
            mlp_out_dim=256,
            mlp_layers=3
    ):
        super().__init__()

        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=sr,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "hop_length": hop_length,
                "n_mels": n_mels,
                "normalized": normalized,
                "f_min": f_min,
                "f_max": f_max,
            })

        self.layernorm = nn.LayerNorm(n_mfcc)
        self.gru = nn.GRU(n_mfcc, gru_hidden_dim, batch_first=True)
        self.linear = nn.Linear(gru_hidden_dim, mlp_in_dim)
        self.mlp = MLP(mlp_in_dim, mlp_out_dim, mlp_out_dim, mlp_layers)

    def forward(self, y):
        mfcc = self.mfcc(y)
        mfcc = self.layernorm(torch.transpose(mfcc, -2, -1))
        mfcc, _ = self.gru(mfcc)
        mfcc = self.linear(mfcc)
        mfcc = self.mlp(mfcc)
        return mfcc


class MelbandsEncoder(nn.Module):
    def __init__(
            self,
            sr=48000,
            n_mels=80,
            n_fft=2048,
            hop_length=512,
            normalized=True,
            f_min=20.0,
            f_max=20000.0,
            gru_hidden_dim=512,
            mlp_in_dim=16,
            mlp_out_dim=256,
            mlp_layers=3
    ):
        super().__init__()

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            normalized=normalized,
            f_min=f_min,
            f_max=f_max,
        )

        self.layernorm = nn.LayerNorm(n_mels)
        self.gru = nn.GRU(n_mels, gru_hidden_dim, batch_first=True)
        self.linear = nn.Linear(gru_hidden_dim, mlp_in_dim)
        self.mlp = MLP(mlp_in_dim, mlp_out_dim, mlp_out_dim, mlp_layers)

    def forward(self, y):
        melspec = self.melspec(y)
        melspec = self.layernorm(torch.transpose(melspec, -2, -1))
        melspec, _ = self.gru(melspec)
        melspec = self.linear(melspec)
        melspec = self.mlp(melspec)
        return melspec


class PitchEncoder(nn.Module):
    def __init__(
            self,
            sr=48000,
            f_min=20.0,
            f_max=20000.0,
            num_frames: int = None,
            mlp_out_dim=256,
            mlp_layers=3
    ):
        super().__init__()

        self.sr = sr
        self.f_min = f_min
        self.f_max = f_max
        self.num_frames = num_frames

        self.mlp = MLP(1, mlp_out_dim, mlp_out_dim, mlp_layers)

    def resize_interp(self, x, out_size):
        x_3d = x.unsqueeze(1)
        y_3d = torch.nn.functional.interpolate(
            x_3d, size=out_size, mode='linear', align_corners=True)
        y = y_3d.squeeze(1)
        return y

    def forward(self, y):
        pitch = torchyin.estimate(
            y, self.sr, pitch_min=self.f_min, pitch_max=self.f_max)
        if self.num_frames is not None:
            pitch = self.resize_interp(pitch, self.num_frames)
        pitch = pitch.unsqueeze(-1)
        pitch = self.mlp(pitch)
        return pitch


# function to calculate number of hops
def num_hops(
        buffer_length: int,
        hop_length: int
) -> int:
    return int(buffer_length / hop_length) + 1


class Wave2Params(nn.Module):
    def __init__(
            self,
            sr=48000,
            n_mels=80,
            n_mfcc=30,
            n_fft=2048,
            hop_length=512,
            normalized=True,
            f_min=20.0,
            f_max=20000.0,
            gru_hidden_dim=512,
            mlp_in_dim=16,
            mlp_out_dim=256,
            mlp_layers=3,
            buffer_length_s: int = 4,

    ):
        super().__init__()

        self.sr = sr
        self.buffer_length_s = buffer_length_s

        # feature extraction: Mel spectrogram, MFCCs, pitch
        self.melbands_encoder = MelbandsEncoder(
            sr=sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            normalized=normalized,
            f_min=f_min,
            f_max=f_max,
            gru_hidden_dim=gru_hidden_dim,
            mlp_in_dim=mlp_in_dim,
            mlp_out_dim=mlp_out_dim,
            mlp_layers=mlp_layers
        )

        self.mfcc_encoder = MFCCEncoder(
            sr=sr,
            n_mels=n_mels,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
            normalized=normalized,
            f_min=f_min,
            f_max=f_max,
            gru_hidden_dim=gru_hidden_dim,
            mlp_in_dim=mlp_in_dim,
            mlp_out_dim=mlp_out_dim,
            mlp_layers=mlp_layers
        )

        self.pitch_encoder = PitchEncoder(
            sr=sr,
            f_min=f_min,
            f_max=f_max,
            mlp_out_dim=mlp_out_dim,
            mlp_layers=mlp_layers,
        )

        # synth params from the concatenated encoded features
        n_hops = num_hops(buffer_length_s * sr, hop_length)
        self.synth_params = nn.Sequential(
            nn.Linear(3 * mlp_out_dim * n_hops, 3),
            nn.ReLU(),
        )
        # synth
        self.synth = ddsp_utils.FMSynth(sr=sr)

    def forward(self, y):
        # extract & encode features
        melbands_encoded = self.melbands_encoder(y)
        mfcc_encoded = self.mfcc_encoder(y)
        # set the number of frames for pitch encoding to the number of fft hops
        self.pitch_encoder.num_frames = mfcc_encoded.shape[-2]
        pitch = self.pitch_encoder(y)
        # concatenate features
        encoded = torch.cat(
            (melbands_encoded, mfcc_encoded, pitch), dim=-1)
        # flatten encoded
        encoded_flatten = torch.flatten(encoded, start_dim=-2)
        # predict synth params
        synth_params = self.synth_params(encoded_flatten)
        # take each param from all batches and repeat it for the number of samples in the buffer
        carr_freq = synth_params[:,
                                 0].unsqueeze(-1).repeat(1, self.buffer_length_s * self.sr)
        harm_ratio = synth_params[:,
                                  1].unsqueeze(-1).repeat(1, self.buffer_length_s * self.sr)
        mod_index = synth_params[:,
                                 2].unsqueeze(-1).repeat(1, self.buffer_length_s * self.sr)
        # generate synth buffer
        synth_buffer = self.synth(carr_freq, harm_ratio, mod_index)

        return synth_buffer, synth_params


class FM_Autoencoder_Wave(nn.Module):
    def __init__(self, config, device, z_dim=32, mlp_dim=512):
        super().__init__()

        self.encoder = Wave2MFCCEncoder(z_dim=z_dim)
        self.decoder = MLP(z_dim, mlp_dim, mlp_dim, 3)
        self.synthparams = nn.Linear(mlp_dim, 3)
        self.synth_act = nn.ReLU()
        self.synth = FmSynth2Wave(config, device)

    def forward(self, x):
        z = self.encoder(x)
        mlp_out = self.decoder(z)
        params = self.synthparams(mlp_out)
        params = self.synth_act(params)
        # print("params.shape", params.shape)
        # print("params", params)
        carrier_frequency = params[:, 0]
        # print("carrier_frequency", carrier_frequency)
        modulator_frequency = params[:, 1]
        # print("modulator_frequency", modulator_frequency)
        modulation_index = params[:, 2]
        # print("modulation_index", modulation_index)
        # print("mod index sum:", modulation_index.sum())
        fm_signal = self.synth(
            carrier_frequency, modulator_frequency, modulation_index)
        return fm_signal


class FM_Autoencoder_Wave2(nn.Module):
    def __init__(self, config, device, z_dim=32, mlp_dim=512):
        super().__init__()

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=80,
            normalized=True,
            f_min=20.0,
            f_max=8000.0,
        ).to(device)

        self.spec_encoder = Encoder(input_dim_h=80, input_dim_w=188)

        self.z = nn.Sequential(
            nn.Linear(128 * 20 * 47, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, z_dim),
            nn.ReLU(),
        )

        self.decoder = MLP(z_dim, mlp_dim, mlp_dim, 3)
        self.synthparams = nn.Linear(mlp_dim, 3)
        self.synth_act = nn.ReLU()
        self.synth = FmSynth2Wave(config, device)

    def forward(self, x):
        # print("x.shape", x.shape)
        mel_spec = self.mel_transform(x)
        # print("mel_spec.shape", mel_spec.shape)
        mel_spec = mel_spec.unsqueeze(1)
        encoded = self.spec_encoder(mel_spec)
        # print("encoded.shape", encoded.shape)
        encoded_flatten = encoded.view(encoded.shape[0], -1)
        z = self.z(encoded_flatten)
        # print("z.shape", z.shape)
        mlp_out = self.decoder(z)
        # print("mlp_out.shape", mlp_out.shape)
        params = self.synthparams(mlp_out)
        params = self.synth_act(params)
        # print("params.shape", params.shape)
        # print("params.shape", params.shape)
        # print("params", params)
        carrier_frequency = params[:, 0]
        # print("carrier_frequency", carrier_frequency)
        modulator_frequency = params[:, 1]
        # print("modulator_frequency", modulator_frequency)
        modulation_index = params[:, 2]
        # print("modulation_index", modulation_index)
        # print("mod index sum:", modulation_index.sum())
        fm_signal = self.synth(
            carrier_frequency, modulator_frequency, modulation_index)
        return fm_signal


class FM_Autoencoder(nn.Module):
    def __init__(
            self,
            config,
            device,
            num_mels=80,
            num_hops=188,
            num_params=3,
            dim=256
    ):
        super().__init__()

        self.num_mels = num_mels
        self.num_hops = num_hops
        self.num_params = num_params

        # self.encoder = Mel2Params(num_mels, num_hops, num_params, dim=dim)
        self.encoder = Mel2Params2(num_mels, num_hops, num_params, dim=dim)
        self.synth = FmSynth(config, device)

    def forward(self, mel_spec):
        params = self.encoder(mel_spec)
        carrier_frequency = params[:, 0]
        modulator_frequency = params[:, 1]
        modulation_index = params[:, 2]
        mel_spec_recon = self.synth(
            carrier_frequency,
            modulator_frequency,
            modulation_index
        )
        return mel_spec_recon, params


class FM_Param_Autoencoder(nn.Module):
    def __init__(
            self,
            config,
            device,
            num_mels=80,
            num_hops=188,
            num_params=3,
            dim=256
    ):
        super().__init__()

        self.num_mels = num_mels
        self.num_hops = num_hops
        self.num_params = num_params

        self.synth = FmSynth(config, device)
        self.decoder = Mel2Params2(num_mels, num_hops, num_params, dim=dim)
        # self.decoder = Mel2Params(num_mels, num_hops, num_params, dim=dim)

    def forward(self, params):
        mel_spec = self.synth(
            params[:, 0],  # carrier_frequency
            params[:, 1],  # modulator_frequency
            params[:, 2]  # modulation_index
        )
        params_recon = self.decoder(mel_spec)
        return params_recon


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

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


class Encoder(nn.Module):
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
        super().__init__()

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
