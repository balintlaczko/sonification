import torch
from torch import nn
from ..utils.tensor import wrap
from ..utils.dsp import num_hops
from ..utils.misc import scale_linear, midi2frequency
from layers import MLP, MultiScaleEncoder
import torchaudio
import torchyin


class Phasor(nn.Module):
    def __init__(
        self,
        sr: int
    ):
        super().__init__()

        self.sr = sr

    def forward(self, freq: torch.Tensor):  # input shape: (batch_size, n_samples)
        batch_size = freq.shape[0]
        increment = freq[:, :-1] / self.sr
        phase = torch.cumsum(increment, dim=-1)
        phase = torch.cat([torch.zeros(batch_size, 1).to(phase.device), phase], dim=-1)
        phasor = wrap(phase, 0.0, 1.0)
        return phasor


class Sinewave(nn.Module):
    def __init__(
        self,
        sr: int
    ):
        super().__init__()

        self.sr = sr
        self.phasor = Phasor(self.sr)

    def forward(self, freq: torch.Tensor):
        phasor = self.phasor(freq)
        sine = torch.sin(2 * torch.pi * phasor)
        return sine


class FMSynth(nn.Module):
    def __init__(
        self,
        sr: int,
    ):
        super().__init__()

        self.sr = sr
        self.modulator_sine = Sinewave(self.sr)
        self.carrier_sine = Sinewave(self.sr)

    def forward(
            self,
            carrier_frequency: torch.Tensor,
            harmonicity_ratio: torch.Tensor,
            modulation_index: torch.Tensor,
    ):
        modulator_frequency = carrier_frequency * harmonicity_ratio
        modulator_buf = self.modulator_sine(modulator_frequency)
        modulation_amplitude = modulator_frequency * modulation_index
        return self.carrier_sine(carrier_frequency + modulator_buf * modulation_amplitude)


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

        self.mel_encoder = MultiScaleEncoder(input_dim_h=num_mels, input_dim_w=num_hops)

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
            use_gru=True,
            gru_hidden_dim=512,
            mlp_in_dim=16,
            mlp_out_dim=256,
            mlp_layers=3
    ):
        super().__init__()

        self.use_gru = use_gru

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
        if self.use_gru:
            self.gru = nn.GRU(n_mfcc, gru_hidden_dim, batch_first=True)
            self.linear = nn.Linear(gru_hidden_dim, mlp_in_dim)
        else:
            self.linear = nn.Linear(n_mfcc, mlp_in_dim)
        self.mlp = MLP(mlp_in_dim, mlp_out_dim, mlp_out_dim, mlp_layers)

    def forward(self, y):
        # input shape: (batch_size, n_samples)
        mfcc = self.mfcc(y)
        # (batch_size, n_mfcc, n_frames)
        mfcc = self.layernorm(torch.transpose(mfcc, -2, -1))
        # (batch_size, n_frames, n_mfcc)
        if self.use_gru:
            mfcc, _ = self.gru(mfcc)
            # (batch_size, n_frames, gru_hidden_dim)
        mfcc = self.linear(mfcc)
        # (batch_size, n_frames, mlp_in_dim)
        mfcc = self.mlp(mfcc)
        # (batch_size, n_frames, mlp_out_dim)
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
            use_gru=True,
            gru_hidden_dim=512,
            mlp_in_dim=16,
            mlp_out_dim=256,
            mlp_layers=3
    ):
        super().__init__()

        self.use_gru = use_gru

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
        if self.use_gru:
            self.gru = nn.GRU(n_mels, gru_hidden_dim, batch_first=True)
            self.linear = nn.Linear(gru_hidden_dim, mlp_in_dim)
        else:
            self.linear = nn.Linear(n_mels, mlp_in_dim)
        self.mlp = MLP(mlp_in_dim, mlp_out_dim, mlp_out_dim, mlp_layers)

    def forward(self, y):
        # input shape: (batch_size, n_samples)
        melspec = self.melspec(y)
        # (batch_size, n_mels, n_frames)
        melspec = self.layernorm(torch.transpose(melspec, -2, -1))
        # (batch_size, n_frames, n_mels)
        if self.use_gru:
            melspec, _ = self.gru(melspec)
            # (batch_size, n_frames, gru_hidden_dim)
        melspec = self.linear(melspec)
        # (batch_size, n_frames, mlp_in_dim)
        melspec = self.mlp(melspec)
        # (batch_size, n_frames, mlp_out_dim)
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
            use_gru=True,
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
            use_gru=use_gru,
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
            use_gru=use_gru,
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
            # nn.Linear(3 * mlp_out_dim * n_hops, 3),
            nn.Linear(3 * mlp_out_dim, 3),
            # nn.ReLU(),
            nn.Sigmoid(),
        )
        # synth
        self.synth = FMSynth(sr=sr)

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
        # average time steps
        encoded = torch.mean(encoded, dim=-2, keepdim=True)
        # flatten encoded
        encoded_flatten = torch.flatten(encoded, start_dim=-2)
        # predict synth params (0-1)
        synth_params = self.synth_params(encoded_flatten)
        # fix nan
        # synth_params = torch.nan_to_num(synth_params)
        # scale synth params from 0-1 to their respective ranges
        carr_freq = synth_params[:, 0]
        carr_freq_midi = scale_linear(carr_freq, 0, 1, 44, 88)
        carr_freq_hz = midi2frequency(carr_freq_midi)
        harm_ratio = synth_params[:, 1]
        harm_ratio_scaled = scale_linear(harm_ratio, 0, 1, 1, 10)
        mod_index = synth_params[:, 2]
        mod_index_scaled = scale_linear(mod_index, 0, 1, 0.1, 10)
        # take each param from all batches and repeat it for the number of samples in the buffer
        carr_freq_array = carr_freq_hz.unsqueeze(-1).repeat(
            1, int(self.buffer_length_s * self.sr))
        harm_ratio_array = harm_ratio_scaled.unsqueeze(
            -1).repeat(1, int(self.buffer_length_s * self.sr))
        mod_index_array = mod_index_scaled.unsqueeze(
            -1).repeat(1, int(self.buffer_length_s * self.sr))
        # generate synth buffer
        synth_buffer = self.synth(
            carr_freq_array, harm_ratio_array, mod_index_array)

        return synth_buffer, synth_params
