# %%
# imports
import librosa
from torchaudio.functional import amplitude_to_DB, DB_to_amplitude
import torch
import numpy as np
import cv2
from sonification.utils import matrix
from sonification.utils.dsp import midi2frequency, db2amp
from torchaudio.transforms import MelSpectrogram, Loudness, InverseMelScale, GriffinLim
from sonification.models.ddsp import Sinewave
from scipy.io import wavfile as wav
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import torchyin
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from torch import nn
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import Trainer
from sonification.utils.tensor import permute_dims
from sonification.models.loss import kld_loss, recon_loss, MMDloss
import torch.nn.functional as F
from sonification.models.layers import LinearDiscriminator
# %%
# create a torch-based matrix generator to render an image of a white square on a black background

img_size = 512
square_size = 50

# create a black image
img = torch.zeros((img_size, img_size), dtype=torch.uint8)

# create a white square
# random x and y coordinates
x = torch.randint(0, img_size - square_size, (1,))
y = torch.randint(0, img_size - square_size, (1,))

# set the square to white
img[y:y + square_size, x:x + square_size] = 255

# %%
# display the image
img_np = img.numpy()
matrix.view(img_np)

# %%
# function form


def white_square_on_black_bg(x, y, img_size=512, square_size=50):
    # create a black image
    img = torch.zeros((img_size, img_size))
    # set the square to white
    img[y:y + square_size, x:x + square_size] = 1

    return img


# %%
# test white_square_on_black_bg
img_size = 512
square_size = 50
for i in range(5):
    x = torch.randint(0, img_size - square_size, (1,))
    y = torch.randint(0, img_size - square_size, (1,))
    img = white_square_on_black_bg(x, y)
    img_np = img.numpy()
    matrix.view(img_np*255)

# %%
# create permute_dims function


def permute_dims(x):
    # x is (batch, z)
    # permutate dimensions independently
    b, z_dim = x.shape
    y = torch.zeros_like(x)
    for i in range(z_dim):
        y[:, i] = x[:, i][torch.randperm(b)]
    return y


# %%
# test permute_dims
z_dim = 10
b = 5
arange = torch.arange(b).repeat(z_dim, 1).T
print(arange)
arange_perm = permute_dims(arange)
print(arange)
print(arange_perm)

# %%
# from factorVAE repo


def permute_dims_2(z):
    assert z.dim() == 2

    B, _ = z.size()
    perm_z = []
    for z_j in z.split(1, 1):
        perm = torch.randperm(B).to(z.device)
        perm_z_j = z_j[perm]
        perm_z.append(perm_z_j)

    return torch.cat(perm_z, 1)


# %%
# test permute_dims
z_dim = 10
b = 5
arange = torch.arange(b).repeat(z_dim, 1).T
print(arange)
arange_perm = permute_dims_2(arange)
print(arange)
print(arange_perm)

# %%
root_path = "./experiments/lsm_paper"
csv_path = "sinewave.csv"
sr = 44100
samps = 16384*2
n_fft = 8192
f_min = 60
f_max = 1200
pad = 1
n_mels = 64
power = 1
norm = "slaney"
mel_scale = "slaney"


# %%
sinewave_gen = Sinewave(sr=sr)
# %%
in_tensor = torch.ones(1, samps) * 830.61
in_tensor.shape
# %%
sinewave = sinewave_gen(in_tensor)
sinewave.shape
# %%
# save to disk
path = "sinewave.wav"
wav.write(path, sr, sinewave[0].numpy())
# %%
# create mel spectrogram
mel_spectrogram = MelSpectrogram(
    sample_rate=sr,
    n_fft=n_fft,
    f_min=f_min,
    f_max=f_max,
    pad=pad,
    n_mels=n_mels,
    power=2,
    norm=norm,
    mel_scale=mel_scale
    # mel_scale="htk"
)
# %%
melspec = mel_spectrogram(sinewave)
# average time dimension
melspec_avg = melspec.mean(dim=2)
melspec_avg

# %%

melspec_avg_db = amplitude_to_DB(
    melspec_avg, multiplier=10, amin=1e-5, db_multiplier=20, top_db=80)
melspec_avg_db

# %%
melspec_avg_np = melspec_avg[0].numpy()
librosa_melspec_avg = librosa.power_to_db(
    melspec_avg_np, ref=1.0, amin=1e-5, top_db=80)
librosa_melspec_avg
# %%
# normalize tensor
melspec_avg_db_norm = (melspec_avg_db - melspec_avg_db.min()) / \
    (melspec_avg_db.max() - melspec_avg_db.min())
melspec_avg_db_norm
# %%
# plot the mel spectrogram
# melspec_avg_np = melspec_avg_db[0].unsqueeze(-1).numpy()
melspec_avg_np = melspec_avg[0].unsqueeze(-1).numpy()
# reverse the mel scale
# melspec_avg_np = np.flip(melspec_avg_np, axis=0)
# repeat the mel spectrogram to make it more visible
# melspec_avg_np = np.repeat(melspec_avg_np, 10, axis=1)
# transpose
melspec_avg_np = melspec_avg_np.T
fig, ax = plt.subplots()
cax = ax.matshow(melspec_avg_np, interpolation='nearest', cmap='viridis')
fig.colorbar(cax)
plt.show()
# %%
loud = Loudness(sample_rate=sr)
# %%
loud(sinewave*0.5)
# %%
for multiplier in [1, 0.5, 0.25]:
    loudness = loud(sinewave*multiplier)
    print(loudness)

# %%
for multiplier in [1, 0.5, 0.25]:
    mspec = mel_spectrogram(sinewave*multiplier)
    mspec_avg = mspec.mean(dim=2)
    mspec_avg_db = amplitude_to_DB(
        mspec_avg, multiplier=10, amin=1e-5, db_multiplier=20, top_db=80)
    print(mspec_avg_db.mean())
    print(mspec_avg_db.min(), mspec_avg_db.max())
# %%

# resynth chain: repeat -> inverse mel -> griffin lim -> waveform

melspec_avg_rep = melspec_avg.unsqueeze(-1).repeat(1, 1, 9)
print(melspec_avg_rep.shape)

inverse_mel = InverseMelScale(
    n_stft=n_fft//2+1,
    n_mels=n_mels,
    sample_rate=sr,
    f_min=f_min,
    f_max=f_max,
    norm=norm,
    mel_scale=mel_scale,
)

spec = inverse_mel(melspec_avg_rep)
# spec = inverse_mel(melspec)
print(spec.shape)

# %%
gl = GriffinLim(
    n_fft=n_fft,
    n_iter=32,
    power=2,
    momentum=0.99)

waveform = gl(spec)
print(waveform.shape)

# %%
# get pitch with torchyin
pitch = torchyin.estimate(waveform, sr)
pitch.mean()

# resynthesize sinewave based on pitch
in_tensor = torch.ones(1, samps) * pitch.mean()
sinewave_recon = sinewave_gen(in_tensor)
sinewave_recon.shape

# %%
# save reconstructed waveform to disk
path = "reconstructed.wav"
wav.write(path, sr, sinewave_recon[0].numpy())
# %%


class Sinewave_dataset(Dataset):
    """Dataset of sine waves with varying pitch and loudness"""

    def __init__(
            self,
            root_path="",
            csv_path="sinewave.csv",
            sr=44100,
            samps=16384,
            n_fft=8192,
            f_min=60,
            f_max=1200,
            pad=1,
            n_mels=64,
            power=2,
            norm="slaney",
            mel_scale="slaney",
            flag="train",
            scaler=None) -> None:
        super().__init__()

        # parse inputs
        self.root_path = root_path
        self.csv_path = csv_path
        self.sr = sr
        self.samps = samps
        self.n_fft = n_fft
        self.f_min = f_min
        self.f_max = f_max
        self.pad = pad
        self.n_mels = n_mels
        self.power = power
        self.norm = norm
        self.mel_scale = mel_scale

        # parse flag
        assert flag in ['train', 'val', 'all']
        self.flag = flag

        # generators
        self.sinewave_gen = Sinewave(sr=self.sr)
        self.mel_spec = MelSpectrogram(
            sample_rate=self.sr,
            n_fft=self.n_fft,
            f_min=self.f_min,
            f_max=self.f_max,
            pad=self.pad,
            n_mels=self.n_mels,
            power=self.power,
            norm=self.norm,
            mel_scale=self.mel_scale)

        # scaler
        if scaler is not None:
            self.scaler = scaler
        else:
            self.scaler = MinMaxScaler()
        self.scaler_fitted = False

        # read data
        self.__read_data__()

    def __read_data__(self):
        # read csv
        self.df = pd.read_csv(os.path.join(self.root_path, self.csv_path))
        # filter for the set we want (train/val)
        if self.flag != 'all':
            self.df = self.df[self.df.dataset == self.flag]
        # fit scaler
        self.fit_scaler()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.scaler_fitted:
            x_1 = self.all_tensors[idx]
            if len(x_1.shape) == 2:
                x_1 = x_1.unsqueeze(0)
            x_1 = x_1.permute(0, 2, 1)  # (B, C=1, n_mels)
            num_elems = x_1.shape[0]
            # generate random indices for the second tensor
            indices = torch.randint(0, len(self.df), (num_elems,))
            x_2 = self.all_tensors[indices].permute(
                0, 2, 1)  # (B, C=1, n_mels)
            if num_elems == 1:
                return x_1[0], x_2[0]
            return x_1, x_2  # (B, C=1, n_mels) (B, C=1, n_mels)
        # get the row
        row = self.df.iloc[idx]
        # get the pitch and loudness
        pitch = row.pitch
        loudness = row.loudness
        # convert to frequency and amplitude
        freq = midi2frequency(np.array([pitch]))
        amp = db2amp(np.array([loudness]))
        num_elems = freq.shape[-1]
        # synthetize the sine wave
        if num_elems == 1:
            sinewave = self.sinewave_gen(torch.ones(
                1, self.samps) * freq) * amp
        else:
            sinewave = self.sinewave_gen(torch.ones(
                num_elems, self.samps) * freq.T) * amp.T
        # convert to float32
        sinewave = sinewave.float()
        # transform to mel spectrogram
        mel_spec = self.mel_spec(sinewave)
        # average time dimension
        mel_spec_avg = mel_spec.mean(dim=2, keepdim=True)
        mel_spec_avg_db = amplitude_to_DB(
            mel_spec_avg, multiplier=10, amin=1e-5, db_multiplier=20, top_db=80)
        # transform with scaler
        if self.scaler_fitted:
            mel_spec_avg_db = self.scaler.transform(
                mel_spec_avg_db.squeeze(-1).numpy())
            mel_spec_avg_db = torch.tensor(mel_spec_avg_db).unsqueeze(-1)
        # return the mel spectrogram
        return mel_spec_avg_db.permute(0, 2, 1)  # (B, C=1, n_mels)

    def fit_scaler(self):
        if self.flag != 'train' and self.scaler != None:
            print("Transforming dataset with external scaler")
            all_tensors = self.__getitem__(
                np.arange(len(self.df)))  # (B, C=1, n_mels)
            all_tensors = all_tensors.squeeze(1).numpy()  # (B, n_mels)
            self.all_tensors = self.scaler.transform(all_tensors)
            self.all_tensors = torch.tensor(
                self.all_tensors).unsqueeze(-1)  # (B, n_mels, C=1)
            self.scaler_fitted = True
            print("Scaler transformed")
        elif self.flag == 'train':
            all_tensors = self.__getitem__(
                np.arange(len(self.df)))  # (B, C=1, n_mels)
            all_tensors = all_tensors.squeeze(1).numpy()  # (B, n_mels)
            self.all_tensors = self.scaler.fit_transform(all_tensors)
            self.all_tensors = torch.tensor(
                self.all_tensors).unsqueeze(-1)  # (B, n_mels, C=1)
            self.scaler_fitted = True
            print("Scaler fit+transformed")


# %%
sinewave_ds_train = Sinewave_dataset(flag="train")
sinewave_ds_val = Sinewave_dataset(flag="val", scaler=sinewave_ds_train.scaler)

# %%
x_1, x_2 = sinewave_ds_val[:12]
print(x_1.shape, x_2.shape)
# %%
# plot the two mel spectrograms
specs, _ = sinewave_ds_train[:100]
print(specs.shape)
specs_np = specs.squeeze(1).T.numpy()

fig, ax = plt.subplots()
cax = ax.matshow(specs_np, interpolation='nearest', cmap='viridis')
fig.colorbar(cax)
plt.show()
# %%


class ConvEncoder1D(nn.Module):
    def __init__(self, in_channels, output_size, layers_channels=[16, 32, 64, 128, 256, 512], input_size=512):
        super(ConvEncoder1D, self).__init__()
        self.in_channels = in_channels  # 2 for red and green
        self.output_size = output_size

        layers = []
        in_channel = self.in_channels
        for out_channel in layers_channels:
            layers.extend([
                nn.Conv1d(in_channel, out_channel, 3, stride=2, padding=1),
                nn.BatchNorm1d(out_channel),
                nn.LeakyReLU(0.2)
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


class ConvDecoder1D(nn.Module):
    def __init__(self, latent_size, out_channels, layers_channels=[512, 256, 128, 64, 32, 16], output_size=512):
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
        for out_channel in layers_channels[1:]:
            layers.extend([
                nn.ConvTranspose1d(in_channel, out_channel,
                                   3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm1d(out_channel),
                nn.LeakyReLU(0.2),
            ])
            in_channel = out_channel

        layers.extend([
            nn.ConvTranspose1d(
                layers_channels[-1], out_channels, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(out_channels),
            nn.Sigmoid(),
        ])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# %%


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


# %%
# test that shapes are correct
conv_vae = ConvVAE1D(
    in_channels=1, latent_size=2, layers_channels=[16, 32, 64, 128, 256], input_size=64)
# %%
test_input = torch.rand(128, 1, 64)
recon, mean, logvar, z = conv_vae(test_input)
recon.shape, mean.shape, logvar.shape, z.shape

# %%
conv_vae.eval()
specs, _ = sinewave_ds_train[:10]
print(specs.shape)
recon, mean, logvar, z = conv_vae(specs)
recon.shape, mean.shape, logvar.shape, z.shape
# %%
# plot reconstructions
recon_np = recon.detach().squeeze(1).numpy()
print(recon_np.shape)
# inverse transform with scaler
recon_np = sinewave_ds_train.scaler.inverse_transform(recon_np).T
fig, ax = plt.subplots()
cax = ax.matshow(recon_np, interpolation='nearest', cmap='viridis')
fig.colorbar(cax)
plt.show()
# %%


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
        self.kld_weight = args.kld_weight
        self.tc_weight = args.tc_weight
        self.l1_weight = args.l1_weight

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

        # VAE KLD loss
        kld_loss = self.kld(mean, logvar) * self.kld_weight

        # VAE TC loss
        d_z = self.D(z)
        vae_tc_loss = F.cross_entropy(d_z, ones) * self.tc_weight

        # L1 penalty
        l1_penalty = torch.tensor(0, dtype=torch.float32, device=self.device)
        if self.l1_weight > 0:
            l1_penalty = sum([p.abs().sum()
                              for p in self.VAE.parameters()]) * self.l1_weight

        # VAE loss
        vae_loss = vae_recon_loss + kld_loss + vae_tc_loss + l1_penalty

        # VAE backward pass
        vae_optimizer.zero_grad()
        self.manual_backward(vae_loss, retain_graph=True)

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
        self.log_dict({
            "vae_loss": vae_loss,
            "vae_recon_loss": vae_recon_loss,
            "vae_kld_loss": kld_loss,
            "vae_tc_loss": vae_tc_loss,
            "d_tc_loss": d_tc_loss,
            "vae_l1_penalty": l1_penalty
        })

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
        kld_loss = self.kld(mean, logvar) * self.kld_weight

        # VAE TC loss
        d_z = self.D(z)
        vae_tc_loss = F.cross_entropy(d_z, ones) * self.tc_weight

        # L1 penalty
        l1_penalty = torch.tensor(0, dtype=torch.float32, device=self.device)
        if self.l1_weight > 0:
            l1_penalty = sum([p.abs().sum()
                              for p in self.VAE.parameters()]) * self.l1_weight

        # VAE loss
        vae_loss = vae_recon_loss + kld_loss + vae_tc_loss + l1_penalty

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
            "val_vae_l1_penalty": l1_penalty
        })

    def on_validation_epoch_end(self) -> None:
        self.VAE.eval()
        # get the epoch number from trainer
        epoch = self.trainer.current_epoch

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

# %%
# set up test training cycle


class Args:
    def __init__(self):
        self.in_channels = 1
        self.img_size = 64
        self.latent_size = 2
        self.layers_channels = [1024, 1024, 1024, 1024, 1024]
        self.d_hidden_size = 512
        self.d_num_layers = 5
        self.train_epochs = 10
        self.batch_size = 512
        self.lr_vae = 1e-5
        self.lr_decay_vae = 0.99
        self.lr_d = 1e-5
        self.lr_decay_d = 0.99
        self.kld_weight = 0.01
        self.tc_weight = 0.01
        self.l1_weight = 0
        self.ckpt_path = "./ckpt/sinewave_fvae_test"
        self.ckpt_name = "test-v1"
        self.plot_interval = 1
        self.logdir = "./logs/sinewave_fvae_test"
        self.comment = "test"


args = Args()

# %%
# create train and val datasets and loaders
sinewave_ds_train = Sinewave_dataset(flag="train")
sinewave_ds_val = Sinewave_dataset(flag="val", scaler=sinewave_ds_train.scaler)

train_loader = DataLoader(
    sinewave_ds_train, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)
val_loader = DataLoader(
    sinewave_ds_val, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True)

# %%
# create the model
args.train_scaler = sinewave_ds_train.scaler
model = PlFactorVAE1D(args)

# %%
# checkpoint callbacks
checkpoint_path = os.path.join(args.ckpt_path, args.ckpt_name)
best_checkpoint_callback = ModelCheckpoint(
    monitor="val_vae_loss",
    dirpath=checkpoint_path,
    filename=args.ckpt_name + "_val_{epoch:02d}-{val_loss:.4f}",
    save_top_k=1,
    mode="min",
)
last_checkpoint_callback = ModelCheckpoint(
    monitor="epoch",
    dirpath=checkpoint_path,
    filename=args.ckpt_name + "_last_{epoch:02d}",
    save_top_k=1,
    mode="max",
)

# logger callbacks
tensorboard_logger = TensorBoardLogger(
    save_dir=args.logdir, name=args.ckpt_name)

# %%
trainer = Trainer(
    max_epochs=args.train_epochs,
    enable_checkpointing=True,
    callbacks=[best_checkpoint_callback, last_checkpoint_callback],
    logger=tensorboard_logger,
    log_every_n_steps=1,
)

# %%
# save hyperparameters
hyperparams = dict(
    in_channels=args.in_channels,
    img_size=args.img_size,
    latent_size=args.latent_size,
    layers_channels=args.layers_channels,
    d_hidden_size=args.d_hidden_size,
    d_num_layers=args.d_num_layers,
    batch_size=args.batch_size,
    lr_vae=args.lr_vae,
    lr_decay_vae=args.lr_decay_vae,
    lr_d=args.lr_d,
    lr_decay_d=args.lr_decay_d,
    kld_weight=args.kld_weight,
    tc_weight=args.tc_weight,
    l1_weight=args.l1_weight,
    comment=args.comment
)

trainer.logger.log_hyperparams(hyperparams)
# %%
# train the model
trainer.fit(model, train_loader, val_loader)

# %%
