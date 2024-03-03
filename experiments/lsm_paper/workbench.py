# %%
# imports
from pythonosc import udp_client
from sonification.utils.matrix import square_over_bg
from pythonosc import osc_server
from pythonosc import dispatcher
from torchsummary import summary
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
from sonification.utils.tensor import permute_dims, scale
from sonification.models.loss import kld_loss, recon_loss, MMDloss
import torch.nn.functional as F
from sonification.models.layers import LinearDiscriminator
from sonification.datasets import Sinewave_dataset, White_Square_dataset
from sonification.models.models import ConvVAE1D, PlFactorVAE1D, PlFactorVAE
from sklearn.neighbors import KDTree

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


def square_over_bg(
    x: int,
    y: int,
    img_size: int = 512,
    square_size: int = 50,
) -> torch.Tensor:
    """
    Create a binary image of a square over a black background.

    Args:
        x (int): The x coordinate of the top-left corner of the square.
        y (int): The y coordinate of the top-left corner of the square.
        img_size (int, optional): The size of each side of the image. Defaults to 512.
        square_size (int, optional): The size of each side of the square. Defaults to 50.

    Returns:
        torch.Tensor: _description_
    """
    # create a black image
    img = torch.zeros((img_size, img_size))
    # set the square to white
    img[y:y + square_size, x:x + square_size] = 1

    return img
# %%
# square over bg with falloff


def square_over_bg_falloff(
    x: int,
    y: int,
    img_size: int = 64,
    square_size: int = 2,
    falloff_mult: int = 0.5
) -> torch.Tensor:
    """
    Create a binary image of a square over a black background with a falloff.

    Args:
        x (int): The x coordinate of the top-left corner of the square.
        y (int): The y coordinate of the top-left corner of the square.
        img_size (int, optional): The size of each side of the image. Defaults to 512.
        square_size (int, optional): The size of each side of the square. Defaults to 50.
        falloff_mult (int, optional): The falloff multiplier. Defaults to 0.5.

    Returns:
        torch.Tensor: _description_
    """
    # create a black image
    img = torch.zeros((img_size, img_size))
    # set the square to white
    img[y:y + square_size, x:x + square_size] = 1
    # create falloff
    falloff = torch.zeros((img_size, img_size))
    for i in range(img_size):
        for j in range(img_size):
            _x, _y = x + square_size / 2, y + square_size / 2
            v_to_square = torch.tensor(
                [i, j]) - torch.tensor([_y, _x], dtype=torch.float32)
            v_length = torch.norm(v_to_square)
            falloff[i, j] = 1 - torch.clip(scale(v_length, 0, img_size,
                                           0, img_size, exp=falloff_mult) / img_size, 0, 1)

    return torch.clip(img + falloff, 0, 1)

# %%


def square_over_bg_falloff(
    x: int,
    y: int,
    img_size: int = 64,
    square_size: int = 2,
    falloff_mult: int = 0.5
) -> torch.Tensor:
    """
    Create a binary image of a square over a black background with a falloff.

    Args:
        x (int): The x coordinate of the top-left corner of the square.
        y (int): The y coordinate of the top-left corner of the square.
        img_size (int, optional): The size of each side of the image. Defaults to 512.
        square_size (int, optional): The size of each side of the square. Defaults to 50.
        falloff_mult (int, optional): The falloff multiplier. Defaults to 0.5.

    Returns:
        torch.Tensor: _description_
    """
    # create a black image
    img = torch.zeros((img_size, img_size))
    # set the square to white
    img[y:y + square_size, x:x + square_size] = 1
    # create falloff
    falloff = torch.zeros((img_size, img_size))
    _x, _y = x + square_size / 2, y + square_size / 2
    i, j = torch.meshgrid(torch.arange(img_size), torch.arange(img_size))
    v_to_square = torch.stack(
        [i, j]) - torch.tensor([_y, _x], dtype=torch.float32).view(2, 1, 1)
    v_length = torch.norm(v_to_square, dim=0)
    falloff = 1 - torch.clip(scale(v_length, 0, img_size,
                             0, img_size, exp=falloff_mult) / img_size, 0, 1)

    return torch.clip(img + falloff, 0, 1)


# %%
# plot it
img = square_over_bg_falloff(30, 40, 64, 8)
img_np = img.numpy()
fig, ax = plt.subplots()
cax = ax.matshow(img_np, interpolation='nearest', cmap='viridis')
fig.colorbar(cax)
plt.show()
# %%
