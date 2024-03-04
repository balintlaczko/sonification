# %%
# imports
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
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
from sonification.models.models import ConvVAE, ConvVAE1D, PlFactorVAE1D, PlFactorVAE, PlMapper, PlVAE
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
# create MNIST dataset
# create a transform to pad the images to 32x32
pad = transforms.Pad(padding=2)
mnist_train = MNIST(root="./", train=True, transform=transforms.Compose(
    [transforms.ToTensor(), pad]), download=True)
mnist_val = MNIST(root="./", train=False, transform=transforms.Compose(
    [transforms.ToTensor(), pad]), download=True)

# %%
# create dataloaders
batch_size = 128
train_loader = DataLoader(
    mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(mnist_val, batch_size=batch_size,
                        shuffle=False, drop_last=False)

# %%
# get a batch of data
x, y = next(iter(train_loader))
x.shape, y.shape

# %%
# plot a batch of MNIST images
x_np = x.squeeze(1)[1].numpy()
print(x_np.shape)
fig, ax = plt.subplots()
cax = ax.matshow(x_np, interpolation='nearest', cmap='gray')
fig.colorbar(cax)
plt.show()

# %%
# create a VAE model
conv_vae = ConvVAE(
    in_channels=1, latent_size=2, layers_channels=[16, 32, 64, 128], input_size=32)
conv_vae.eval()

# %%
# test that shapes are correct
test_input = torch.rand(42, 1, 32, 32)
print(test_input.shape)
recon, mean, logvar, z = conv_vae(test_input)
recon.shape, mean.shape, logvar.shape, z.shape

# %%
# plot reconstructions
recon_np = recon.detach().squeeze(1).numpy()
print(recon_np.shape)
fig, ax = plt.subplots()
cax = ax.matshow(recon_np[0], interpolation='nearest', cmap='gray')
fig.colorbar(cax)
plt.show()


# %%
# create Args class


class Args:
    def __init__(self):
        self.in_channels = 1
        self.img_size = 32
        self.latent_size = 2
        self.layers_channels = [128, 256, 512, 1024]
        self.batch_size = 2048
        self.lr_vae = 1e-3
        self.lr_decay_vae = 0.9999
        self.recon_weight = 1.0
        self.target_recon_loss = 0.01
        self.dynamic_kld = 1
        self.kld_weight_max = 1.0
        self.kld_weight_min = 0.001
        self.kld_start_epoch = 0
        self.kld_warmup_epochs = 1
        self.plot_interval = 1
        self.ckpt_path = "./ckpt"
        self.ckpt_name = "mnist_test-v2"
        self.logdir = "./logs"
        self.train_epochs = 100000
        self.resume_ckpt_path = None


args = Args()

# %%
# create the model
model = PlVAE(args)

# %%
# checkpoint callbacks
checkpoint_path = os.path.join(
    args.ckpt_path, args.ckpt_name)
best_checkpoint_callback = ModelCheckpoint(
    monitor="val_vae_loss",
    dirpath=checkpoint_path,
    filename=args.ckpt_name + "_{epoch:02d}-{val_vae_loss:.4f}",
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
# create Trainer

trainer = Trainer(
    max_epochs=args.train_epochs,
    enable_checkpointing=True,
    callbacks=[best_checkpoint_callback, last_checkpoint_callback],
    logger=tensorboard_logger,
)

# save hyperparameters
hyperparams = dict(
    in_channels=args.in_channels,
    img_size=args.img_size,
    latent_size=args.latent_size,
    layers_channels=args.layers_channels,
    batch_size=args.batch_size,
    lr_vae=args.lr_vae,
    lr_decay_vae=args.lr_decay_vae,
    recon_weight=args.recon_weight,
    target_recon_loss=args.target_recon_loss,
    dynamic_kld=args.dynamic_kld,
    kld_weight_max=args.kld_weight_max,
    kld_weight_min=args.kld_weight_min,
    kld_start_epoch=args.kld_start_epoch,
    kld_warmup_epochs=args.kld_warmup_epochs,
    comment="test"
)

trainer.logger.log_hyperparams(hyperparams)

# %%

# train the model
torch.set_float32_matmul_precision('high')
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader,
            ckpt_path=args.resume_ckpt_path)

# %%
