# %%
# imports

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sonification.datasets import Sinewave_dataset
from sonification.models.models import PlFactorVAE1D
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sonification.utils.matrix import view
from torch.utils.data import DataLoader
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors


# %%
# create args


class Args:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


args = Args(
    # root path
    root_path='',
    # dataset
    csv_path='sinewave.csv',
    img_size=64,
    # model
    in_channels=1,
    latent_size=2,
    kernel_size=[3, 3, 3, 3, 3],
    layers_channels=[64, 64, 64, 64, 64],
    d_hidden_size=64,
    d_num_layers=5,
    vae_dropout=0.0,
    d_dropout=0.0,
    # training
    recon_weight=10,
    target_recon_loss=0.008,
    dynamic_kld=1,
    dynamic_kld_increment=0.000004,
    cycling_kld=0,
    cycling_kld_period=10000,
    cycling_kld_ramp_up_phase=0.5,
    kld_weight_max=10,
    kld_weight_min=0.4,
    kld_start_epoch=0,
    kld_warmup_epochs=1,
    tc_weight=0.1,
    tc_start=0,
    tc_warmup_epochs=1,
    l1_weight=0.0,
    lr_d=0.01,
    lr_decay_d=0.999955,
    lr_decay_vae=0.999955,
    lr_vae=0.1,
    # checkpoint & logging
    ckpt_path='./ckpt/sinewave_fvae-mae-v3',
    ckpt_name='mae-v22.4',
    logdir='./logs/sinewave_fvae-mae-v3',
    plot_interval=1000,
)

# %%
# create train and val datasets and loaders
sinewave_ds_train = Sinewave_dataset(
    root_path=args.root_path, csv_path=args.csv_path, flag="train")
sinewave_ds_val = Sinewave_dataset(
    root_path=args.root_path, csv_path=args.csv_path, flag="val", scaler=sinewave_ds_train.scaler)

# %%
# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt_path = '../../ckpt/sinewave_fvae-mae-v3/mae-v22.4/mae-v22.4_last_epoch=400040.ckpt'
ckpt = torch.load(ckpt_path, map_location=device)
args.train_scaler = sinewave_ds_train.scaler
model = PlFactorVAE1D(args).to(device)
model.load_state_dict(ckpt['state_dict'])
model.eval()

# %%
batch_size = 256
dataset = sinewave_ds_val
loader = DataLoader(dataset, batch_size=batch_size,
                    shuffle=False, drop_last=False)
z_all = torch.zeros(
    len(dataset), model.args.latent_size).to(model.device)
for batch_idx, data in enumerate(loader):
    x, y = data
    x_recon, mean, logvar, z = model.VAE(x.to(model.device))
    z = z.detach()
    z_all[batch_idx*batch_size: batch_idx*batch_size + batch_size] = z
z_all = z_all.cpu().numpy()
# create the figure
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
ax.scatter(z_all[:, 0], z_all[:, 1])
ax.set_title(
    f"Latent space")
plt.show()

# %%
z_x_min, z_x_max = z_all[:, 0].min(), z_all[:, 0].max()
z_y_min, z_y_max = z_all[:, 1].min(), z_all[:, 1].max()

z_x_min, z_x_max, z_y_min, z_y_max

# %%
x_steps = torch.linspace(z_x_min, z_x_max, 64)
y_steps = torch.linspace(z_y_min, z_y_max, 64)

x_steps, y_steps

# %%
# create a plot for traversing in the latent space
# fig, ax = plt.subplots(len(x_steps), len(y_steps), figsize=(20, 20))
fig, ax = plt.subplots(1, len(y_steps), figsize=(200, 20))
all_decoded = []

for y_idx, y_step in enumerate(y_steps):
    latent_coords = torch.zeros(len(x_steps), 2)
    latent_coords[:, 1] = y_step
    latent_coords[:, 0] = x_steps
    decoded = model.decode(latent_coords.to(model.device))
    decoded = decoded.detach().squeeze(1).cpu().numpy()
    decoded_scaled = dataset.scaler.inverse_transform(decoded).T
    all_decoded.append(decoded_scaled)

# Create a normalization object for colormap
vmin = np.min(all_decoded)
vmax = np.max(all_decoded)
norm = colors.Normalize(vmin=vmin, vmax=vmax)

for y_idx, y_step in enumerate(y_steps):
    decoded_scaled = all_decoded[y_idx]
    ax[y_idx].matshow(
        decoded_scaled, interpolation='nearest', cmap='viridis', norm=norm)
    # remove axis labels
    ax[y_idx].axis('off')
# smaller margins between subplots
plt.subplots_adjust(wspace=0.1, hspace=0.1)
# reduce the overall margins
plt.tight_layout()
# remove white background
fig.patch.set_visible(False)
plt.savefig("traverse_latent_space_sinewave_big.png")
plt.show()

# %%
df = sinewave_ds_val.df
# get the sample with the lowest pitch and lowest loudness
df_lowest_pitch = df[df['pitch'] == df['pitch'].min()]
df_lowest_pitch_lowest_loudness = df_lowest_pitch[df_lowest_pitch['loudness']
                                                  == df_lowest_pitch['loudness'].min()]
df_lowest_pitch_lowest_loudness
# %%
# now highest pitch lowest loudness
df_highest_pitch = df[df['pitch'] == df['pitch'].max()]
df_highest_pitch_lowest_loudness = df_highest_pitch[df_highest_pitch['loudness']
                                                    == df_highest_pitch['loudness'].min()]
df_highest_pitch_lowest_loudness
# %%
# now lowest pitch highest loudness
df_lowest_pitch = df[df['pitch'] == df['pitch'].min()]
df_lowest_pitch_highest_loudness = df_lowest_pitch[df_lowest_pitch['loudness']
                                                   == df_lowest_pitch['loudness'].max()]
df_lowest_pitch_highest_loudness
# %%
# now highest pitch highest loudness
df_highest_pitch = df[df['pitch'] == df['pitch'].max()]
df_highest_pitch_highest_loudness = df_highest_pitch[df_highest_pitch['loudness']
                                                     == df_highest_pitch['loudness'].max()]
df_highest_pitch_highest_loudness

# %%
# get their indices
idx_top_left = df_highest_pitch_highest_loudness.index[0]
idx_bottom_left = df_lowest_pitch_highest_loudness.index[0]
idx_top_right = df_highest_pitch_lowest_loudness.index[0]
idx_bottom_right = df_lowest_pitch_lowest_loudness.index[0]
print(idx_top_left, idx_bottom_left, idx_top_right, idx_bottom_right)

# %%
# get their latent space coordinates
indices = [idx_top_left, idx_bottom_left, idx_top_right, idx_bottom_right]
latent_corners = []

for idx in indices:
    x, y = sinewave_ds_val[idx - 8000]  # offset by the length of training set
    x = x.unsqueeze(0)
    x_recon, mean, logvar, z = model.VAE(x.to(model.device))
    latent_corners.append(z.detach().cpu().numpy())

# %%
# interpolate between top left and bottom left
n_points = 64
top_left = latent_corners[0]
bottom_left = latent_corners[1]
top_right = latent_corners[2]
bottom_right = latent_corners[3]

# interpolate between top left and bottom left
leftmost_col = np.zeros((n_points, 2))
leftmost_col[:, 0] = np.linspace(top_left[0, 0], bottom_left[0, 0], n_points)
leftmost_col[:, 1] = np.linspace(top_left[0, 1], bottom_left[0, 1], n_points)

# interpolate between top right and bottom right
rightmost_col = np.zeros((n_points, 2))
rightmost_col[:, 0] = np.linspace(
    top_right[0, 0], bottom_right[0, 0], n_points)
rightmost_col[:, 1] = np.linspace(
    top_right[0, 1], bottom_right[0, 1], n_points)

# %%
# fill a matrix of n_points x n_points x 2 with the interpolated values
latent_matrix = np.zeros((n_points, n_points, 2))
for i in range(n_points):
    latent_matrix[:, i, 0] = np.linspace(
        leftmost_col[i, 0], rightmost_col[i, 0], n_points)
    latent_matrix[:, i, 1] = np.linspace(
        leftmost_col[i, 1], rightmost_col[i, 1], n_points)

# %%
# create a plot for traversing in the latent matrix
fig, ax = plt.subplots(1, n_points, figsize=(200, 3), dpi=300)
# fig, ax = plt.subplots(1, n_points, dpi=300)
all_decoded = []

for i in range(n_points):
    latent_coords = torch.tensor(latent_matrix[:, i], dtype=torch.float32)
    decoded = model.decode(latent_coords.to(model.device))
    decoded = decoded.detach().squeeze(1).cpu().numpy()
    decoded_scaled = dataset.scaler.inverse_transform(decoded).T
    all_decoded.append(decoded_scaled)

# Create a normalization object for colormap
vmin = np.min(all_decoded)
vmax = np.max(all_decoded)
norm = colors.Normalize(vmin=vmin, vmax=vmax)

for i in range(n_points):
    decoded_scaled = all_decoded[i]
    ax[i].matshow(
        decoded_scaled, interpolation='nearest', cmap='viridis', norm=norm)
    # remove axis labels
    ax[i].axis('off')
# smaller margins between subplots
plt.subplots_adjust(wspace=0.1, hspace=0.1)
# reduce the overall margins
plt.tight_layout()
# remove white background
fig.patch.set_visible(False)
plt.savefig("traverse_latent_space_sinewave_mae-v22.4.png")
plt.show()
# %%
