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
# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt_path = '../../ckpt/sinewave_fvae-mae-v3/mae-v33.7/mae-v33.7_last_epoch=26350.ckpt'
ckpt = torch.load(ckpt_path, map_location=device)
args = ckpt["hyper_parameters"]["args"]

# %%
# create train and val datasets and loaders
sinewave_ds_train = Sinewave_dataset(csv_path=args.csv_path, flag="train")
sinewave_ds_val = Sinewave_dataset(csv_path=args.csv_path, flag="val", scaler=sinewave_ds_train.scaler)
args.train_scaler = sinewave_ds_train.scaler

# %%
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
plt.savefig("traverse_latent_space_sinewave_test.png")
plt.show()


# %%
# set percentiles
percentile_low = 25
percentile_high = 75

# get the percentiles
df = sinewave_ds_val.df
# get all unique pitch values sorter from low to high
pitch_values = np.sort(df['pitch'].unique())
print(pitch_values)
# get the percentiles for the pitch values
n_pitches = len(pitch_values)
pitch_low = pitch_values[int(n_pitches * percentile_low / 100)]
pitch_high = pitch_values[int(n_pitches * percentile_high / 100)]
# get all unique loudness values sorter from low to high
loudness_values = np.sort(df['loudness'].unique())
print(loudness_values)
# get the percentiles for the loudness values
n_loudnesses = len(loudness_values)
loudness_low = loudness_values[int(n_loudnesses * percentile_low / 100)]
loudness_high = loudness_values[int(n_loudnesses * percentile_high / 100)]
pitch_low, pitch_high, loudness_low, loudness_high

# %%
df = sinewave_ds_val.df
# get the sample with the lowest pitch and lowest loudness
# df_lowest_pitch = df[df['pitch'] == df['pitch'].min()]
# df_lowest_pitch_lowest_loudness = df_lowest_pitch[df_lowest_pitch['loudness'] == df_lowest_pitch['loudness'].min()]
df_lowest_pitch = df[df['pitch'] == pitch_low]
# get the closest loudness to the lowest loudness
df_lowest_pitch_lowest_loudness = df_lowest_pitch.iloc[(df_lowest_pitch['loudness'] - loudness_low).abs().argsort()[:1]]
df_lowest_pitch_lowest_loudness

# %%
# visualize the point in latent space
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
ax.scatter(z_all[:, 0], z_all[:, 1])
idx = df_lowest_pitch_lowest_loudness.index[0]
# get the row number of the sample
idx = np.where(df.index == idx)[0][0]
ax.scatter(z_all[idx, 0], z_all[idx, 1], c='r', s=100)
ax.set_title(f"Lowest pitch, lowest loudness")
plt.show()

# %%
# now highest pitch lowest loudness
# df_highest_pitch = df[df['pitch'] == df['pitch'].max()]
# df_highest_pitch_lowest_loudness = df_highest_pitch[df_highest_pitch['loudness']
#                                                     == df_highest_pitch['loudness'].min()]
df_highest_pitch = df[df['pitch'] == pitch_high]
# get the closest loudness to the lowest loudness
df_highest_pitch_lowest_loudness = df_highest_pitch.iloc[(df_highest_pitch['loudness'] - loudness_low).abs().argsort()[:1]]
df_highest_pitch_lowest_loudness

# %%
# visualize the point in latent space
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
ax.scatter(z_all[:, 0], z_all[:, 1])
idx = df_highest_pitch_lowest_loudness.index[0]
# get the row number of the sample
idx = np.where(df.index == idx)[0][0]
ax.scatter(z_all[idx, 0], z_all[idx, 1], c='r', s=100)
ax.set_title(f"Highest pitch, lowest loudness")
plt.show()

# %%
# now lowest pitch highest loudness
# df_lowest_pitch = df[df['pitch'] == df['pitch'].min()]
# df_lowest_pitch_highest_loudness = df_lowest_pitch[df_lowest_pitch['loudness']
#                                                    == df_lowest_pitch['loudness'].max()]
df_lowest_pitch = df[df['pitch'] == pitch_low]
# get the closest loudness to the highest loudness
df_lowest_pitch_highest_loudness = df_lowest_pitch.iloc[(df_lowest_pitch['loudness'] - loudness_high).abs().argsort()[:1]]
df_lowest_pitch_highest_loudness

# %%
# visualize the point in latent space
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
ax.scatter(z_all[:, 0], z_all[:, 1])
idx = df_lowest_pitch_highest_loudness.index[0]
# get the row number of the sample
idx = np.where(df.index == idx)[0][0]
ax.scatter(z_all[idx, 0], z_all[idx, 1], c='r', s=100)
ax.set_title(f"Lowest pitch, highest loudness")
plt.show()

# %%
# now highest pitch highest loudness
# df_highest_pitch = df[df['pitch'] == df['pitch'].max()]
# df_highest_pitch_highest_loudness = df_highest_pitch[df_highest_pitch['loudness']
#                                                      == df_highest_pitch['loudness'].max()]
df_highest_pitch = df[df['pitch'] == pitch_high]
# get the closest loudness to the highest loudness
df_highest_pitch_highest_loudness = df_highest_pitch.iloc[(df_highest_pitch['loudness'] - loudness_high).abs().argsort()[:1]]
df_highest_pitch_highest_loudness

# %%
# visualize the point in latent space
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
ax.scatter(z_all[:, 0], z_all[:, 1])
idx = df_highest_pitch_highest_loudness.index[0]
# get the row number of the sample
idx = np.where(df.index == idx)[0][0]
ax.scatter(z_all[idx, 0], z_all[idx, 1], c='r', s=100)
ax.set_title(f"Highest pitch, highest loudness")
plt.show()

# %%
# visualize the corners in the latent space
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
ax.scatter(z_all[:, 0], z_all[:, 1])
idx_top_left = df_highest_pitch_highest_loudness.index[0]
idx_bottom_left = df_lowest_pitch_highest_loudness.index[0]
idx_top_right = df_highest_pitch_lowest_loudness.index[0]
idx_bottom_right = df_lowest_pitch_lowest_loudness.index[0]
# get the row number of the sample
idx_top_left = np.where(df.index == idx_top_left)[0][0]
idx_bottom_left = np.where(df.index == idx_bottom_left)[0][0]
idx_top_right = np.where(df.index == idx_top_right)[0][0]
idx_bottom_right = np.where(df.index == idx_bottom_right)[0][0]
ax.scatter(z_all[idx_top_left, 0], z_all[idx_top_left, 1], c='r', s=100)
ax.scatter(z_all[idx_bottom_left, 0], z_all[idx_bottom_left, 1], c='r', s=100)
ax.scatter(z_all[idx_top_right, 0], z_all[idx_top_right, 1], c='r', s=100)
ax.scatter(z_all[idx_bottom_right, 0], z_all[idx_bottom_right, 1], c='r', s=100)
ax.set_title(f"Corners in latent space")
plt.show()

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
# visualize encoded corners
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
ax.scatter(z_all[:, 0], z_all[:, 1])
for corner in latent_corners:
    ax.scatter(corner[0, 0], corner[0, 1], c='r', s=100)
ax.set_title(f"Encoded corners")
plt.show()

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
plt.savefig("traverse_latent_space_sinewave_mae-v33.7.png")
plt.show()
# %%
