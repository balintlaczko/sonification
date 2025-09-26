# %%
# imports

import os
import torch
import numpy as np
import pandas as pd
from sonification.models.models import PlSineFactorVAE
from sonification.utils.misc import midi2frequency, frequency2midi
from sonification.utils.tensor import db2amp, scale
from sonification.utils.dsp import db2amp, amp2db
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# %%
ckpt_path = '../../ckpt/sine_vae'
model_version = '29'
ckpt_name = 'imv_new_v' + model_version
ckpt_path = os.path.join(ckpt_path, ckpt_name)
# list files, find the one that has "last" in it
ckpt_files = [f for f in os.listdir(ckpt_path) if 'last' in f]
if len(ckpt_files) == 0:
    raise ValueError(f"No checkpoint file found in {ckpt_path} with 'last' in the name")
ckpt_file = ckpt_files[0]
ckpt_path = os.path.join(ckpt_path, ckpt_file)
print(f"Checkpoint file found: {ckpt_path}")

# %%
# load checkpoint and extract saved args
ckpt = torch.load(ckpt_path, map_location='cpu')
args = ckpt["hyper_parameters"]['args']
# print(args)

# %%
# create model with args and load state dict
model = PlSineFactorVAE(args)
model.load_state_dict(ckpt['state_dict'])
model.eval()
print("Model loaded")


# %%
# test decode random latent
z = torch.randn(1, args.latent_size)
print(f"Random latent z: {z}")
with torch.no_grad():
    decoded = model.model.decoder(z)
print(f"Predicted spectrum shape: {decoded.shape}")

# %%
# create a meshgrid of parameters to synthesize

# create ranges for each parameter
pitch_steps = 128
amp_steps = 128
pitches = np.linspace(38, 86, pitch_steps)  # MIDI note numbers
freqs = midi2frequency(pitches)
amps = np.linspace(0.01, 1.0, amp_steps)  # linear amplitude

# create a meshgrid
freqs, amps = np.meshgrid(freqs, amps)
print(f"Meshgrid shapes: freqs: {freqs.shape}, amps: {amps.shape}")

# create a dictionary where each key-value pair corresponds to a column in the dataframe
data = {
    "x": np.tile(np.arange(pitch_steps), amp_steps),
    "y": np.repeat(np.arange(amp_steps), pitch_steps),
    "pitch": frequency2midi(freqs.flatten()),
    "freq": freqs.flatten(),
    "amp": amps.flatten(),
    "db": amp2db(amps.flatten()),
}

# Create the dataframe
df = pd.DataFrame(data)
# df.head()

# %%
# get the cuda/mps device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
# print(f"Using device: {device}")
# move model to device
model = model.to(device)
model.mel_spectrogram.to(device)  # move mel spectrogram to device


# %%
# iterate the dataframe in batches and synthesize
batch_size = 128
n_batches = len(df) // batch_size #+ 1
print(f"Number of batches: {n_batches}, batch size: {batch_size}")
z_all = torch.zeros((len(df), args.latent_size)).to(device)
with torch.no_grad():
    for i in tqdm(range(n_batches)):
        batch_df = df.iloc[i * batch_size: (i + 1) * batch_size]
        # convert to tensor
        batch_z = torch.tensor(batch_df[['freq', 'amp']].values, dtype=torch.float32)
        freqs = batch_z[:, 0]
        amps = batch_z[:, 1]
        # repeat in samples dimension
        freqs = freqs.unsqueeze(1).repeat(1, args.length_samps).to(device)
        # synthesize
        x = model.input_synth(freqs) * amps.unsqueeze(1).to(device)
        # add channel dimension
        in_wf = x.unsqueeze(1)
        # get the mel spectrogram
        in_spec = model.mel_spectrogram(in_wf)
        # convert to db
        in_spec = model.amplitude_to_db(in_spec)
        # average time dimension, keep batch and mel dims
        in_spec = torch.mean(in_spec, dim=-1)
        # scale by bin minmax
        in_spec = scale(in_spec, args.bin_minmax[0], args.bin_minmax[1], 0, 1)
        # encode
        mu, logvar = model.model.encode(in_spec)
        z = model.model.reparameterize(mu, logvar)
        # store
        z_all[i * batch_size: i * batch_size + batch_size, :] = z
z_all = z_all.cpu().numpy()


# %%
# create a scatter plot of the latent space
import itertools

num_dims = z_all.shape[1]
dim_pairs = list(itertools.combinations(range(num_dims), 2))
num_pairs = len(dim_pairs)

if num_pairs > 0:
    fig, axes = plt.subplots(num_pairs, 2, figsize=(20, 10 * num_pairs), squeeze=False)

    epoch_idx = ckpt_file.split('_')[-1].split('.')[0].split("=")[-1]
    fig.suptitle(f"Model v{model_version} - Epoch {epoch_idx}", fontsize=16)

    for i, (dim1, dim2) in enumerate(dim_pairs):
        ax1 = axes[i, 0]
        ax2 = axes[i, 1]

        # Plot 1: Colored by pitch
        sc1 = ax1.scatter(z_all[:, dim1], z_all[:, dim2], c=df['pitch'], cmap='viridis', s=3)
        fig.colorbar(sc1, ax=ax1, label='Pitch (MIDI)', shrink=0.8)
        ax1.set_title(f"Latent Dims {dim1} vs {dim2} by Pitch")
        ax1.set_xlabel(f"Latent Dim {dim1}")
        ax1.set_ylabel(f"Latent Dim {dim2}")
        ax1.set_aspect('equal', adjustable='box')

        # Plot 2: Colored by amplitude
        sc2 = ax2.scatter(z_all[:, dim1], z_all[:, dim2], c=df['amp'], cmap='viridis', s=3)
        fig.colorbar(sc2, ax=ax2, label='Amplitude', shrink=0.8)
        ax2.set_title(f"Latent Dims {dim1} vs {dim2} by Amplitude")
        ax2.set_xlabel(f"Latent Dim {dim1}")
        ax2.set_ylabel(f"Latent Dim {dim2}")
        ax2.set_aspect('equal', adjustable='box')
        # # Plot 2: Colored by decibels
        # sc2 = ax2.scatter(z_all[:, dim1], z_all[:, dim2], c=df['db'], cmap='viridis', s=3)
        # fig.colorbar(sc2, ax=ax2, label='Decibels', shrink=0.8)
        # ax2.set_title(f"Latent Dims {dim1} vs {dim2} by Decibels")
        # ax2.set_xlabel(f"Latent Dim {dim1}")
        # ax2.set_ylabel(f"Latent Dim {dim2}")
        # ax2.set_aspect('equal', adjustable='box')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

########################################################################

# %%
# set percentiles
percentile_low = 5
percentile_high = 95
steps = 64

z_x_min = np.percentile(z_all[:, 0], percentile_low)
z_x_max = np.percentile(z_all[:, 0], percentile_high)
z_y_min = np.percentile(z_all[:, 1], percentile_low)
z_y_max = np.percentile(z_all[:, 1], percentile_high)

x_steps = torch.linspace(z_x_min, z_x_max, steps)
y_steps = torch.linspace(z_y_min, z_y_max, steps)

z_x_min, z_x_max, z_y_min, z_y_max

# %%
# create a plot for traversing in the latent space through Y
num_cols = int(np.sqrt(len(y_steps)))
fig, ax = plt.subplots(num_cols, num_cols, figsize=(20, 20))
all_decoded = []

with torch.no_grad():
    for y_idx, y_step in enumerate(y_steps):
        latent_coords = torch.zeros(len(x_steps), args.latent_size)
        latent_coords[:, 1] = y_step
        latent_coords[:, 0] = x_steps
        decoded = model.model.decoder(latent_coords.to(device))
        # decoded = db2amp(decoded)
        decoded = decoded.squeeze(1).cpu().numpy()
        all_decoded.append(decoded)

# Create a normalization object for colormap
vmin = np.min(all_decoded)
vmax = np.max(all_decoded)
norm = colors.Normalize(vmin=vmin, vmax=vmax)

for y_idx, y_step in enumerate(y_steps):
    decoded_scaled = all_decoded[y_idx]
    ax[y_idx // num_cols, y_idx % num_cols].matshow(
        decoded_scaled.T, interpolation='nearest', cmap='viridis', norm=norm)
    # remove axis labels
    ax[y_idx // num_cols, y_idx % num_cols].axis('off')
# smaller margins between subplots
plt.subplots_adjust(wspace=0.1, hspace=0.1)
# reduce the overall margins
plt.tight_layout()
# remove white background
fig.patch.set_visible(False)
# plt.savefig("traverse_latent_space_sinewave_testY.png")
plt.show()

# %%
# create a plot for traversing in the latent space through X
fig, ax = plt.subplots(num_cols, num_cols, figsize=(20, 20))
all_decoded = []

with torch.no_grad():
    for x_idx, x_step in enumerate(x_steps):
        latent_coords = torch.zeros(len(y_steps), args.latent_size)
        latent_coords[:, 1] = y_steps
        latent_coords[:, 0] = x_step
        decoded = model.model.decoder(latent_coords.to(device))
        # decoded = scale(decoded, 0, 1, args.bin_minmax[0], args.bin_minmax[1])
        decoded = decoded.squeeze(1).cpu().numpy()
        all_decoded.append(decoded)

# Create a normalization object for colormap
vmin = np.min(all_decoded)
vmax = np.max(all_decoded)
norm = colors.Normalize(vmin=vmin, vmax=vmax)

for y_idx, y_step in enumerate(y_steps):
    decoded_scaled = all_decoded[y_idx]
    ax[y_idx // num_cols, y_idx % num_cols].matshow(
        decoded_scaled.T, interpolation='nearest', cmap='viridis', norm=norm)
    # remove axis labels
    ax[y_idx // num_cols, y_idx % num_cols].axis('off')
# smaller margins between subplots
plt.subplots_adjust(wspace=0.1, hspace=0.1)
# reduce the overall margins
plt.tight_layout()
# remove white background
fig.patch.set_visible(False)
# plt.savefig("traverse_latent_space_sinewave_testX.png")
plt.show()

# %%
