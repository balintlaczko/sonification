# %%
# imports

import argparse
import json
import os
import torch
import random
import numpy as np
import pandas as pd
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from sonification.models.models import PlFMEmbedder
from sonification.utils.misc import midi2frequency
from sonification.utils.tensor import scale
from torch.utils.data import DataLoader
from tqdm import tqdm
import umap

# %%
ckpt_path = '../../ckpt/fm_embedder'
ckpt_name = 'imv_v4.3'
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
print(args)

# %%
# create model with args and load state dict
model = PlFMEmbedder(args)
model.create_shadow_model()
model.load_state_dict(ckpt['state_dict'])
model.eval()

# %%
# test the model with a dummy input
batch_size = 16
x = torch.rand(batch_size, args.length_samps).to(model.device)  # (B, length_samps)
y = model(x)
print(f"Input shape: {x.shape}, Output shape: {y.shape}")

# %%
import matplotlib.pyplot as plt

y_np = y.detach().cpu().numpy()

def plot_overlay(y, max_lines=1024, alpha=0.12, lw=0.7, color='C0', show=True):
    """
    Plot a 1D array as a single line, or a batch of arrays as overlaid lines.

    Args:
        y: np.ndarray or array-like. If 1D, plots a single line.
           If ND with shape (B, ...), flattens per-sample features and overlays B lines.
        max_lines: Maximum number of lines to overlay (randomly sampled if exceeded).
        alpha: Line alpha for overlaid lines.
        lw: Line width.
        color: Line color.
        show: Whether to call plt.show().
    Returns:
        ax: The matplotlib Axes used for plotting.
    """
    y_np = np.asarray(y)

    if y_np.ndim == 1:
        idx = np.arange(y_np.shape[0])
        fig, ax = plt.subplots()
        ax.plot(idx, y_np, color=color, linewidth=lw)
        ax.set_xlabel('Index')
        ax.set_ylabel('Value')
        ax.set_title('y (single line)')
        fig.tight_layout()
        if show:
            plt.show()
        return ax
    else:
        lines = y_np.reshape(y_np.shape[0], -1)
        n = lines.shape[0]
        if n > max_lines:
            rng = np.random.default_rng(0)
            sel = rng.choice(n, size=max_lines, replace=False)
            lines = lines[sel]

        idx = np.arange(lines.shape[1])
        fig, ax = plt.subplots()
        for i in range(lines.shape[0]):
            ax.plot(idx, lines[i], color=color, alpha=alpha, linewidth=lw)
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Value')
        ax.set_title(f'y (overlay {lines.shape[0]} lines)')
        fig.tight_layout()
        if show:
            plt.show()
        return ax

# call the function
plot_overlay(y_np)

# %%
# create a meshgrid of parameters to synthesize (SynthMaps style)

# create ranges for each parameter
pitch_steps = 51
harm_ratio_steps = 51
mod_idx_steps = 51
pitches = np.linspace(38, 86, pitch_steps)
freqs = midi2frequency(pitches)  # x
ratios = np.linspace(0, 1, harm_ratio_steps) * args.max_harm_ratio  # y
indices = np.linspace(0, 1, mod_idx_steps) * args.max_mod_idx  # z

# make into 3D mesh
freqs, ratios, indices = np.meshgrid(freqs, ratios, indices)  # y, x, z!
print(freqs.shape, ratios.shape, indices.shape)

# Create a dictionary where each key-value pair corresponds to a column in the dataframe
data = {
    "x": np.tile(np.repeat(np.arange(pitch_steps), mod_idx_steps), harm_ratio_steps),
    "y": np.repeat(np.arange(harm_ratio_steps), pitch_steps * mod_idx_steps),
    "z": np.tile(np.arange(mod_idx_steps), harm_ratio_steps * pitch_steps),
    "freq": freqs.flatten(),
    "harm_ratio": ratios.flatten(),
    "mod_index": indices.flatten()
}

# Create the dataframe
df = pd.DataFrame(data)
df.head()

# %%
# get the cuda/mps device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
# move model to device
model = model.to(device)
model.mel_spectrogram.to(device)  # move mel spectrogram to device

# %%
# iterate the dataframe in batches and synthesize
batch_size = 32
n_batches = len(df) // batch_size + 1
print(f"Number of batches: {n_batches}, batch size: {batch_size}")
z_all = torch.zeros((len(df), args.latent_size), device=device)
with torch.no_grad():
    for i in tqdm(range(n_batches)):
        batch_df = df.iloc[i * batch_size: (i + 1) * batch_size]
        # convert to tensor
        batch_z = torch.tensor(batch_df[['freq', 'harm_ratio', 'mod_index']].values, dtype=torch.float32)
        freqs = batch_z[:, 0]
        ratios = batch_z[:, 1]
        indices = batch_z[:, 2]
        # repeat in samples dimension
        freqs = freqs.unsqueeze(1).repeat(1, args.length_samps).to(device)
        ratios = ratios.unsqueeze(1).repeat(1, args.length_samps).to(device)
        indices = indices.unsqueeze(1).repeat(1, args.length_samps).to(device)
        # synthesize
        x = model.input_synth(freqs, ratios, indices)
        # embed
        y = model(x)
        # store in z_all
        z_all[i * batch_size: (i + 1) * batch_size] = y

# %%
z_all.shape  # (n_samples, latent_size)
# %%
plot_overlay(z_all.cpu().numpy(), max_lines=z_all.shape[0], alpha=0.12, lw=0.7, color='C1', show=True)

# %%
# UMAP 2D
Z = z_all.detach().cpu().numpy()
n = Z.shape[0]
max_points = 500000
if n > max_points:
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=max_points, replace=False)
else:
    idx = np.arange(n)

n_neighbors = 10
emb = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1, random_state=0).fit_transform(Z[idx])

# Build RGB colors from dataframe x, y, z columns
colors = df.iloc[idx][["x", "y", "z"]].to_numpy().astype(float)
mins = colors.min(axis=0)
maxs = colors.max(axis=0)
den = np.where((maxs - mins) == 0, 1.0, (maxs - mins))
colors = (colors - mins) / den  # normalize to [0, 1]

plt.figure(figsize=(7, 6))
plt.scatter(emb[:, 0], emb[:, 1], s=3, c=colors, alpha=0.7)
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.title(f"UMAP n_neighbors={n_neighbors}")
plt.tight_layout()
plt.show()

# %%
# UMAP 3D

Z = z_all.detach().cpu().numpy()
n = Z.shape[0]
max_points = 500000
if n > max_points:
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=max_points, replace=False)
else:
    idx = np.arange(n)

n_neighbors = 20
min_dist = 0.1  # minimum distance between points in UMAP
metric = 'cosine'  # distance metric for UMAP
emb = umap.UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=0).fit_transform(Z[idx])

# Build RGB colors from dataframe x, y, z columns
colors = df.iloc[idx][["x", "y", "z"]].to_numpy().astype(float)
mins = colors.min(axis=0)
maxs = colors.max(axis=0)
den = np.where((maxs - mins) == 0, 0.8, (maxs - mins))
colors = (colors - mins) / den  # normalize to [0, 1]

fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2], s=3, c=colors, alpha=0.7)
ax.set_xlabel("UMAP-1")
ax.set_ylabel("UMAP-2")
ax.set_zlabel("UMAP-3")
ax.set_title(f"UMAP 3D n_neighbors={n_neighbors}")
fig.tight_layout()
plt.show()

# %%
from sonification.utils.array import array2fluid_dataset

# create a directory for the predictions if it doesn't exist
ckpt_dir = os.path.dirname(ckpt_path)
predictions_dir = os.path.join(ckpt_dir, 'predictions')
os.makedirs(predictions_dir, exist_ok=True)

# %%
# save the fm parameters to a json file
fm_params = df.iloc[idx][["freq", "harm_ratio", "mod_index"]]
fm_params = fm_params.to_numpy().astype(float)
fm_params_json = array2fluid_dataset(fm_params)
fm_params_json_path = os.path.join(predictions_dir, "fm_params.json")
with open(fm_params_json_path, "w") as f:
    json.dump(fm_params_json, f)
print(f"FM parameters saved to {fm_params_json_path}")

# %%
# save the predictions to a json file
predictions_fluid_dataset = array2fluid_dataset(z_all.cpu().numpy())
json_path = os.path.join(predictions_dir, "model_embeddings.json")
with open(json_path, "w") as f:
    json.dump(predictions_fluid_dataset, f)
print(f"Model embeddings saved to {json_path}")

# %%
# save the umap embedding to a json file
# umap_embedding = array2fluid_dataset(emb)
# umap_json_path = os.path.join(predictions_dir, f"umap_embeddings_{n_neighbors}.json")
# with open(umap_json_path, "w") as f:
#     json.dump(umap_embedding, f)
# print(f"UMAP embeddings saved to {umap_json_path}")

# %%
# save the umap embedding to a json file
umap_embedding = array2fluid_dataset(emb)
umap_json_path = os.path.join(predictions_dir, f"umap_embeddings_3D_{n_neighbors}_min_dist_{min_dist}_metric_{metric}.json")
with open(umap_json_path, "w") as f:
    json.dump(umap_embedding, f)
print(f"UMAP embeddings saved to {umap_json_path}")

# %%
# save colors to a json file
colors_json = array2fluid_dataset(colors)
colors_json_path = os.path.join(predictions_dir, "colors.json")
with open(colors_json_path, "w") as f:
    json.dump(colors_json, f)
print(f"Colors saved to {colors_json_path}")
# %%
