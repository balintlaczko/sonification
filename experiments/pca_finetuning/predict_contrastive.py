# %%
# imports

import json
import os
import torch
import numpy as np
import pandas as pd
from sonification.models.models import PlFMEmbedder
from sonification.utils.misc import midi2frequency
from tqdm import tqdm
import umap
import matplotlib.pyplot as plt
from sonification.utils.array import array2fluid_dataset

# %%
ckpt_path = '../../ckpt/fm_embedder'
ckpt_name = 'imv_v4.5'
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
batch_size = 64
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
# standardize the embeddings
z_mean = z_all.mean(dim=0, keepdim=True)
z_std = z_all.std(dim=0, keepdim=True)
z_all_standardized = (z_all - z_mean) / (z_std + 1e-6)  # avoid division by zero
print(z_mean.shape, z_std.shape, z_all_standardized.shape)

# %%
# robustscale embeddings
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
z_all_robustscaled = scaler.fit_transform(z_all_standardized.detach().cpu().numpy())
print(z_all_robustscaled.shape)

# %%
# create a PCA projection for z_all
from sklearn.decomposition import PCA

pca_dims = 16
pca = PCA(n_components=pca_dims, whiten=True)
z_all_pca = pca.fit_transform(z_all_robustscaled)
# get the explained variance ratio
explained_variance = pca.explained_variance_ratio_
print(f"Explained variance ratio for PCA with {pca_dims} components: {explained_variance.sum() * 100:.2f}%")

# %%
# UMAP
mode = 'pca'  # 'standardized', 'robustscaled', 'pca' or 'raw'
if mode == 'standardized':
    Z = z_all_standardized.detach().cpu().numpy()
elif mode == 'robustscaled':
    Z = z_all_robustscaled
elif mode == 'pca':
    Z = z_all_pca
else:
    Z = z_all.detach().cpu().numpy()
n = Z.shape[0]
max_points = 500000
if n > max_points:
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=max_points, replace=False)
else:
    idx = np.arange(n)

n_components = 3  # 3 for 3D UMAP
n_neighbors = 200
min_dist = 0.1  # minimum distance between points in UMAP
metric = 'euclidean'  # distance metric for UMAP
emb = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric).fit_transform(Z[idx])

# Build RGB colors from dataframe x, y, z columns
colors = df.iloc[idx][["x", "y", "z"]].to_numpy().astype(float)
mins = colors.min(axis=0)
maxs = colors.max(axis=0)
den = np.where((maxs - mins) == 0, 1, (maxs - mins))
colors = (colors - mins) / den  # normalize to [0, 1]
# scale them to [0, 0.7]
colors *= 0.9

plot_title = f"UMAP {mode}, {n_components}D, n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}"
fig = plt.figure(figsize=(7, 6))
if n_components == 2:
    ax = fig.add_subplot(111)
    ax.scatter(emb[:, 0], emb[:, 1], s=3, c=colors, alpha=0.7)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
else:
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2], s=3, c=colors, alpha=0.7)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_zlabel("UMAP-3")
ax.set_title(plot_title)
fig.tight_layout()
plt.show()

# %%
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
# if UMAP is 2D then add the x col from the df as the 3rd column
if n_components == 2:
    # scale x to the observed UMAP range
    x_min = df['x'].min()
    x_max = df['x'].max()
    x = df.iloc[idx]['x'].to_numpy().astype(float)
    x_norm = (x - x_min) / (x_max - x_min)  # scale to [0, 1]
    umap_min = emb.min()
    umap_max = emb.max()
    x_scaled = (x_norm * (umap_max - umap_min)) + umap_min  # scale to UMAP range
    # add x as the 3rd column
    emb = np.hstack((emb, x_scaled.reshape(-1, 1)))

# %%
# save the umap embedding to a json file
umap_embedding = array2fluid_dataset(emb)
umap_json_path = os.path.join(predictions_dir, f"umap_embeddings_{mode}_{n_components}D_{n_neighbors}_min_dist_{min_dist}_metric_{metric}.json")
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
