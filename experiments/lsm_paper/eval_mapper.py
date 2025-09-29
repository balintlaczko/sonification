# %%
# imports

import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sonification.models.models import PlImgFactorVAE, PlSineFactorVAE, PlMapper
from sonification.models.loss import pearson_correlation_loss
from sonification.datasets import White_Square_dataset_2 as White_Square_dataset
from sonification.utils.tensor import scale
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import itertools
import time

# %%
# find the latest mapper checkpoint
ckpt_path = '../../ckpt/mapper'
model_version = '4'
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
print(args.in_model)

# %%
# load in model checkpoint and extract saved args
ckpt = torch.load("./../." + args.img_model_ckpt_path, map_location='cpu')
in_model_args = ckpt["hyper_parameters"]['args']

# create in model with args and load state dict
in_model = PlImgFactorVAE(in_model_args)
in_model.load_state_dict(ckpt['state_dict'])
in_model.eval()
print("In model loaded")

# load out model checkpoint and extract saved args
ckpt = torch.load("./../." + args.audio_model_ckpt_path, map_location='cpu')
out_model_args = ckpt["hyper_parameters"]['args']

# create out model with args and load state dict
out_model = PlSineFactorVAE(out_model_args)
out_model.load_state_dict(ckpt['state_dict'])
out_model.eval()
print("Out model loaded")

# # create mapper model where args also references the in and out models
# args.in_model = in_model
# args.out_model = out_model
model = PlMapper(args)
# model.in_model.requires_grad_(False)
# model.out_model.requires_grad_(False)
model.eval()
print("Mapper model created")

# %%
# create image dataset
dataset = White_Square_dataset(img_size=in_model_args.img_size, square_size=in_model_args.square_size)

# create image dataloader
batch_size=128
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
print("Image dataloader created")

# %%
# get the cuda/mps device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
# print(f"Using device: {device}")
# move model to device
model = model.to(device)
model.eval()
in_model = in_model.to(device)
in_model.eval()
model.in_model.eval()
out_model = out_model.to(device)
out_model.eval()
model.out_model.eval()

# %%
# iterate through the dataloader and encode, map, decode, re-encode
z_1_all = torch.zeros(len(dataset), model.in_model.args.latent_size).to(device) # img encoded
z_1b_all = torch.zeros(len(dataset), model.in_model.args.latent_size).to(device) # img encoded
z_2_all = torch.zeros(len(dataset), model.out_model.args.latent_size).to(device) # mapped
z_3_all = torch.zeros(len(dataset), model.out_model.args.latent_size).to(device) # audio re-encoded
losses = []
with torch.no_grad():
    for batch_idx, data in tqdm(enumerate(dataloader)):
        x, _ = data
        # print(f"Batch {batch_idx}, x shape: {x.shape}")
        mu, logvar = model.in_model.model.encode(x.to(device))
        z_1 = model.in_model.model.reparameterize(mu, logvar)

        mu, logvar = in_model.model.encode(x.to(device))
        z_1b = in_model.model.reparameterize(mu, logvar)

        # store
        z_1_all[batch_idx * batch_size: batch_idx * batch_size + batch_size, :] = z_1
        z_1b_all[batch_idx * batch_size: batch_idx * batch_size + batch_size, :] = z_1b
        # map to audio latent space
        z_2 = model(z_1)
        z_2_all[batch_idx * batch_size: batch_idx * batch_size + batch_size, :] = z_2
        # decode to mel spectrogram
        x_hat = model.out_model.model.decoder(z_2)
        # encode back to latent space
        mu, logvar = model.out_model.model.encode(x_hat)
        z_3 = model.out_model.model.reparameterize(mu, logvar)
        z_3_all[batch_idx * batch_size: batch_idx * batch_size + batch_size, :] = z_3
        # calc batchwise distances and l1 loss
        dist_z_1 = torch.cdist(z_1, z_1, p=2)**2 # img encoded
        dist_z_2 = torch.cdist(z_2, z_2, p=2)**2 # mapped
        # # get the normalized distances
        # dist_z_1_norm = dist_z_1 / (dist_z_1.mean() + 1e-8)
        # dist_z_2_norm = dist_z_2 / (dist_z_2.mean() + 1e-8)
        # # compute the locality loss with l1
        # locality_l1_loss = F.l1_loss(dist_z_1_norm, dist_z_2_norm)
        locality_loss = pearson_correlation_loss(dist_z_1, dist_z_2)
        losses.append(locality_loss.item())

# %%
# compare z_1 and z_1b on a scatter plot
n_points = 5
offset = 100
plt.figure(figsize=(6,6))
plt.scatter(z_1_all[offset:offset+n_points,0].cpu(), z_1_all[offset:offset+n_points,1].cpu(), label='in model', alpha=0.5)
plt.scatter(z_1b_all[offset:offset+n_points,0].cpu(), z_1b_all[offset:offset+n_points,1].cpu(), label='out model', alpha=0.5)
plt.xlabel('z1')
plt.ylabel('z2')
plt.title('Comparison of z1 from in model and out model')
plt.legend()
plt.axis('equal')
plt.grid()
plt.show()

# %%
# encode the same image n times and see the variance in the latent space
n_points = 256
offset = 50
x, _ = dataset[offset]
x = x.unsqueeze(0).to(device)
z_1_samples = torch.zeros(n_points, model.in_model.args.latent_size).to(device)
z_1b_samples = torch.zeros(n_points, model.in_model.args.latent_size).to(device)
with torch.no_grad():
    for i in range(n_points):
        mu, logvar = in_model.model.encode(x)
        z_1 = in_model.model.reparameterize(mu, logvar)
        z_1_samples[i, :] = z_1
        z_1b_samples[i, :] = mu
# plot
plt.figure(figsize=(6,6))
plt.scatter(z_1_samples[:,0].cpu(), z_1_samples[:,1].cpu(), label='z', alpha=0.5)
plt.scatter(z_1b_samples[:,0].cpu(), z_1b_samples[:,1].cpu(), label='mu', alpha=0.5)
plt.xlabel('z1')
plt.ylabel('z2')
plt.title('Variance of z1 from in model for the same input image')
plt.legend()
plt.axis('equal')
plt.grid()
plt.show()

# %%
# compute pairwise distances
dist_z_1 = torch.cdist(z_1_all, z_1_all, p=2)**2 # img encoded
dist_z_2 = torch.cdist(z_2_all, z_2_all, p=2)**2 # mapped

# get the normalized distances
dist_z_1_norm = dist_z_1 / (dist_z_1.mean() + 1e-8)
dist_z_2_norm = dist_z_2 / (dist_z_2.mean() + 1e-8)

# # compute the locality loss
# locality_l1_loss = torch.sum(torch.abs(dist_z_1_norm - dist_z_2_norm), dim=-1).unsqueeze(-1)
# locality_l1_loss.shape

# %%
# compute the locality loss with l1
locality_l1_loss = F.l1_loss(dist_z_1_norm, dist_z_2_norm, reduction='none')
print(locality_l1_loss.shape)
# sum over the last dimension and unsqueeze
locality_l1_loss = torch.sum(locality_l1_loss, dim=-1, keepdim=True)
locality_l1_loss.shape

# %%
locality_l1_loss = F.l1_loss(dist_z_1_norm, dist_z_2_norm)
locality_l1_loss


# %%
a = np.arange(10)
a_mean = a.mean()
a_norm = a / (a_mean + 1e-8)
a, a_mean, a_norm

# %%