# %%
# imports

import os
import torch
import numpy as np
import pandas as pd
from sonification.models.models import PlImgFactorVAE
from sonification.datasets import White_Square_dataset_2 as White_Square_dataset
from sonification.utils.tensor import scale
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import itertools
import time

# %%
ckpt_path = '../../ckpt/squares_vae'
model_version = '21'
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
model = PlImgFactorVAE(args)
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
# create dataset
dataset = White_Square_dataset(img_size=args.img_size, square_size=args.square_size)

# create dataloader
batch_size=32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

# %%
# get the cuda/mps device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
# print(f"Using device: {device}")
# move model to device
model = model.to(device)

# %%
# iterate through the dataloader and encode the images to the latent space
z_all = torch.zeros(len(dataset), model.args.latent_size).to(device)
with torch.no_grad():
    for batch_idx, data in tqdm(enumerate(dataloader)):
        x, y = data
        x_recon, mean, logvar, z = model(x.to(device))
        # if batch_idx < 5:
        #     plt.imshow(x_recon[0, 0, ...].cpu().numpy(), cmap="gray")
        #     time.sleep(0.5)
        #     plt.close()
        # store
        z_all[batch_idx * batch_size: batch_idx * batch_size + batch_size, :] = z
z_all = z_all.cpu().numpy()

# %%
# create a scatter plot of the latent space
num_dims = z_all.shape[1]
dim_pairs = list(itertools.combinations(range(num_dims), 2))
num_pairs = len(dim_pairs)
df = dataset.df

if num_pairs > 0:
    fig, axes = plt.subplots(num_pairs, 2, figsize=(20, 10 * num_pairs), squeeze=False)

    epoch_idx = ckpt_file.split('_')[-1].split('.')[0].split("=")[-1]
    fig.suptitle(f"Model v{model_version} - Epoch {epoch_idx}", fontsize=16)

    for i, (dim1, dim2) in enumerate(dim_pairs):
        ax1 = axes[i, 0]
        ax2 = axes[i, 1]

        # Plot 1: Colored by x position
        sc1 = ax1.scatter(z_all[:, dim1], z_all[:, dim2], c=df['x'], cmap='viridis', s=3)
        fig.colorbar(sc1, ax=ax1, label='X Position', shrink=0.8)
        ax1.set_title(f"Latent Dims {dim1} vs {dim2} by X Position")
        ax1.set_xlabel(f"Latent Dim {dim1}")
        ax1.set_ylabel(f"Latent Dim {dim2}")
        ax1.set_aspect('equal', adjustable='box')

        # Plot 2: Colored by y position
        sc2 = ax2.scatter(z_all[:, dim1], z_all[:, dim2], c=df['y'], cmap='viridis', s=3)
        fig.colorbar(sc2, ax=ax2, label='Y Position', shrink=0.8)
        ax2.set_title(f"Latent Dims {dim1} vs {dim2} by Y Position")
        ax2.set_xlabel(f"Latent Dim {dim1}")
        ax2.set_ylabel(f"Latent Dim {dim2}")
        ax2.set_aspect('equal', adjustable='box')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# %%
# decode a grid of latent points and plot the decoded images
num_images = 8
random_indices = np.random.choice(len(dataset), num_images, replace=False)
img_size = args.img_size
x_samples = torch.zeros(num_images, 1, img_size, img_size)
for i, idx in enumerate(random_indices):
    x, y = dataset[idx]
    x_samples[i, ...] = x
x_samples = x_samples.to(device)
with torch.no_grad():
    x_recon, mean, logvar, z = model(x_samples)
# plot the original and reconstructed images side by side
fig, axes = plt.subplots(2, num_images, figsize=(20, 5))
for i in range(num_images):
    axes[0, i].imshow(x_samples[i, 0, ...].cpu().numpy(), cmap="gray")
    axes[0, i].set_title(f"Original {i}")
    axes[1, i].imshow(x_recon[i, 0, ...].cpu().numpy(), cmap="gray")
    axes[1, i].set_title(f"Reconstructed {i}")
plt.show()


########################################################################

# %%
# set percentiles
percentile_low = 1
percentile_high = 99
steps = 64

z_x_min = np.percentile(z_all[:, 0], percentile_low)
z_x_max = np.percentile(z_all[:, 0], percentile_high)
z_y_min = np.percentile(z_all[:, 1], percentile_low)
z_y_max = np.percentile(z_all[:, 1], percentile_high)

x_steps = torch.linspace(z_x_min, z_x_max, steps)
y_steps = torch.linspace(z_y_min, z_y_max, steps)

z_x_min, z_x_max, z_y_min, z_y_max

# %%
# get a sample from the dataset
idx = 0
x, y = dataset[idx]
print(x.shape, y.shape)
# plot image
plt.imshow(x[0, ...], cmap="gray")

# %%
x_flatten = x.view(1, -1)
x_flatten.shape
x_unflatten = x_flatten.view(1, 1, img_size, img_size)
x_unflatten.shape
plt.imshow(x_unflatten[0, 0, ...], cmap="gray")

# %%
# plot 8 samples from the dataset
fig, ax = plt.subplots(2, 4, figsize=(20, 10))
for i in range(8):
    x, y = dataset[i]
    ax[i//4, i % 4].imshow(x[0, ...], cmap="gray")
    ax[i//4, i % 4].set_title(f"{i}")
plt.show()


# %%
# plot 8 reconstructions
fig, ax = plt.subplots(2, 4, figsize=(20, 10))
for i in range(8):
    x, y = dataset[i]
    x = x.unsqueeze(0)
    x_recon, _, _, _ = model(x.to(model.device))
    ax[i//4, i %
        4].imshow(x_recon[0, 0, ...].detach().cpu().numpy(), cmap="gray")
    ax[i//4, i % 4].set_title(f"{i}")
plt.show()

# %%
# plot the latent point with the input image for 4 samples
fig, ax = plt.subplots(2, 4, figsize=(20, 10))
for i in range(4):
    x, y = dataset[i]
    x = x.unsqueeze(0)
    x_recon, mean, logvar, z = model(x.cuda())
    z = z.detach().cpu().numpy()
    ax[0, i].imshow(x[0, 0, ...], cmap="gray")
    ax[0, i].set_title(f"Input {i}")
    ax[1, i].scatter(z[0, 0], z[0, 1])
    # fix scaling of the plot
    ax[1, i].set_xlim(-3, 3)
    ax[1, i].set_ylim(-3, 3)
    ax[1, i].set_title(f"Latent {i}")

# %%
batch_size = 128
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
x_steps = torch.linspace(z_x_min, z_x_max, 20)
y_steps = torch.linspace(z_y_min, z_y_max, 20)

x_steps, y_steps

# %%
test_latent = torch.tensor([x_steps[0], y_steps[0]])
test_latent

# %%
x_recon = model.decode(test_latent.unsqueeze(0).to(model.device))
x_recon.shape

# %%
# create a plot for traversing in the latent space
fig, ax = plt.subplots(len(x_steps), len(y_steps), figsize=(20, 20))
for y_idx, y_step in enumerate(y_steps):
    for x_idx, x_step in enumerate(x_steps):
        latent_sample = torch.tensor([x_step, y_step]).unsqueeze(0)
        decoded = model.decode(latent_sample.to(model.device))
        ax[x_idx, y_idx].imshow(
            decoded[0, 0, ...].detach().cpu().numpy(), cmap="gray")
        # remove axis labels
        ax[x_idx, y_idx].axis('off')
        # remove margins
        # ax[x_idx, y_idx].xaxis.set_major_locator(plt.NullLocator())
        # ax[x_idx, y_idx].yaxis.set_major_locator(plt.NullLocator())
# smaller margins between subplots
plt.subplots_adjust(wspace=0.1, hspace=0.1)
# reduce the overall margins
plt.tight_layout()
plt.savefig("traverse_latent_space_64x2.png")
plt.show()


# %%
loader = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False)
# get dataset from loader

# %%
for batch_idx, data in enumerate(loader):
    x, y = data
    print(batch_idx, x.shape)

# %%
mnist_path = './MNIST'
# create a transofrm to apply to each datapoint
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = MNIST(mnist_path, transform=transform, download=True)
test_dataset = MNIST(mnist_path, transform=transform, download=True)

# create train and test dataloaders
batch_size = 100
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size, shuffle=False)

# %%
# get a sample from the dataset
idx = 0
x, y = train_dataset[idx]
print(x.shape, y)
x

# %%
# forward pass
x_recon, mean, logvar, z = model(x.cuda())
print(x_recon.shape, mean.shape, logvar.shape, z.shape)

# %%
# plot the original and reconstructed images
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(x[0, 0, ...].cpu().detach().numpy(), cmap="gray")
ax[0].set_title("Original")
ax[1].imshow(x_recon[0, 0, ...].cpu().detach().numpy(), cmap="gray")
ax[1].set_title("Reconstructed")
plt.show()

# %%
# get mse loss
mse_loss = torch.nn.MSELoss()
loss = mse_loss(x_recon, x.cuda())
loss
# %%

torch.mean(x_recon)

# %%
bce_loss = torch.nn.BCELoss()
x_recon = torch.sigmoid(x_recon)
loss = bce_loss(x_recon, x.cuda())
loss

# %%
samples_uniform = torch.rand(144, 2)
# compute the distance matrix
dist = torch.cdist(samples_uniform, samples_uniform)
# take the median of the upper triangle
dist = torch.triu(dist, diagonal=1)
dist = dist[dist != 0]
dist.median()
# plot the samples
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.scatter(samples_uniform[:, 0], samples_uniform[:, 1])
ax.set_title(
    f"Uniform samples")
plt.show()
# %%


def gap_loss(latent_vectors):
    # normalize the latent vectors between 0 and 1
    latent_vectors = (latent_vectors - latent_vectors.min()) / \
        (latent_vectors.max() - latent_vectors.min())
    # Calculate pairwise distances
    pairwise_distances = torch.cdist(latent_vectors, latent_vectors, p=2)

    # Set diagonal to infinity to ignore self-distances
    pairwise_distances.fill_diagonal_(float('inf'))
    print(pairwise_distances.shape)
    # Minimize the smallest distance
    return pairwise_distances.min()


# Example usage
# A batch of 32 latent vectors of size 100
latent_vectors = torch.rand(144, 2)
loss = gap_loss(latent_vectors)
loss
# %%


def gap_loss(latent_vectors):
    # normalize the latent vectors between 0 and 1
    latent_vectors_norm = (latent_vectors - latent_vectors.min()) / \
        (latent_vectors.max() - latent_vectors.min())
    num_dims = latent_vectors.shape[1]
    # loop over the dimensions
    loss = 0
    for i in range(num_dims):
        # sort the latent vectors
        sorted_latent, _ = torch.sort(latent_vectors_norm[:, i])
        # calculate the difference between the sorted vectors
        diff = sorted_latent[1:] - sorted_latent[:-1]
        # take the standard deviation of the differences
        loss += diff.std().abs()
    return loss


# %%
latent_vectors = torch.rand(144, 2)
loss = gap_loss(latent_vectors)
loss
# %%


def gap_loss(latent_vectors):
    # normalize the latent vectors between 0 and 1
    latent_vectors_norm = (latent_vectors - latent_vectors.min()) / \
        (latent_vectors.max() - latent_vectors.min())
    # Calculate pairwise distances
    dist = torch.cdist(latent_vectors_norm, latent_vectors_norm, p=2)
    dist = torch.triu(dist, diagonal=1)
    dist = dist[dist != 0]
    return dist.std()


# %%
dataset_size = 10
latent_size = 2
lattice = torch.linspace(0, 1, dataset_size)

# %%


def sample_lattice(batch_size):
    # sample random coordinates from the lattice for each latent dimension for a batch
    lattice_samples = []
    for i in range(latent_size):
        # shuffle the lattice
        z_lattice = lattice[torch.randperm(dataset_size)]
        # sample from the shuffled lattice
        lattice_samples.append(z_lattice[:batch_size])
    return torch.stack(lattice_samples, dim=1)


# %%
dataset_size = 144
latent_size = 2
side_size = int(dataset_size**(1/latent_size))
lattice = torch.linspace(0, 1, side_size)


def sample_lattice(batch_size):
    # sample random coordinates from the lattice for each latent dimension for a batch
    lattice_samples = []
    for i in range(latent_size):
        # get n random values (repetitions allowed) from the lattice
        z_lattice = lattice[torch.randint(0, side_size, (batch_size,))]
        # z_lattice = lattice[torch.randperm(side_size)]
        # sample from the shuffled lattice
        lattice_samples.append(z_lattice[:batch_size])
    return torch.stack(lattice_samples, dim=1)


# %%
sample_lattice(10)
# %%
# plot sampled points from the lattice
samples = sample_lattice(144)
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.scatter(samples[:, 0], samples[:, 1])
ax.set_title(
    f"Samples from the lattice")
plt.show()
# %%
inpoints = torch.rand(100, 2)
rachet = 1/12
outpoints = rachet * torch.round(inpoints / rachet)
# plot inpoints and outpoints
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.scatter(inpoints[:, 0], inpoints[:, 1], label="inpoints")
ax.scatter(outpoints[:, 0], outpoints[:, 1], label="outpoints")
ax.set_title(
    f"Samples from the lattice")
plt.show()

# %%
inpoints = sample_lattice(144)
# inpoints = torch.cat((inpoints, inpoints))
dist = torch.cdist(inpoints, inpoints)
# print(dist)
dist = dist.fill_diagonal_(float('inf'))
# dist = torch.triu(dist, diagonal=1)
dist = dist[dist == 0]
len(dist)//2

# %%


def count_repetitions(samples):
    # count the number of repetitions in the lattice samples
    # calculate the distance matrixs
    dist = torch.cdist(samples, samples)
    # set the diagonal to infinity to ignore self-distances
    dist = dist.fill_diagonal_(float('inf'))
    # count the number of repetitions
    dist = dist[dist == 0]
    return len(dist)//2


# %%
inpoints = sample_lattice(144)
print(inpoints.shape)
print(torch.unique(inpoints, dim=0).shape[0])
print(count_repetitions(inpoints))
print(torch.unique(inpoints, dim=0).shape[0] + count_repetitions(inpoints))
# %%
