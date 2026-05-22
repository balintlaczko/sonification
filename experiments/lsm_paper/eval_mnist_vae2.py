# %%
# imports

import os
import torch
import numpy as np
import pandas as pd
from sonification.models.models import PlImgFactorVAE
from sonification.datasets import MNISTPairDataset
from sonification.utils.tensor import scale
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import itertools
import time
from torchvision import datasets, transforms
import math
from pythonosc import dispatcher, osc_server, udp_client

# %%
ckpt_path = '../../ckpt/mnist_vae2'
model_version = '3.5'
ckpt_name = 'v' + model_version
ckpt_path = os.path.join(ckpt_path, ckpt_name)
# list files, find the one that has "last" or "best" in it
key = "last"  # "best or "last"
ckpt_files = [f for f in os.listdir(ckpt_path) if key in f]
if len(ckpt_files) == 0:
    raise ValueError(f"No checkpoint file found in {ckpt_path} with '{key}' in the name")
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
print(f"Predicted image shape: {decoded.shape}")

# %%
# create dataset

transform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.ToTensor()
])

dataset = MNISTPairDataset(root='./data', train=False, download=True, transform=transform)

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
z_all_min, z_all_max = z_all.min(), z_all.max()
print(f"Latent space range: min={z_all_min}, max={z_all_max}")

# %%
# create a scatter plot of the latent space
num_dims = z_all.shape[1]
dim_pairs = list(itertools.combinations(range(num_dims), 2))
num_pairs = len(dim_pairs)

if num_pairs > 0:
    cols = math.ceil(math.sqrt(num_pairs))
    rows = math.ceil(num_pairs / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

    # Flatten the axes array to make it easy to iterate over 1D index
    axes = np.atleast_1d(axes).flatten()

    # Set font properties for the plot
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 12
    })

    # --- Calculate global min and max for consistent axis scaling ---
    global_min = z_all.min()
    global_max = z_all.max()
    # Add a 5% margin so points don't sit exactly on the plot boundaries
    margin = (global_max - global_min) * 0.05
    axis_min = global_min - margin
    axis_max = global_max + margin

    for i, (dim1, dim2) in enumerate(dim_pairs):
        ax = axes[i]

        # Consider lowering size (s) and adding alpha for better visibility if data is dense
        ax.scatter(z_all[:, dim1], z_all[:, dim2], s=1, alpha=0.5)
        ax.set_xlabel(f"Dim {dim1}")
        ax.set_ylabel(f"Dim {dim2}")
        ax.set_aspect('equal', adjustable='datalim') 

        # --- Apply the global axis limits ---
        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)
        
        # Optional: Disable axis ticks if it gets too cluttered
        # ax.set_xticks([])
        # ax.set_yticks([])

    # Hide any extra subplots in the grid that aren't used
    for i in range(num_pairs, len(axes)):
        axes[i].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.subplots_adjust(wspace=0.3) 
    plt.show()
    # plt.savefig("mnist_model_latent_space.png", dpi=300)

    # Reset rcParams to default to not affect other plots
    plt.rcdefaults()


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

# %%
# set up an OSC server to receive latent codes from Max and send back decoded images

client = udp_client.SimpleUDPClient("127.0.0.1", 12345)

def handle_latent_code(unused_addr, *args):
    # args is a tuple of the latent code
    latent_code = torch.tensor(args, dtype=torch.float32)
    # decode the latent code
    with torch.no_grad():
        latent_code_scaled = scale(latent_code, -1, 1, z_all_min, z_all_max)
        predicted_images = model.model.decoder(latent_code_scaled.unsqueeze(0).to(device))

    # Flatten the image to a 1D list of floats 
    flat_image = predicted_images.squeeze().cpu().numpy().flatten().tolist()

    # send back the decoded images to Max
    client.send_message("/decoded_images", flat_image)

# create a dispatcher
d = dispatcher.Dispatcher()
# map the OSC address to the function
d.map("/latent_code", handle_latent_code)

# create an OSC server
ip = "127.0.0.1"
port = 12346
server = osc_server.ThreadingOSCUDPServer((ip, port), d)
print(f"Serving on {server.server_address}")
# start the server
try:
    server.serve_forever()
except KeyboardInterrupt:
    print("Server stopped by user")
finally:
    server.shutdown()
    print("Server shutdown")
    server.server_close()
    print("Server closed")

########################################################################


# %%
# plot 4 random samples from the dataset
# Set font properties for the plot
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 18
})
num_images = 4
random_indices = np.random.choice(len(dataset), num_images, replace=False)
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for i, idx in enumerate(random_indices):
    x, y = dataset[idx]
    row = i // 2
    col = i % 2
    axes[row, col].imshow(x[0, ...].cpu().numpy(), cmap="gray")
    # axes[row, col].set_title(f"Sample Index: {idx}")
    axes[row, col].axis('off')
plt.tight_layout()
plt.savefig("image_samples.png", dpi=300)
# Reset rcParams to default to not affect other plots
plt.rcdefaults()

# %%
# set percentiles
percentile_low = 1
percentile_high = 99
steps = 20

z_x_min = np.percentile(z_all[:, 0], percentile_low)
z_x_max = np.percentile(z_all[:, 0], percentile_high)
z_y_min = np.percentile(z_all[:, 1], percentile_low)
z_y_max = np.percentile(z_all[:, 1], percentile_high)

x_steps = torch.linspace(z_x_min, z_x_max, steps)
y_steps = torch.linspace(z_y_min, z_y_max, steps)

z_x_min, z_x_max, z_y_min, z_y_max

# %%
# create a plot for traversing in the latent space
fig, ax = plt.subplots(len(x_steps), len(y_steps), figsize=(20, 20))
for y_idx, y_step in enumerate(y_steps):
    for x_idx, x_step in enumerate(x_steps):
        latent_sample = torch.tensor([x_step, y_step]).unsqueeze(0)
        decoded = model.model.decoder(latent_sample.to(model.device))
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
# plt.savefig("traverse_latent_space_64x2.png")
plt.show()

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
    x_recon, mean, logvar, z = model(x.to(model.device))
    z = z.detach().cpu().numpy()
    ax[0, i].imshow(x[0, 0, ...], cmap="gray")
    ax[0, i].set_title(f"Input {i}")
    ax[1, i].scatter(z[0, 0], z[0, 1])
    # fix scaling of the plot
    ax[1, i].set_xlim(-3, 3)
    ax[1, i].set_ylim(-3, 3)
    ax[1, i].set_title(f"Latent {i}")

# %%