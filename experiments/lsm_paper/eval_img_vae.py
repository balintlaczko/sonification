# %%
# imports

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sonification.datasets import White_Square_dataset
from sonification.models.models import PlFactorVAE
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sonification.utils.matrix import view
from torch.utils.data import DataLoader

# %%
# load the dataset
root_path = './'
csv_path = 'white_squares_xy_16_4.csv'
img_size = 16
square_size = 4
dataset = White_Square_dataset(
    root_path, csv_path, img_size, square_size, flag="all")

# %%
# load the model from the checkpoint
ckpt_path = '../../ckpt/white_squares_fvae/factorvae-v6/factorvae-v6_last_epoch=129780.ckpt'
model = PlFactorVAE.load_from_checkpoint(ckpt_path)
model.eval()

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
plt.savefig("test_figure.png")
plt.show()

# %%
# save plot to image
plt.savefig("test_figure.png")

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
fig, ax = plt.subplots(len(x_steps), len(y_steps), figsize=(20, 10))
for y_idx, y_step in enumerate(y_steps):
    for x_idx, x_step in enumerate(x_steps):
        latent_sample = torch.tensor([x_step, y_step]).unsqueeze(0)
        decoded = model.decode(latent_sample.to(model.device))
        ax[x_idx, y_idx].imshow(
            decoded[0, 0, ...].detach().cpu().numpy(), cmap="gray")
        # ax[x_idx, y_idx].set_title(
        #     f"{str(x_idx).zfill(2)}_{str(y_idx).zfill(2)}")
        # remove axis labels
        ax[x_idx, y_idx].axis('off')
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
