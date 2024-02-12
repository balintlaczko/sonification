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
csv_path = 'white_squares_xy_64_4.csv'
img_size = 64
square_size = 4
dataset = White_Square_dataset(
    root_path, csv_path, img_size, square_size, flag="train")

# %%
# load the model from the checkpoint
ckpt_path = '../../ckpt/white_squares_fvae/v6-vae-tanh/v6-vae-tanh_last_epoch=3012.ckpt'
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
    x_recon, _, _, _ = model(x.cuda())
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
