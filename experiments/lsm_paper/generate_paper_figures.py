# %%
# imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sonification.datasets import Sinewave_dataset, White_Square_dataset
import torch
import os
from matplotlib import colors, gridspec

# %%
# generate random samples from the white square dataset

# create the dataset
dataset = White_Square_dataset(
    root_path="", 
    csv_path="white_squares_xy_64_2.csv", 
    flag="all",
    img_size=64,
    square_size=2
    )

# generate a 2 x 4 plot of random samples
n_samples = 8
fig, axs = plt.subplots(2, 4, figsize=(12, 6))
for i in range(n_samples):
    img1, img2 = dataset[i]
    ax = axs[i // 4, i % 4]
    ax.imshow(img1.squeeze(), cmap="gray")
    ax.axis("off")

plt.tight_layout()
plt.show()

# %%
# generate random samples from the sinewave dataset

# create the dataset
dataset = Sinewave_dataset(
    root_path="", 
    csv_path="sinewave.csv", 
    flag="all",
    )

def plot_random_samples(dataset, num_samples=64):
    """Save a figure of N reconstructions"""
    # get the train dataset's scaler
    scaler = dataset.scaler
    # get a random indices
    indices = torch.randint(0, len(dataset), (num_samples,))
    x, _ = dataset[indices]
    x = x.squeeze(1).cpu().numpy()
    x_scaled = scaler.inverse_transform(x).T
    # Create a normalization object for colormap
    norm = colors.Normalize(vmin=np.min(x_scaled), vmax=np.max(x_scaled))
    # create figure
    fig = plt.figure(figsize=(5, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 0.05])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    # Create the first subplot
    cax = ax0.matshow(x_scaled, interpolation='nearest',
                cmap='viridis', norm=norm)
    # ax0.set_title('Ground Truth')
    # Create a colorbar that is shared between the two subplots
    fig.colorbar(cax, cax=ax1)
    plt.show()

plot_random_samples(dataset)

# %%
