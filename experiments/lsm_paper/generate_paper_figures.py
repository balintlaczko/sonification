# %%
# imports
import numpy as np
import matplotlib.pyplot as plt
from sonification.datasets import Sinewave_dataset, White_Square_dataset
from sonification.utils.matrix import square_over_bg_falloff, view
import torch
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
# create a function that maps a color pan value form 0 to 1 to hues between red and green, then converts to rgb
def hue_to_rgb(x):
    """
    Convert a float value between 0 and 1 into an RGB color,
    where 0 maps to red and 1 maps to green via hue.

    Args:
        x (float): A float between 0 and 1 representing the input value.

    Returns:
        tuple: A tuple of (r, g, b) values, each between 0 and 255.
    """
    if not (0 <= x <= 1):
        raise ValueError("Input must be a float between 0 and 1.")

    # Map x to a hue in degrees (0 = red, 120 = green)
    hue = x * 120

    # Convert HSL to RGB (assuming full saturation and lightness = 0.5)
    c = 1  # Chroma (saturation is full)
    x_component = c * (1 - abs((hue / 60) % 2 - 1))

    if 0 <= hue < 60:
        r, g, b = c, x_component, 0
    elif 60 <= hue <= 120:
        r, g, b = x_component, c, 0
    else:
        r, g, b = 0, 0, 0  # Should not occur since hue is [0, 120]

    # Convert RGB from [0, 1] range to [0, 255] range
    # r, g, b = int(r * 255), int(g * 255), int(b * 255)

    return r, g, b

# Example usage
print(hue_to_rgb(0.0))  # Red
print(hue_to_rgb(0.5))  # Yellow
print(hue_to_rgb(1.0))  # Green


# %%
# generate random samples with varying square size and color

n_samples = 8
fig, axs = plt.subplots(2, 4, figsize=(12, 6))
color_pan_range = np.linspace(0, 1, n_samples)
for i in range(n_samples):
    # get a random square size
    square_size = np.random.randint(2, 17)
    # get a random position
    x = np.random.randint(0, 64 - square_size)
    y = np.random.randint(0, 64 - square_size)
    # get a random color
    color_pan = np.random.rand()
    # color_pan = color_pan_range[i]
    color = hue_to_rgb(color_pan)
    # instead generate a random hue between red and green and convert to rgb

    # generate the image
    img1 = square_over_bg_falloff(x, y, square_size=square_size, color=color)
    ax = axs[i // 4, i % 4]
    ax.imshow(img1)
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
