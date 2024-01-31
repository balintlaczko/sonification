# %%
# imports
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from utils import view

# %%
# constants
IMG_SIZE = 256
RECT_SIZE = 50

# %%
# generate an example image of a white rectangle on a black background
img = np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.uint8)
# in the middle of the image
cv2.rectangle(img, (IMG_SIZE//2 - RECT_SIZE//2, IMG_SIZE//2 - RECT_SIZE//2),
              (IMG_SIZE//2 + RECT_SIZE//2, IMG_SIZE//2 + RECT_SIZE//2), (255), -1)

view(img)

# %%
# now at random positions
for _ in range(10):
    img = np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.uint8)
    x = np.random.randint(0, IMG_SIZE - RECT_SIZE)
    y = np.random.randint(0, IMG_SIZE - RECT_SIZE)
    cv2.rectangle(img, (x, y), (x + RECT_SIZE, y + RECT_SIZE), (255), -1)
    view(img)

# %%
# generate all possible xy coordinates
x_range = np.arange(0, IMG_SIZE - RECT_SIZE)
y_range = np.arange(0, IMG_SIZE - RECT_SIZE)
x_coords, y_coords = np.meshgrid(x_range, y_range)
# collect xy pairs in a list
xy_coords = np.stack((x_coords, y_coords), axis=2)
# flatten the list
xy_coords = xy_coords.reshape(-1, 2)
xy_coords.shape

# %%
# visualize the first few xy pairs
for i in range(10):
    img = np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.uint8)
    x, y = xy_coords[i]
    cv2.rectangle(img, (x, y), (x + RECT_SIZE, y + RECT_SIZE), (255), -1)
    view(img)
    print(i, x, y)


# %%
