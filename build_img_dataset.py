# %%
# imports
import os
import numpy as np
from PIL import Image

# %%
# Define the paths to the folders containing the images
folder_bst = r"C:\Users\Balint Laczko\Desktop\work\Sonification\Amani_230117\merged images\BST"
folder_st = r"C:\Users\Balint Laczko\Desktop\work\Sonification\Amani_230117\merged images\ST"
folder_rf = r"C:\Users\Balint Laczko\Desktop\work\Sonification\Amani_230117\merged images\RF"

# Get a list of all the image files in each folder
bst_images = [os.path.join(folder_bst, f) for f in os.listdir(folder_bst) if f.endswith('.jpg')]
st_images = [os.path.join(folder_st, f) for f in os.listdir(folder_st) if f.endswith('.jpg')]
rf_images = [os.path.join(folder_rf, f) for f in os.listdir(folder_rf) if f.endswith('.jpg')]

# Create a list to store the image paths corresponding to the indices in the array
image_paths = bst_images + st_images + rf_images

# Define the shape of the output array
num_images = len(image_paths)
target_img_hw = 512
image_shape = (num_images, target_img_hw, target_img_hw, 2)
print(f"Found {num_images} images")

# Create the output array
images_array = np.zeros(image_shape, dtype=np.uint8)

# %%
# Test: read in the first image and print the sum of its three channels to see if there is anything in the blue channel
with Image.open(image_paths[0]) as img:
    img_array = np.array(img)
    print(np.sum(img_array, axis=(0, 1)))

# %%
# Test: read in the first image and zero its blue channel
with Image.open(image_paths[0]) as img:
    img_array = np.array(img)
    img_array[:, :, 2] = 0
    img = Image.fromarray(img_array)
    img.show()
    print(np.sum(img_array, axis=(0, 1)))


# %%
# Loop over the image paths and load each image into the array
for i, path in enumerate(image_paths):
    with Image.open(path) as img:
        # resize to target hw
        img = img.resize((target_img_hw, target_img_hw))
        img_array = np.array(img)
        # discard blue channel
        img_array = img_array[:, :, :2]
        images_array[i] = img_array

# %%
# export the array to a npy file
target_path = r"C:\Users\Balint Laczko\Desktop\work\Sonification\Amani_230117\merged images\images_array.npy"
np.save(target_path, images_array)

# %%
