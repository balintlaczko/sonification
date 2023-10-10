# %%
# imports
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

# %%

class Amanis_RG_dataset(Dataset):
    """Amani's dataset of merged images of the RF, ST and BST"""

    def __init__(self, data_npy):
        """
        Args:
            data_npy (string): Path to the npy file with the images (blue channel already dropped).
        """
        self.images_array = np.load(data_npy)

    def __len__(self):
        return len(self.images_array)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.images_array[idx]
        sample = torch.from_numpy(sample).float() / 255
        # channels last to channels first
        sample = sample.permute(2, 0, 1)

        return sample
