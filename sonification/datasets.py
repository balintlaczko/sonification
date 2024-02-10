import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .utils.matrix import square_over_bg

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


class White_Square_dataset(Dataset):
    """Dataset of white squares on black background"""

    def __init__(
            self,
            root_path,
            csv_path="white_squares_xy.csv",
            img_size=512,
            square_size=50,
            flag="train") -> None:
        super().__init__()

        # parse inputs
        self.root_path = root_path
        self.csv_path = csv_path
        self.img_size = img_size
        self.square_size = square_size

        # parse flag
        assert flag in ['train', 'val']
        self.flag = flag

        # read data
        self.__read_data__()


    def __read_data__(self):
        # read csv
        self.df = pd.read_csv(os.path.join(self.root_path, self.csv_path))
        # filter for the set we want (train/val)
        self.df = self.df[self.df.dataset == self.flag]


    def __len__(self):
        return len(self.df)
    

    def __getitem__(self, idx):
        # get the row
        row = self.df.iloc[idx]
        # get the x and y coordinates
        x = row.x
        y = row.y
        # create the image
        img = square_over_bg(x, y, self.img_size, self.square_size)
        # add a channel dimension
        img = img.unsqueeze(0)

        # get another random row
        row = self.df.sample().iloc[0]
        # get the x and y coordinates
        x = row.x
        y = row.y
        # create the image
        img2 = square_over_bg(x, y, self.img_size, self.square_size)
        # add a channel dimension
        img2 = img2.unsqueeze(0)

        # return the images
        return img, img2