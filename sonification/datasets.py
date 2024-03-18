import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .utils.matrix import square_over_bg, square_over_bg_falloff
from .utils.dsp import midi2frequency, db2amp, fm_synth_2
from .models.ddsp import Sinewave
from torchaudio.transforms import MelSpectrogram
from torchaudio.functional import amplitude_to_DB
from sklearn.preprocessing import MinMaxScaler


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
            flag="train",
            all_in_memory=True) -> None:
        super().__init__()

        # parse inputs
        self.root_path = root_path
        self.csv_path = csv_path
        self.img_size = img_size
        self.square_size = square_size

        # parse flag
        assert flag in ['train', 'val', 'all']
        self.flag = flag

        # read data
        self.all_in_memory = all_in_memory
        self.__read_data__()

    def __read_data__(self):
        # read csv
        self.df = pd.read_csv(os.path.join(self.root_path, self.csv_path))
        # filter for the set we want (train/val)
        if self.flag != 'all':
            self.df = self.df[self.df.dataset == self.flag]
        if self.all_in_memory:
            self.render_all_to_memory()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.all_in_memory:
            return self.all_tensors[idx], self.all_tensors[torch.randint(0, len(self.df), (1,))][0]
        # get the row
        row = self.df.iloc[idx]
        # get the x and y coordinates
        x = row.x
        y = row.y
        # create the image
        img = square_over_bg_falloff(x, y, self.img_size, self.square_size)
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

    def render_all_to_memory(self):
        self.all_tensors = torch.zeros(
            len(self.df), 1, self.img_size, self.img_size)
        for idx in range(len(self.df)):
            # get the row
            row = self.df.iloc[idx]
            # get the x and y coordinates
            x = row.x
            y = row.y
            # create the image
            img = square_over_bg_falloff(x, y, self.img_size, self.square_size)
            # add a channel dimension
            img = img.unsqueeze(0)
            self.all_tensors[idx] = img[0]
        print("All tensors rendered to memory")


class Sinewave_dataset(Dataset):
    """Dataset of sine waves with varying pitch and loudness"""

    def __init__(
            self,
            root_path="",
            csv_path="sinewave.csv",
            sr=44100,
            samps=16384,
            n_fft=8192,
            f_min=60,
            f_max=1200,
            pad=1,
            n_mels=64,
            power=2,
            norm="slaney",
            mel_scale="slaney",
            flag="train",
            scaler=None) -> None:
        super().__init__()

        # parse inputs
        self.root_path = root_path
        self.csv_path = csv_path
        self.sr = sr
        self.samps = samps
        self.n_fft = n_fft
        self.f_min = f_min
        self.f_max = f_max
        self.pad = pad
        self.n_mels = n_mels
        self.power = power
        self.norm = norm
        self.mel_scale = mel_scale

        # parse flag
        assert flag in ['train', 'val', 'all']
        self.flag = flag

        # generators
        self.sinewave_gen = Sinewave(sr=self.sr)
        self.mel_spec = MelSpectrogram(
            sample_rate=self.sr,
            n_fft=self.n_fft,
            f_min=self.f_min,
            f_max=self.f_max,
            pad=self.pad,
            n_mels=self.n_mels,
            power=self.power,
            norm=self.norm,
            mel_scale=self.mel_scale)

        # scaler
        if scaler is not None:
            self.scaler = scaler
        else:
            self.scaler = MinMaxScaler()
        self.scaler_fitted = False

        # read data
        self.__read_data__()

    def __read_data__(self):
        # read csv
        self.df = pd.read_csv(os.path.join(self.root_path, self.csv_path))
        # filter for the set we want (train/val)
        if self.flag != 'all':
            self.df = self.df[self.df.dataset == self.flag]
        # fit scaler
        self.fit_scaler()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.scaler_fitted:
            x_1 = self.all_tensors[idx]
            if len(x_1.shape) == 2:
                x_1 = x_1.unsqueeze(0)
            x_1 = x_1.permute(0, 2, 1)  # (B, C=1, n_mels)
            num_elems = x_1.shape[0]
            # generate random indices for the second tensor
            indices = torch.randint(0, len(self.df), (num_elems,))
            x_2 = self.all_tensors[indices].permute(
                0, 2, 1)  # (B, C=1, n_mels)
            if num_elems == 1:
                # dataloader expects to get it without the batch dimension
                return x_1[0], x_2[0]
            return x_1, x_2  # (B, C=1, n_mels) (B, C=1, n_mels)
        # get the row
        row = self.df.iloc[idx]
        # get the pitch and loudness
        pitch = row.pitch
        loudness = row.loudness
        # convert to frequency and amplitude
        freq = midi2frequency(np.array([pitch]))
        amp = db2amp(np.array([loudness]))
        num_elems = freq.shape[-1]
        # synthetize the sine wave
        if num_elems == 1:
            sinewave = self.sinewave_gen(torch.ones(
                1, self.samps) * freq) * amp
        else:
            sinewave = self.sinewave_gen(torch.ones(
                num_elems, self.samps) * freq.T) * amp.T
        # convert to float32
        sinewave = sinewave.float()
        # transform to mel spectrogram
        mel_spec = self.mel_spec(sinewave)
        # average time dimension
        mel_spec_avg = mel_spec.mean(dim=2, keepdim=True)
        mel_spec_avg_db = amplitude_to_DB(
            mel_spec_avg, multiplier=10, amin=1e-5, db_multiplier=20, top_db=80)
        # transform with scaler
        if self.scaler_fitted:
            mel_spec_avg_db = self.scaler.transform(
                mel_spec_avg_db.squeeze(-1).numpy())
            mel_spec_avg_db = torch.tensor(mel_spec_avg_db).unsqueeze(-1)
        # return the mel spectrogram
        return mel_spec_avg_db.permute(0, 2, 1)  # (B, C=1, n_mels)

    def fit_scaler(self):
        if self.flag != 'train' and self.scaler != None:
            print("Transforming dataset with external scaler")
            all_tensors = self.__getitem__(
                np.arange(len(self.df)))  # (B, C=1, n_mels)
            all_tensors = all_tensors.squeeze(1).numpy()  # (B, n_mels)
            self.all_tensors = self.scaler.transform(all_tensors)
            self.all_tensors = torch.tensor(
                self.all_tensors).unsqueeze(-1)  # (B, n_mels, C=1)
            self.scaler_fitted = True
            print("Scaler transformed")
        elif self.flag == 'train':
            all_tensors = self.__getitem__(
                np.arange(len(self.df)))  # (B, C=1, n_mels)
            all_tensors = all_tensors.squeeze(1).numpy()  # (B, n_mels)
            self.all_tensors = self.scaler.fit_transform(all_tensors)
            self.all_tensors = torch.tensor(
                self.all_tensors).unsqueeze(-1)  # (B, n_mels, C=1)
            self.scaler_fitted = True
            print("Scaler fit+transformed")


class FmSynthDataset(Dataset):
    def __init__(self, csv_path, sr=48000, dur=1):
        self.df = pd.read_csv(csv_path)
        self.sr = sr
        self.dur = dur

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        f_carrier = np.array([row.freq])
        harm_ratio = np.array([row.harm_ratio])
        mod_idx = np.array([row.mod_index])
        fm_synth = fm_synth_2(self.dur * self.sr, self.sr,
                              f_carrier, harm_ratio, mod_idx)
        return fm_synth, row.freq, row.harm_ratio, row.mod_index
