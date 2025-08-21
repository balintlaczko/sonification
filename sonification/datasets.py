import os
import numpy as np
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from .utils.matrix import square_over_bg, square_over_bg_falloff
from .utils.dsp import midi2frequency, db2amp
from .models.ddsp import Sinewave, FMSynth
from torchaudio.transforms import MelSpectrogram
from torchaudio.functional import amplitude_to_DB
from sklearn.preprocessing import MinMaxScaler
import json
from .models.ddsp import FMSynth
from .utils.tensor import midi2frequency


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


class CellularDataset(Dataset):
    def __init__(
            self,
            csv_path,
            root_dir,
            img_size=2048,
            kernel_size=256,
            stride=10,
            flag="train") -> None:
        super().__init__()

        self.root_dir = root_dir
        self.img_size = img_size
        self.kernel_size = kernel_size
        self.stride = stride

        # count patches, map patch indices to coordinates
        self.n_patches_per_img = (
            (self.img_size - self.kernel_size) // self.stride)**2 + 1
        self.idx2yx = []
        for i in range(0, self.img_size - self.kernel_size - 1, self.stride):
            for j in range(0, self.img_size - self.kernel_size - 1, self.stride):
                self.idx2yx.append((i, j))
        self.n_patches_per_img = len(self.idx2yx)

        # read the data and get the min/max values
        self.read_data(csv_path, flag)
        self.get_data_minmax()

    def read_data(self, csv_path, flag):
        # parse flag
        assert flag in ['train', 'val', 'all']
        self.flag = flag

        # read the csv of image paths
        self.df = pd.read_csv(csv_path)
        # filter for the set we want (train/val)
        if self.flag != 'all':
            self.df = self.df[self.df.dataset == self.flag]

    def get_data_minmax(self):
        # get the min/max values
        if self.flag == 'val':
            return
        self.r_min = self.df.r_min.min()
        self.r_max = self.df.r_max.max()
        self.g_min = self.df.g_min.min()
        self.g_max = self.df.g_max.max()

    def scale(self, patch_r, patch_g):
        patch_r = (patch_r - self.r_min) / (self.r_max - self.r_min)
        patch_g = (patch_g - self.g_min) / (self.g_max - self.g_min)
        return patch_r, patch_g

    def scale_inv(self, patch_r, patch_g):
        patch_r = patch_r * (self.r_max - self.r_min) + self.r_min
        patch_g = patch_g * (self.g_max - self.g_min) + self.g_min
        return patch_r, patch_g

    def __len__(self):
        # return len(self.df) * self.n_patches_per_img
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # img_idx = idx // self.n_patches_per_img
        # patch_idx = idx % self.n_patches_per_img

        # get the image paths
        # row = self.df.iloc[img_idx]
        row = self.df.iloc[idx]
        img_path_r = os.path.join(self.root_dir, row.path_r)
        img_path_g = os.path.join(self.root_dir, row.path_g)

        # load the images
        img_r = cv2.imread(img_path_r, cv2.IMREAD_UNCHANGED)
        img_g = cv2.imread(img_path_g, cv2.IMREAD_UNCHANGED)

        # # get the patch coordinates
        # y, x = self.idx2yx[patch_idx]
        # patch_r = img_r[y:y+self.kernel_size, x:x+self.kernel_size]
        # patch_g = img_g[y:y+self.kernel_size, x:x+self.kernel_size]

        # scale the patches
        # patch_r, patch_g = self.scale(patch_r, patch_g)
        patch_r, patch_g = self.scale(img_r, img_g)

        # stack the patches
        patch = np.stack([patch_r, patch_g], axis=2)
        patch = patch.astype(np.float32)
        patch = torch.from_numpy(patch).permute(2, 0, 1)  # C, H, W

        return patch


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
        assert self.power in [1, 2]
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
            self.scaler_fitted = True
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
            mel_spec_avg,
            multiplier=10 if self.power == 2 else 20,
            amin=1e-5,
            db_multiplier=20,
            top_db=80)
        # mel_spec_avg_db = 20 * torch.log10(mel_spec_avg + 1e-5)
        # mel_spec_avg_db = mel_spec_avg
        # transform with scaler
        if self.scaler_fitted:
            mel_spec_avg_db = self.scaler.transform(
                mel_spec_avg_db.squeeze(-1).numpy())
            mel_spec_avg_db = torch.tensor(mel_spec_avg_db).unsqueeze(-1)
        # return the mel spectrogram
        return mel_spec_avg_db.permute(0, 2, 1)  # (B, C=1, n_mels)

    def fit_scaler(self):
        if self.flag not in ['train', 'all'] and self.scaler_fitted:
            print("Transforming dataset with external scaler")
            print(self.flag)
            print(self.scaler)
            self.scaler_fitted = False # TODO: ugly fix, please fix
            all_tensors = self.__getitem__(
                np.arange(len(self.df)))  # (B, C=1, n_mels)
            all_tensors = all_tensors.squeeze(1).numpy()  # (B, n_mels)
            self.all_tensors = self.scaler.transform(all_tensors)
            self.all_tensors = torch.tensor(
                self.all_tensors).unsqueeze(-1)  # (B, n_mels, C=1)
            self.scaler_fitted = True
            print("Scaler transformed")
        elif self.flag in ['train', 'all']:
            self.scaler_fitted = False # TODO: ugly fix, please fix
            all_tensors = self.__getitem__(
                np.arange(len(self.df)))  # (B, C=1, n_mels)
            all_tensors = all_tensors.squeeze(1).numpy()  # (B, n_mels)
            self.all_tensors = self.scaler.fit_transform(all_tensors)
            self.all_tensors = torch.tensor(
                self.all_tensors).unsqueeze(-1)  # (B, n_mels, C=1)
            self.scaler_fitted = True
            print("Scaler fit+transformed")


class FmSynthDataset(Dataset):
    def __init__(self, csv_path=None, dataframe=None, sr=48000, dur=1):
        if dataframe is not None:
            self.df = dataframe
        else:
            self.df = pd.read_csv(csv_path)
        self.sr = sr
        self.dur = dur
        self.synth = FMSynth(sr)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        nsamps = int(self.dur * self.sr)
        row = self.df.iloc[idx]
        row_tensor = torch.tensor(row.values, dtype=torch.float32)
        row_tensor = row_tensor.unsqueeze(0) if len(
            row_tensor.shape) < 2 else row_tensor
        # find the column id for "freq", "harm_ratio", "mod_index"
        freq_col_id = self.df.columns.get_loc("freq")
        harm_ratio_col_id = self.df.columns.get_loc("harm_ratio")
        mod_index_col_id = self.df.columns.get_loc("mod_index")
        # extract the carrier frequency, harm_ratio, mod_index, repeat for nsamps
        carr_freq = row_tensor[:, freq_col_id].unsqueeze(-1).repeat(1, nsamps)
        harm_ratio = row_tensor[:,
                                harm_ratio_col_id].unsqueeze(-1).repeat(1, nsamps)
        mod_index = row_tensor[:,
                               mod_index_col_id].unsqueeze(-1).repeat(1, nsamps)
        fm_synth = self.synth(carr_freq, harm_ratio, mod_index)
        return fm_synth


class FMTripletDataset(Dataset):
    def __init__(self, json_path, sr=48000, n_samples=8192, n_fft=4096, f_min=20, f_max=16000, n_mels=512, power=1, normalized=True):
        with open(json_path, 'r') as f:
            self.triplets = json.load(f)

        self.sr = sr
        self.n_samples = n_samples
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.fm_synth = FMSynth(sr=self.sr).to(self.device)
        self.mel_spectrogram = MelSpectrogram(
            sample_rate=self.sr,
            n_fft=n_fft,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=power,
            normalized=normalized
        ).to(self.device)
        self.fm_synth.eval()
        self.mel_spectrogram.eval()

    def __len__(self):
        return self.triplets["totalTrials"]

    def _params_to_spec(self, params):
        # Convert parameter dict to a mel spectrogram tensor
        freq = torch.tensor([params["carrier"]]).unsqueeze(1).repeat(1, self.n_samples)
        harm_ratio = torch.tensor([params["harmRatio"]]).unsqueeze(1).repeat(1, self.n_samples)
        mod_index = torch.tensor([params["modIndex"]]).unsqueeze(1).repeat(1, self.n_samples)

        audio = self.fm_synth(freq, harm_ratio, mod_index).detach()
        mel_spec = self.mel_spectrogram(audio.unsqueeze(1))
        # mel_spec = scale(mel_spec, mel_spec.min(), mel_spec.max(), 0, 1)
        mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min()).clamp_min(torch.finfo(mel_spec.dtype).eps)
        return mel_spec.squeeze(0)

    def __getitem__(self, idx):
        triplet_data = self.triplets["data"][idx]

        # Synthesize spectrograms for A, B, and X
        spec_a = self._params_to_spec(triplet_data["A"])
        spec_b = self._params_to_spec(triplet_data["B"])
        spec_x = self._params_to_spec(triplet_data["X"])

        # The user chose whether X is closer to A or B.
        # This defines our Anchor, Positive, and Negative.
        if triplet_data["choice"] == "A":
            # Anchor is X, Positive is A, Negative is B
            anchor = spec_x
            positive = spec_a
            negative = spec_b
        else: # choice == "B"
            # Anchor is X, Positive is B, Negative is A
            anchor = spec_x
            positive = spec_b
            negative = spec_a

        return anchor, positive, negative