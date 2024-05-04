# %%
# imports
# from torchsummary import summary
import numpy as np
from lightning.pytorch import LightningModule
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from sonification.models.ddsp import FMSynth, Sinewave
from sonification.models.layers import ConvEncoder1D, LinearProjector
from sonification.utils.dsp import transposition2duration
from sonification.utils.array import fluid_dataset2array
from sonification.utils.tensor import db2amp, amp2db, frequency2midi, midi2frequency
from torchaudio.functional import amplitude_to_DB
from torchaudio.transforms import MelSpectrogram
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KDTree
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
from torch import nn
import os
from matplotlib import pyplot as plt
import pandas as pd
import json
from copy import deepcopy
from collections import OrderedDict
from sys import stderr
import torchyin

# %%


class Args:
    pass


class FmSynthDataset(Dataset):
    def __init__(self, csv_path=None, dataframe=None, sr=48000, dur=1):
        if dataframe is not None:
            self.df = dataframe
        else:
            self.df = pd.read_csv(csv_path)
        self.sr = sr
        self.dur = dur
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.synth = FMSynth(sr).to(self.device)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        nsamps = int(self.dur * self.sr)
        row = self.df.iloc[idx]
        row_tensor = torch.tensor(
            row.values, dtype=torch.float32).to(self.device)
        row_tensor = row_tensor.unsqueeze(0) if len(
            row_tensor.shape) < 2 else row_tensor
        # find the column id for "freq", "harm_ratio", "mod_index"
        freq_col_id = self.df.columns.get_loc("freq")
        harm_ratio_col_id = self.df.columns.get_loc("harm_ratio")
        mod_index_col_id = self.df.columns.get_loc("mod_index")
        # extract the carrier frequency, harm_ratio, mod_index, repeat for nsamps
        carr_freq = row_tensor[:,
                               freq_col_id].unsqueeze(-1).repeat(1, nsamps)
        harm_ratio = row_tensor[:,
                                harm_ratio_col_id].unsqueeze(-1).repeat(1, nsamps)
        mod_index = row_tensor[:,
                               mod_index_col_id].unsqueeze(-1).repeat(1, nsamps)
        fm_synth = self.synth(carr_freq, harm_ratio, mod_index)
        return fm_synth


class FmMel2PCADataset(Dataset):
    def __init__(
            self,
            fm_params_dataframe,
            pca_array,
            sr=48000,
            dur=1,
            n_mels=200,
            mel_scaler=None,
            pca_scaler=None):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.fm_synth = FmSynthDataset(
            dataframe=fm_params_dataframe, sr=sr, dur=dur)
        self.pca = torch.tensor(pca_array).float().to(self.device)
        self.sr = sr
        self.dur = dur
        self.n_mels = n_mels
        self.mel_spec = MelSpectrogram(
            sample_rate=self.sr,
            n_fft=4096,
            f_min=70,
            f_max=20000,
            pad=1,
            n_mels=self.n_mels,
            power=2,
            norm='slaney',
            mel_scale='slaney').to(self.device)

        # hold mel tensors in memory
        self.all_mels = torch.zeros(
            len(self.fm_synth), 1, self.n_mels).to(self.device)
        self.load_all_mels()

        # scalers
        self.mel_scaler = mel_scaler
        self.pca_scaler = pca_scaler

    def fit_scalers(self):
        # fit mel scaler
        if self.mel_scaler is None:
            self.mel_scaler = MinMaxScaler()
        self.fit_mel_scaler()
        # fit pca scaler
        if self.pca_scaler is None:
            self.pca_scaler = MinMaxScaler()
        print("Fitting PCA scaler...")
        self.pca_scaler.fit(self.pca.cpu().numpy())
        print("PCA scaler fit")

    def __len__(self):
        return len(self.fm_synth)

    def __getitem__(self, idx):
        # print(idx)
        mel = self.all_mels[idx]  # (B, 1, n_mels)
        if isinstance(idx, int):
            mel = mel.unsqueeze(0)
        mel_scaled = self.mel_scaler.transform(
            mel.squeeze(1).cpu().numpy())  # (B, n_mels)
        if not isinstance(idx, int):
            mel_scaled = torch.tensor(
                mel_scaled, device=self.device).reshape(-1, 1, self.n_mels)  # (B, 1, n_mels)
        pca = self.pca[idx]  # (B, 2)
        if isinstance(idx, int):
            pca = pca.unsqueeze(0)
        pca_scaled = self.pca_scaler.transform(pca.cpu().numpy())
        pca_scaled = torch.tensor(pca_scaled, device=self.device)
        if isinstance(idx, int):
            pca_scaled = pca_scaled.squeeze(0)
        return mel_scaled, pca_scaled

    def get_mel(self, idx):
        fm_synth = self.fm_synth[idx]  # (B, T)
        mel = self.mel_spec(fm_synth)  # (B, n_mels, T)
        mel_avg = mel.mean(dim=-1, keepdim=True)  # (B, n_mels, 1)
        mel_avg_db = amplitude_to_DB(
            mel_avg, multiplier=10, amin=1e-5, db_multiplier=20, top_db=80)  # (B, n_mels, 1)
        # apply a -12 dB reduction on the upper half of the mel spectrogram
        # TODO: use lowpass filter as an nn.Module instead
        # mel_avg_db[:, self.n_mels // 2:, :] -= 12
        return mel_avg_db

    def load_all_mels(self):
        batch_size = 256
        # create batches of indices
        indices = torch.arange(len(self.fm_synth))
        batches = torch.split(indices, batch_size)
        pbar = tqdm(batches)
        pbar.set_description("Loading mel tensors")
        # iterate through all fm synths in batches
        for idx in pbar:
            mel = self.get_mel(idx)  # (B, n_mels, 1)
            # swap channels and n_mels dims
            mel = mel.permute(0, 2, 1)  # (B, 1, n_mels)
            self.all_mels[idx] = mel
        print("All mel tensors loaded to memory")

    def fit_mel_scaler(self):
        all_mels = self.all_mels.squeeze(1).cpu().numpy()  # (B, n_mels)
        print("Fitting mel scaler...")
        self.mel_scaler.fit(all_mels)
        print("Mel scaler fit")


class FmMelContrastiveDataset(Dataset):
    def __init__(
            self,
            fm_params_dataframe,
            sr=48000,
            dur=1,
            n_mels=200,
            mel_scaler=None):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.fm_synth = FmSynthDataset(
            dataframe=fm_params_dataframe, sr=sr, dur=dur)
        self.sr = sr
        self.dur = dur
        self.n_mels = n_mels
        self.apply_transpose = True
        self.n_transpositions = 1
        self.apply_noise = True
        self.noise_db_range = 2  # 0 to 2 dB noise will be added
        self.apply_loudness = True
        self.loudness_db_range = 6  # -3 to 3 dB
        self.apply_masking = False
        self.masking_upper_half_only = True
        self.masking_min_p = 0.3
        self.masking_max_p = 0.6
        self.mel_spec = MelSpectrogram(
            sample_rate=self.sr,
            n_fft=4096,
            f_min=70,
            f_max=20000,
            pad=1,
            n_mels=self.n_mels,
            power=2,
            norm='slaney',
            mel_scale='slaney').to(self.device)

        # scalers
        self.mel_scaler = mel_scaler

    def __len__(self):
        return len(self.fm_synth)

    def __getitem__(self, idx):
        assert self.mel_scaler is not None
        # get random numbers between -2 and 2 for transposition
        transpositions = torch.rand(self.n_transpositions) * 4 - 2
        # append 0 to the front, so the first element is the original
        transpositions = torch.cat((torch.tensor([0]), transpositions))
        if not self.apply_transpose:
            transpositions = torch.zeros_like(transpositions)
        mels = []
        waveforms = None
        for i, transpose in enumerate(transpositions):

            mel, y = self.get_mel(
                idx,
                transpose,
                change_loudness=i != 0,
                add_noise=i != 0)  # (B, n_mels, 1), (B, T)

            # mel = self.get_mel(
            #     idx,
            #     transpose,
            #     change_loudness=i != 0,
            #     add_noise=i != 0)  # (B, n_mels, 1)

            if i == 0:
                waveforms = y

            mel_scaled = self.mel_scaler.transform(
                mel.squeeze(-1).cpu().numpy())  # (B, n_mels)
            if not isinstance(idx, int):
                mel_scaled = torch.tensor(
                    mel_scaled, device=self.device).reshape(-1, 1, self.n_mels)  # (B, 1, n_mels)
            if self.apply_masking and i != 0:
                p = torch.rand(1) * \
                    (self.masking_max_p - self.masking_min_p) + self.masking_min_p
                mel_scaled = self.mask_mels(
                    mel_scaled, p=p, upper_half_only=self.masking_upper_half_only)
            mels.append(mel_scaled)

        # a list of tensors of shape (B, 1, n_mels)
        # return mels
        return mels, waveforms

    def mask_mels(self, mels, p=0.3, upper_half_only=True):
        b_size, _, num_mels = mels.shape
        if upper_half_only:
            num_mels = num_mels // 2
        masked_bins = int(num_mels * p)
        mels_indices = torch.arange(num_mels).unsqueeze(0).repeat(b_size, 1)
        for i in range(b_size):
            mels_indices[i] = mels_indices[i][torch.randperm(num_mels)]
        mels_indices = mels_indices[...,
                                    :masked_bins].unsqueeze(1).to(self.device)
        if upper_half_only:
            mels_indices += num_mels
        mels_masked = mels.clone()
        mels_masked.scatter_(2, mels_indices, 0)
        return mels_masked

    def get_mel(self, idx, transpose=0, change_loudness=False, add_noise=False):
        fm_synth = self.fm_synth[idx]  # (B, T)

        if change_loudness:
            # one random dB value per example in batch
            loudness = torch.rand(
                fm_synth.shape[0], 1) * self.loudness_db_range - (self.loudness_db_range / 2)  # -3 to 3 dB
            # scale fm_synth to dB
            fm_synth *= db2amp(loudness).to(self.device)

        if add_noise:
            # full range noise from -1 to 1
            noise = torch.rand_like(fm_synth) * 2 - 1
            # one random dB value per example in batch
            noise_loudness = torch.rand(
                fm_synth.shape[0], 1) * self.noise_db_range  # 0 to 2 dB
            noise_amp = db2amp(noise_loudness) - 1
            # scale noise to dB
            noise *= noise_amp.to(self.device)
            # add noise
            fm_synth += noise

        if transpose != 0:
            target_dur = transposition2duration(transpose)
            target_sr = int(self.sr * target_dur)
            # fm_synth = resample(
            #     fm_synth.numpy(), orig_sr=self.sr, target_sr=target_sr)  # (B, T')
            # fm_synth = torch.tensor(fm_synth)
            # TODO: ugly linear interpolation instead of proper resampling since torchaudio resample is broken and librosa resample is cpu only
            fm_synth = torch.nn.functional.interpolate(
                fm_synth.reshape(-1, 1, fm_synth.shape[-1]), scale_factor=target_dur, mode='linear').squeeze(1)  # (B, T')
        mel = self.mel_spec(fm_synth)  # (B, n_mels, T')
        mel_avg = mel.mean(dim=-1, keepdim=True)  # (B, n_mels, 1)
        mel_avg_db = amplitude_to_DB(
            mel_avg, multiplier=10, amin=1e-5, db_multiplier=20, top_db=80)  # (B, n_mels, 1)
        # apply a -12 dB reduction on the upper half of the mel spectrogram
        # TODO: use lowpass filter as an nn.Module instead
        # mel_avg_db[:, self.n_mels // 2:, :] -= 12
        # return mel_avg_db
        return mel_avg_db, fm_synth


# %%
class ContrastiveSinusoidalDataset(Dataset):
    def __init__(self, num_partials=11, min_pitch=45, pitch_range=96, min_pitchdiff_lowest=7, sr=48000, dur=1, n_mels=200):
        self.num_partials = num_partials
        self.min_pitch = min_pitch
        self.pitch_range = pitch_range
        self.min_pitchdiff_lowest = min_pitchdiff_lowest
        self.sr = sr
        self.dur = dur
        self.n_mels = n_mels
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.synth = Sinewave(sr).to(self.device)
        self.mel_spec = MelSpectrogram(
            sample_rate=self.sr,
            n_fft=4096,
            f_min=70,
            f_max=20000,
            pad=1,
            n_mels=self.n_mels,
            power=2,
            norm='slaney',
            mel_scale='slaney').to(self.device)
        self.mel_scaler = None

    def __len__(self):
        return len(self.df)

    def generate_partials(self, n_batches):
        pitches = torch.rand(n_batches, self.num_partials) * \
            self.pitch_range + self.min_pitch  # (B, num_partials)
        pitches = pitches.sort(dim=1).values  # (B, num_partials)
        diff_between_lowest = pitches[:, 1] - pitches[:, 0]  # (B,)
        # enforce minimum difference between lowest and second lowest pitch
        # shift the rest of the pitches up if the difference is less than self.min_pitchdiff_lowest
        to_add = self.min_pitchdiff_lowest - diff_between_lowest  # (B,)
        # don't add if the difference is already greater than self.min_pitchdiff_lowest
        to_add = torch.maximum(to_add, torch.zeros_like(to_add))
        # (B, num_partials-1)
        to_add = to_add.unsqueeze(-1).repeat(1, self.num_partials-1)
        pitches[:, 1:] += to_add
        return midi2frequency(pitches)

    def generate_batch(self, n_batches):
        assert self.mel_scaler is not None

        nsamps = int(self.dur * self.sr)
        synths = self.generate_partials(n_batches)  # (B, num_partials)
        # (B, num_partials, nsamps)
        synths = synths.unsqueeze(-1).repeat(1, 1, nsamps)

        # (B, num_partials, nsamps)
        synths = self.synth(synths.to(self.device))
        synths_positive = synths[:, :-1, :].sum(
            dim=1).div(self.num_partials)
        synths_negative = synths[:, 1:, :].sum(
            dim=1).div(self.num_partials)
        synths = synths.sum(dim=1).div(self.num_partials)

        synths = self.mel_spec(synths)  # (B, n_mels, T)
        synths_positive = self.mel_spec(synths_positive)  # (B, n_mels, T)
        synths_negative = self.mel_spec(synths_negative)  # (B, n_mels, T)

        synths = synths.mean(dim=-1, keepdim=True)  # (B, n_mels, 1)
        synths_positive = synths_positive.mean(dim=-1, keepdim=True)
        synths_negative = synths_negative.mean(dim=-1, keepdim=True)

        synths = amplitude_to_DB(
            synths, multiplier=10, amin=1e-5, db_multiplier=20, top_db=80)  # (B, n_mels, 1)
        synths_positive = amplitude_to_DB(
            synths_positive, multiplier=10, amin=1e-5, db_multiplier=20, top_db=80)
        synths_negative = amplitude_to_DB(
            synths_negative, multiplier=10, amin=1e-5, db_multiplier=20, top_db=80)

        synths = self.mel_scaler.transform(
            synths.squeeze(-1).cpu().numpy())  # (B, n_mels)
        synths_positive = self.mel_scaler.transform(
            synths_positive.squeeze(-1).cpu().numpy())
        synths_negative = self.mel_scaler.transform(
            synths_negative.squeeze(-1).cpu().numpy())

        synths = torch.tensor(
            synths, device=self.device).reshape(-1, 1, self.n_mels)  # (B, 1, n_mels)
        synths_positive = torch.tensor(
            synths_positive, device=self.device).reshape(-1, 1, self.n_mels)
        synths_negative = torch.tensor(
            synths_negative, device=self.device).reshape(-1, 1, self.n_mels)

        return synths, synths_positive, synths_negative


# %%


class PlMelEncoder(LightningModule):
    def __init__(self, args):
        super().__init__()
        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        # self.save_hyperparameters()

        # model params
        # ConvEncoder1D
        self.conv_in_channels = args.conv_in_channels
        self.conv_layers_channels = args.conv_layers_channels
        self.conv_in_size = args.conv_in_size
        self.conv_out_features = args.conv_out_features
        # LinearProjector
        self.proj_in_features = self.conv_out_features
        self.proj_out_features = args.out_features
        self.proj_hidden_layers_features = args.proj_hidden_layers_features

        # losses
        self.mse = nn.MSELoss()
        self.teacher_loss_weight = args.teacher_loss_weight
        self.triplet_loss_weight = args.triplet_loss_weight
        self.mel_triplet_loss_weight = args.mel_triplet_loss_weight
        self.pitch_loss_weight = args.pitch_loss_weight

        # learning rate
        self.lr = args.lr
        self.lr_decay = args.lr_decay
        self.decay = args.ema_decay

        # sampling rate
        self.sr = args.sr

        # logging
        self.plot_interval = args.plot_interval
        self.args = args

        # models
        conv = ConvEncoder1D(
            in_channels=self.conv_in_channels, output_size=self.conv_out_features, layers_channels=self.conv_layers_channels, input_size=self.conv_in_size)
        proj = LinearProjector(
            in_features=self.proj_in_features, out_features=self.proj_out_features, hidden_layers_features=self.proj_hidden_layers_features)
        self.model = nn.Sequential(conv, proj)

        # self.shadow = deepcopy(self.model)
        # for param in self.shadow.parameters():
        #     param.detach_()

        # mode: "supervised" or "contrastive"
        self.mode = args.mode

    def forward(self, x):
        if self.training or self.mode == "supervised":
            return self.model(x)
        else:
            return self.shadow(x)

    def create_shadow_model(self):
        self.shadow = deepcopy(self.model)
        for param in self.shadow.parameters():
            param.detach_()
        self.shadow.eval()

    # following this source: https://www.zijianhu.com/post/pytorch/ema/
    @torch.no_grad()
    def ema_update(self):
        if not self.training:
            print("EMA update should only be called during training",
                  file=stderr, flush=True)
            return

        model_params = OrderedDict(self.model.named_parameters())
        shadow_params = OrderedDict(self.shadow.named_parameters())

        # check if both model contains the same set of keys
        assert model_params.keys() == shadow_params.keys()

        for name, param in model_params.items():
            # see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
            # shadow_variable -= (1 - decay) * (shadow_variable - variable)
            shadow_params[name].sub_(
                (1. - self.decay) * (shadow_params[name] - param))

        model_buffers = OrderedDict(self.model.named_buffers())
        shadow_buffers = OrderedDict(self.shadow.named_buffers())

        # check if both model contains the same set of keys
        assert model_buffers.keys() == shadow_buffers.keys()

        for name, buffer in model_buffers.items():
            # buffers are copied
            shadow_buffers[name].copy_(buffer)

    def mine_triplets_based_on_mels(self, mels, embeddings, r=0.1):
        # get the distance matrix of the embeddings
        dist = torch.cdist(embeddings, embeddings)
        B = embeddings.shape[0]
        anchor_ids = []
        positive_ids = []
        negative_ids = []
        for i in range(B):
            p0_dist = dist[i]
            mask = p0_dist < r
            # get the indices of the embeddings that are within the radius
            indices = torch.nonzero(mask, as_tuple=True)[0]
            if len(indices) < 3:
                continue
            mels_within_radius = mels[indices]
            mels_within_radius = mels_within_radius.squeeze(1)
            dist_mels = torch.cdist(mels[i], mels_within_radius)
            sort_by_dist = torch.argsort(dist_mels.squeeze(0))
            id_of_closest, id_of_farthest = sort_by_dist[1], sort_by_dist[-1]
            id_of_closest, id_of_farthest = indices[id_of_closest], indices[id_of_farthest]
            anchor_ids.append(i)
            positive_ids.append(id_of_closest.item())
            negative_ids.append(id_of_farthest.item())
        return torch.tensor(anchor_ids), torch.tensor(positive_ids), torch.tensor(negative_ids)

    def mel_triplet_loss(self, mels, embeddings, r=0.1):
        anchor_ids, pos_ids, neg_ids = self.mine_triplets_based_on_mels(
            mels, embeddings, r)
        if len(anchor_ids) > 0:
            anchor, positive, negative = embeddings[anchor_ids], embeddings[pos_ids], embeddings[neg_ids]
            d_ap = torch.norm(anchor - positive, dim=1)
            d_an = torch.norm(anchor - negative, dim=1)
            margin = 0.05
            mel_triplet_loss = torch.nn.functional.relu(
                d_ap - d_an + margin).mean()
            return mel_triplet_loss

    def supervised_loss(self, batch):
        # get sample and label
        x, y = batch
        # predict label
        y_hat = self.model(x)
        # calculate loss
        loss = self.mse(y_hat, y)
        return loss

    def contrastive_loss(self, batch):
        # get sample and augmentation
        mels, waveforms = batch
        x, x_a = mels
        # x, x_a = batch
        y_student = self.model(x)
        y_student_a = self.model(x_a)

        # embeddings should be inveriant of augmentations
        loss_aug = self.mse(y_student, y_student_a)

        loss = loss_aug

        loss_teacher = 0
        if self.teacher_loss_weight > 0:
            y_teacher = self.shadow(x).detach()
            # embeddings should be similar to teacher
            loss_teacher = self.teacher_loss_weight * \
                self.mse(y_student, y_teacher)
            loss += loss_teacher

        loss_pitch = 0
        if self.pitch_loss_weight > 0:
            # finally, the distances between the embeddings should be proportional to detected pitch distances (MIDI)
            pitch = torchyin.estimate(waveforms, self.sr)  # (B, T)
            pitch_median = torch.median(pitch, dim=-1).values  # (B,)
            # clamp to minimum 20 hz
            pitch_median = torch.clamp(pitch_median, 20)
            pitch_midi = frequency2midi(pitch_median)  # (B,)
            pitch_dist = torch.abs(pitch_midi.unsqueeze(-1) -
                                   pitch_midi.unsqueeze(-2))  # (B, B)
            pitch_dist = pitch_dist / pitch_dist.max()  # normalize
            embeddings_dist = torch.cdist(y_student, y_student)
            embeddings_dist = embeddings_dist / embeddings_dist.max()
            loss_pitch = self.pitch_loss_weight * \
                self.mse(embeddings_dist, pitch_dist)
        loss += loss_pitch

        triplet_loss = 0
        if self.contrastive_sinusoidal_dataset is not None and self.triplet_loss_weight > 0 and self.training:
            anchor, positive, negative = self.contrastive_sinusoidal_dataset.generate_batch(
                x.shape[0])
            anchor = self.model(anchor)
            positive = self.model(positive)
            negative = self.model(negative)
            d_ap = torch.norm(anchor - positive, dim=1)
            d_an = torch.norm(anchor - negative, dim=1)
            margin = 0.05
            triplet_loss = torch.relu(
                d_ap - d_an + margin).mean() * self.triplet_loss_weight
        loss += triplet_loss

        mel_triplet_loss = 0
        if self.mel_triplet_loss_weight > 0:
            mel_triplet_loss = self.mel_triplet_loss(
                x, y_student, r=0.1) * self.mel_triplet_loss_weight
        loss += mel_triplet_loss

        if self.training:
            self.log_dict({
                "loss_aug": loss_aug,
                "loss_teacher": loss_teacher / self.teacher_loss_weight if self.teacher_loss_weight > 0 else 0,
                "loss_triplet": triplet_loss / self.triplet_loss_weight if self.triplet_loss_weight > 0 else 0,
                "loss_mel_triplet": mel_triplet_loss / self.mel_triplet_loss_weight if self.mel_triplet_loss_weight > 0 else 0,
                "loss_pitch": loss_pitch / self.pitch_loss_weight if self.pitch_loss_weight > 0 else 0,
            })
        else:
            self.log_dict({
                "val_loss_aug": loss_aug,
                "val_loss_teacher": loss_teacher / self.teacher_loss_weight if self.teacher_loss_weight > 0 else 0,
                "val_loss_mel_triplet": mel_triplet_loss / self.mel_triplet_loss_weight if self.mel_triplet_loss_weight > 0 else 0,
                "val_loss_pitch": loss_pitch / self.pitch_loss_weight if self.pitch_loss_weight > 0 else 0,
            })
        return loss

    def training_step(self, batch, batch_idx):
        self.model.train()
        # epoch_idx = self.trainer.current_epoch

        # get the optimizer and scheduler
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()

        # in supervised mode, just map to the labels
        if self.mode == "supervised":
            loss = self.supervised_loss(batch)
        # in contrastive mode:
        elif self.mode == "contrastive":
            loss = self.contrastive_loss(batch)

        # backward pass
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        scheduler.step()

        # ema update
        if self.mode == "contrastive":
            self.ema_update()

        # log losses
        self.log_dict({
            "train_loss": loss,
        })

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        # epoch_idx = self.trainer.current_epoch

        # in supervised mode, just map to the labels
        if self.mode == "supervised":
            loss = self.supervised_loss(batch)
        # in contrastive mode:
        elif self.mode == "contrastive":
            loss = self.contrastive_loss(batch)

        # log losses
        self.log_dict({
            "val_loss": loss,
        })

    def on_validation_epoch_end(self) -> None:
        self.model.eval()
        # get the epoch number from trainer
        epoch = self.trainer.current_epoch

        if epoch % self.plot_interval != 0 and epoch != 0:
            return

        self.save_latent_space_plot()

    def save_latent_space_plot(self, batch_size=1024):
        """Save a figure of the latent space"""
        # get the length of the training dataset
        dataset = self.trainer.val_dataloaders.dataset
        batchsampler = torch.utils.data.BatchSampler(
            range(len(dataset)), batch_size=batch_size, drop_last=False)
        loader = DataLoader(dataset, batch_size=None, sampler=batchsampler)
        # loader = self.supervised_val_loader
        z_all = torch.zeros(
            len(dataset), self.proj_out_features).to(self.device)
        for batch_idx, data in enumerate(loader):
            if self.mode == "supervised":
                x, _ = data
            elif self.mode == "contrastive":
                mels, _ = data
                x, _ = mels
            # x, _ = data
            z = self.model(x.to(self.device))
            z = z.detach()
            z_all[batch_idx*batch_size: batch_idx *
                  batch_size + batch_size] = z
        z_all = z_all.cpu().numpy()
        # create the figure
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        ax.scatter(z_all[:, 0], z_all[:, 1])
        ax.set_title(
            f"Latent space at epoch {self.trainer.current_epoch}")
        # save figure to checkpoint folder/latent
        save_dir = os.path.join(self.args.ckpt_path,
                                self.args.ckpt_name, "latent")
        os.makedirs(save_dir, exist_ok=True)
        fig_name = f"latent_{str(self.trainer.current_epoch).zfill(5)}.png"
        plt.savefig(os.path.join(save_dir, fig_name))
        plt.close(fig)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.lr_decay)
        return [optimizer], [scheduler]


# %%
csv_abspath = "./experiments/pca_finetuning/fm_synth_params.csv"
pca_abspath = "./experiments/pca_finetuning/pca_mels_mean.json"


def main():
    # read fm synth dataframe from csv
    df_fm = pd.read_csv(csv_abspath, index_col=0)
    # read pca array from json
    pca_array = fluid_dataset2array(json.load(open(pca_abspath, "r")))
    # total number of samples
    n_samples = len(df_fm)
    # n_samples = 10000  # for testing
    # generate dataset splits on indices
    indices = torch.arange(len(df_fm))
    # shuffle
    # indices = torch.randperm(len(df_fm))
    # indices = indices[:n_samples]
    # splits = torch.utils.data.random_split(list(range(n_samples)), [0.8, 0.2])
    # splits = torch.utils.data.random_split(list(indices), [0.8, 0.2])
    # train_split, val_split = splits
    # save splits
    train_split_path = os.path.join(
        "./experiments/pca_finetuning", "train_split.pt")
    val_split_path = os.path.join(
        "./experiments/pca_finetuning", "val_split.pt")
    # torch.save(train_split, train_split_path)
    # torch.save(val_split, val_split_path)
    train_split = torch.load(train_split_path)
    val_split = torch.load(val_split_path)

    # create train and val datasets for fm params and pca
    df_fm_train, df_fm_val = df_fm.loc[list(
        train_split)], df_fm.loc[list(val_split)]
    pca_train, pca_val = pca_array[list(
        train_split)], pca_array[list(val_split)]

    # number of mel bands
    n_mels = 400

    # create train and val datasets for mel spectrograms and pca
    fm_mel_pca_train = FmMel2PCADataset(
        df_fm_train, pca_train, sr=48000, dur=1, n_mels=n_mels)
    fm_mel_pca_val = FmMel2PCADataset(
        df_fm_val, pca_val, sr=48000, dur=1, n_mels=n_mels)

    # fit scalers in the train dataset, pass fit scalers to val dataset
    fm_mel_pca_train.fit_scalers()
    fm_mel_pca_val.mel_scaler = fm_mel_pca_train.mel_scaler
    fm_mel_pca_val.pca_scaler = fm_mel_pca_train.pca_scaler

    # save scalers
    mel_scaler_path = os.path.join(
        "./experiments/pca_finetuning", "mel_scaler.pkl")
    pca_scaler_path = os.path.join(
        "./experiments/pca_finetuning", "pca_scaler.pkl")
    torch.save(fm_mel_pca_train.mel_scaler, mel_scaler_path)
    torch.save(fm_mel_pca_train.pca_scaler, pca_scaler_path)

    # create contrastive datasets based on the same splits
    fm_contrastive_train = FmMelContrastiveDataset(
        df_fm_train, sr=48000, dur=1, n_mels=n_mels, mel_scaler=fm_mel_pca_train.mel_scaler)
    fm_contrastive_val = FmMelContrastiveDataset(
        df_fm_val, sr=48000, dur=1, n_mels=n_mels, mel_scaler=fm_mel_pca_train.mel_scaler)  # use train scaler

    # create samplers & dataloaders
    batch_size = 512
    # num_workers = 0

    # supervised train
    fm_mel_pca_train_randomsampler = torch.utils.data.RandomSampler(
        fm_mel_pca_train, replacement=False)
    fm_mel_pca_train_batchsampler = torch.utils.data.BatchSampler(
        fm_mel_pca_train_randomsampler, batch_size=batch_size, drop_last=True)
    supervised_train_loader = DataLoader(
        fm_mel_pca_train, batch_size=None, sampler=fm_mel_pca_train_batchsampler)

    # supervised val
    # fm_mel_pca_val_randomsampler = torch.utils.data.RandomSampler(
    #     fm_mel_pca_val, replacement=False)
    fm_mel_pca_val_batchsampler = torch.utils.data.BatchSampler(
        range(len(fm_mel_pca_val)), batch_size=batch_size, drop_last=False)
    supervised_val_loader = DataLoader(
        fm_mel_pca_val, batch_size=None, sampler=fm_mel_pca_val_batchsampler)

    # contrastive train
    fm_contrastive_train_randomsampler = torch.utils.data.RandomSampler(
        fm_contrastive_train, replacement=False)
    fm_contrastive_train_batchsampler = torch.utils.data.BatchSampler(
        fm_contrastive_train_randomsampler, batch_size=batch_size, drop_last=True)
    contrastive_train_loader = DataLoader(
        fm_contrastive_train, batch_size=None, sampler=fm_contrastive_train_batchsampler)

    # contrastive val
    # fm_contrastive_val_randomsampler = torch.utils.data.RandomSampler(
    #     fm_contrastive_val, replacement=False)
    fm_contrastive_val_batchsampler = torch.utils.data.BatchSampler(
        range(len(fm_contrastive_val)), batch_size=batch_size, drop_last=False)
    contrastive_val_loader = DataLoader(
        fm_contrastive_val, batch_size=None, sampler=fm_contrastive_val_batchsampler)

    args = Args()
    args.sr = 48000
    args.conv_in_channels = 1
    args.conv_layers_channels = [64, 128, 256]
    args.conv_in_size = n_mels
    args.conv_out_features = 256
    args.out_features = 2
    args.proj_hidden_layers_features = [256, 256, 128]
    args.lr = 1e-4
    args.lr_decay = 0.999
    args.ema_decay = 0.999
    args.teacher_loss_weight = 0.0
    args.triplet_loss_weight = 0.0
    args.mel_triplet_loss_weight = 1
    args.pitch_loss_weight = 0.0
    args.plot_interval = 1
    args.mode = "contrastive"
    args.ckpt_path = "ckpt"
    args.ckpt_name = "pca_finetuning_only_contrastive_9"
    args.resume_ckpt_path = None
    args.logdir = "logs"
    # supervised_epochs = 11
    # args.train_epochs = supervised_epochs
    args.train_epochs = 1000001
    args.comment = "new mel triplet loss with r=0.1"

    # create model
    model = PlMelEncoder(args)
    model.contrastive_sinusoidal_dataset = ContrastiveSinusoidalDataset(
        num_partials=11,
        min_pitch=45,
        pitch_range=96,
        min_pitchdiff_lowest=7,
        sr=48000,
        dur=0.5,
        n_mels=n_mels)
    model.contrastive_sinusoidal_dataset.mel_scaler = fm_mel_pca_train.mel_scaler
    # show model summary via torch summary
    # summary(model, (1, 200))

    # checkpoint callbacks
    checkpoint_path = os.path.join(args.ckpt_path, args.ckpt_name)
    best_checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=checkpoint_path,
        filename=args.ckpt_name + "_val_{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        mode="min",
    )
    last_checkpoint_callback = ModelCheckpoint(
        monitor="epoch",
        dirpath=checkpoint_path,
        filename=args.ckpt_name + "_last_{epoch:02d}",
        save_top_k=1,
        mode="max",
    )

    # logger callbacks
    tensorboard_logger = TensorBoardLogger(
        save_dir=args.logdir, name=args.ckpt_name)

    # create trainer
    trainer = Trainer(
        max_epochs=args.train_epochs,
        enable_checkpointing=True,
        callbacks=[best_checkpoint_callback, last_checkpoint_callback],
        logger=tensorboard_logger,
        log_every_n_steps=1,
    )

    # save hyperparameters
    hyperparams = dict(
        sr=args.sr,
        conv_in_channels=args.conv_in_channels,
        conv_layers_channels=args.conv_layers_channels,
        conv_in_size=args.conv_in_size,
        conv_out_features=args.conv_out_features,
        out_features=args.out_features,
        proj_hidden_layers_features=args.proj_hidden_layers_features,
        mode=args.mode,
        lr=args.lr,
        lr_decay=args.lr_decay,
        ema_decay=args.ema_decay,
        teacher_loss_weight=args.teacher_loss_weight,
        triplet_loss_weight=args.triplet_loss_weight,
        mel_triplet_loss_weight=args.mel_triplet_loss_weight,
        pitch_loss_weight=args.pitch_loss_weight,
        plot_interval=args.plot_interval,
        ckpt_path=args.ckpt_path,
        ckpt_name=args.ckpt_name,
        resume_ckpt_path=args.resume_ckpt_path,
        logdir=args.logdir,
        train_epochs=args.train_epochs,
        comment=args.comment,
    )
    trainer.logger.log_hyperparams(hyperparams)

    # # train the model in supervised mode
    # model.mode = "supervised"
    # model.supervised_val_loader = supervised_val_loader
    # print("Training in supervised mode")
    # trainer.fit(model, supervised_train_loader, supervised_val_loader,
    #             ckpt_path=args.resume_ckpt_path)

    # # checkpoint callbacks
    # checkpoint_path = os.path.join(args.ckpt_path, args.ckpt_name)
    # best_checkpoint_callback = ModelCheckpoint(
    #     monitor="val_loss",
    #     dirpath=checkpoint_path,
    #     filename=args.ckpt_name + "_val_{epoch:02d}-{val_loss:.4f}",
    #     save_top_k=1,
    #     mode="min",
    # )
    # last_checkpoint_callback = ModelCheckpoint(
    #     monitor="epoch",
    #     dirpath=checkpoint_path,
    #     filename=args.ckpt_name + "_last_{epoch:02d}",
    #     save_top_k=1,
    #     mode="max",
    # )

    # # logger callbacks
    # tensorboard_logger = TensorBoardLogger(
    #     save_dir=args.logdir, name=args.ckpt_name)

    # # create trainer
    # args.train_epochs = 1000001
    # trainer_contrastive = Trainer(
    #     max_epochs=args.train_epochs,
    #     enable_checkpointing=True,
    #     callbacks=[best_checkpoint_callback, last_checkpoint_callback],
    #     logger=tensorboard_logger,
    #     log_every_n_steps=1,
    # )

    # train the model in contrastive mode
    model.mode = "contrastive"
    # args.resume_ckpt_path = f"{args.ckpt_path}/{args.ckpt_name}/{args.ckpt_name}_last_epoch={str(supervised_epochs - 1).zfill(2)}.ckpt"
    # resume_ckpt = torch.load(args.resume_ckpt_path, map_location=model.device)
    # model.load_state_dict(resume_ckpt['state_dict'])
    model.create_shadow_model()
    print("Training in contrastive mode")
    # resume_ckpt_path = "ckpt/pca_finetuning_dino_1/pca_finetuning_dino_1_last_epoch=76.ckpt"
    resume_ckpt_path = None
    # trainer_contrastive.fit(
    #     model, contrastive_train_loader, contrastive_val_loader, ckpt_path=resume_ckpt_path)
    trainer.fit(
        model, contrastive_train_loader, contrastive_val_loader, ckpt_path=resume_ckpt_path)


if __name__ == "__main__":
    main()
