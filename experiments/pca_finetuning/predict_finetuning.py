# imports
# from torchsummary import summary
from lightning.pytorch import LightningModule
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from sonification.models.ddsp import FMSynth
from sonification.models.layers import ConvEncoder1D, LinearProjector
from sonification.models.loss import MMDloss
from librosa import resample
# from sonification.datasets import FmSynthDataset
from sonification.utils.dsp import transposition2duration
from torchaudio.functional import amplitude_to_DB
from torchaudio.transforms import MelSpectrogram
from sonification.utils.array import fluid_dataset2array
from sklearn.preprocessing import MinMaxScaler
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
from workbench import Args, FmSynthDataset, FmMel2PCADataset, FmMelContrastiveDataset, PlMelEncoder

args = Args()
args.conv_in_channels = 1
args.conv_layers_channels = [64, 128, 256]
args.conv_in_size = 200
args.conv_out_features = 128
args.out_features = 2
args.proj_hidden_layers_features = [256, 256, 128, 64]
args.contrastive_diff_loss_scaler = 0.25
args.lr = 1e-3
args.lr_decay = 0.9999
args.ema_decay = 0.999
args.plot_interval = 1
args.mode = "contrastive"
args.ckpt_path = "ckpt"
args.ckpt_name = "pca_finetuning_ema_6"
args.last_epoch = 100
args.resume_ckpt_path = f"{args.ckpt_path}/{args.ckpt_name}/{args.ckpt_name}_last_epoch={str(args.last_epoch).zfill(2)}.ckpt"
args.logdir = "logs"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PlMelEncoder(args).to(device)
model.create_shadow_model()
resume_ckpt = torch.load(args.resume_ckpt_path, map_location=model.device)
model.load_state_dict(resume_ckpt['state_dict'])
model.eval()

csv_abspath = "./experiments/pca_finetuning/fm_synth_params.csv"
pca_abspath = "./experiments/pca_finetuning/pca_mels_mean.json"
train_split_path = "./experiments/pca_finetuning/train_split.pt"
train_split = torch.load(train_split_path)
df_fm = pd.read_csv(csv_abspath, index_col=0)
df_fm_train = df_fm.loc[list(train_split)]
pca_array = fluid_dataset2array(json.load(open(pca_abspath, "r")))
pca_train = pca_array[list(train_split)]
dataset_train = FmMel2PCADataset(
    df_fm_train, pca_train, sr=48000, dur=1, n_mels=200)
dataset_train.fit_scalers()
dataset_full = FmMel2PCADataset(df_fm, pca_array, sr=48000, dur=1, n_mels=200)
dataset_full.mel_scaler = dataset_train.mel_scaler
dataset_full.pca_scaler = dataset_train.pca_scaler
batch_size = 1024
dataset_full_sampler = torch.utils.data.BatchSampler(
    range(len(dataset_full)), batch_size=batch_size, drop_last=False)
dataset_full_loader = DataLoader(
    dataset_full, batch_size=None, sampler=dataset_full_sampler)
