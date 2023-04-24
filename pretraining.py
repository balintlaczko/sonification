# %%
# imports

import numpy as np
import shutil
import platform
from pathlib import Path
from scipy.io import wavfile as wav
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from utils import *
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

# %%

# dataset parameters

dataset_folder = "/Volumes/T7/synth_dataset_2"
# if on Windows, use this path
if platform.system() == "Windows":
    dataset_folder = "D:/synth_dataset_2"

num_samples = 1000000
num_params = 3
sr = 48000

# %%

# load datasets

# read unscaled parameter values
params_unscaled = np.load(os.path.join(dataset_folder, "params_unscaled.npy"))
# read scaled parameter values
params_scaled = np.load(os.path.join(dataset_folder, "params_scaled.npy"))
# read spectral shape values
melspec = np.load(os.path.join(dataset_folder, "melspec.npy"))

# %%

# standardize datasets

# standardize unscaled parameters
params_unscaled_stdscaler = StandardScaler().fit(params_unscaled)
params_unscaled_std = params_unscaled_stdscaler.transform(params_unscaled)
# standardize scaled parameters
params_scaled_stdscaler = StandardScaler().fit(params_scaled)
params_scaled_std = params_scaled_stdscaler.transform(params_scaled)
# standardize spectral shape
melspec_stdscaler = StandardScaler().fit(melspec.reshape(-1, 200))
melspec_std = melspec_stdscaler.transform(
    melspec.reshape(-1, 200))

# %%

# minmax scale datasets

# minmax scale unscaled parameters
params_unscaled_mmscaler = MinMaxScaler().fit(params_unscaled)
params_unscaled_mm = params_unscaled_mmscaler.transform(params_unscaled)
# minmax scale scaled parameters
params_scaled_mmscaler = MinMaxScaler().fit(params_scaled)
params_scaled_mm = params_scaled_mmscaler.transform(params_scaled)
# minmax scale spectral shape
melspec_mmscaler = MinMaxScaler().fit(melspec.reshape(-1, 200))
melspec_mm = melspec_mmscaler.transform(
    melspec.reshape(-1, 200))

# %%

# device for training

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# %%

# define model


class FMNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(200, 2048)
        self.act1 = nn.Tanh()
        self.fc2 = nn.Linear(2048, 4096)
        self.act2 = nn.Tanh()
        self.fc3 = nn.Linear(4096, 4096)
        self.act3 = nn.Tanh()
        self.fc4 = nn.Linear(4096, 2048)
        self.act4 = nn.Tanh()
        self.fc5 = nn.Linear(2048, 3)
        self.act_out = nn.Tanh()

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        x = self.act4(self.fc4(x))
        x = self.act_out(self.fc5(x))
        return x


# %%
model = FMNet().to(device)
print(model)

# %%

# initialize hyperparameters

learning_rate = 1e-3
batch_size = 4096 * 16


# %%

# define loss function

loss_fn = nn.MSELoss()

# %%

# define optimizer

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# %%

# train test split

melspec_std_train, melspec_std_test, params_scaled_std_train, params_scaled_std_test = train_test_split(
    melspec_std, params_scaled_std, test_size=0.2, random_state=42)

# %%

# define dataloaders

# train loader
melspec_std_train_tensor = torch.from_numpy(
    melspec_std_train).float().to(device)
params_scaled_std_train_tensor = torch.from_numpy(
    params_scaled_std_train).float().to(device)
train_ds = torch.utils.data.TensorDataset(
    melspec_std_train_tensor, params_scaled_std_train_tensor)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size)

# test loader
melspec_std_test_tensor = torch.from_numpy(melspec_std_test).float().to(device)
params_scaled_std_test_tensor = torch.from_numpy(
    params_scaled_std_test).float().to(device)
test_ds = torch.utils.data.TensorDataset(
    melspec_std_test_tensor, params_scaled_std_test_tensor)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)


# %%

def train_loop(dataloader, model, loss_fn, optimizer, epoch, writer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    batch_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        # Clear gradient buffers
        optimizer.zero_grad()

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        batch_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

        if batch % 1 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(
                f"\rEpoch {epoch+1} loss: {loss:>7f}  [{current:>5d}/{size:>5d}]", end=" ")

    # Log to tensorboard
    epoch_loss = batch_loss / num_batches
    writer.add_scalar("Loss/train", epoch_loss, epoch)


def test_loop(dataloader, model, loss_fn, epoch, writer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches

    print(
        f"Test avg loss: {test_loss:>8f}")
    
    # Log to tensorboard
    writer.add_scalar("Loss/test", test_loss, epoch)


# %%

# set up tensorboard

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/fmnet')


# %%

# train model

global_counter = 0
epochs = 100

for t in range(epochs):
    train_loop(train_loader, model, loss_fn, optimizer, global_counter+t, writer)
    test_loop(test_loader, model, loss_fn, global_counter+t, writer)


# %%

# flush tensorboard writer
writer.flush()
# close tensorboard writer
writer.close()


# %%

# predict synth parameters using stdscalers

test_melspec_str = "1316.668579 862.922119 0.734027 3.301776 2892.480225 -98.244965 38.244732 113.891312 243.333267 0.189771 0.89706 730.881226 7.06159 0.678007 15.95755 14.425442 15.109092 17.064833 15.920691 9.464821 -13.660709 269.082428 215.657455 239.821274 306.043915 267.761963 115.092857 188.519485 1306.813721 845.161377 0.715383 3.222845 2842.234131 -104.676048 28.418276 1308.366577 845.501831 0.721522 3.236959 2842.436523 -99.115463 38.285007 3328.173828 4798.616699 3.951998 19.774744 15779.915039 -9.827665 38.340775"
test_melspec_list = test_melspec_str.split(" ")
test_melspec = np.array(test_melspec_list, dtype=np.float32)
test_melspec = test_melspec.reshape(1, -1)
test_melspec_std = melspec_stdscaler.transform(
    test_melspec)
test_melspec_std_tensor = torch.from_numpy(
    test_melspec_std).float()
test_melspec_std_tensor = test_melspec_std_tensor.to(device)
test_pred = model(test_melspec_std_tensor)
test_pred = test_pred.cpu().detach().numpy()
test_pred = params_scaled_stdscaler.inverse_transform(test_pred)
print(test_pred)

# %%

# predict synth parameters using mmscalers

test_melspec_str = "1316.668579 862.922119 0.734027 3.301776 2892.480225 -98.244965 38.244732 113.891312 243.333267 0.189771 0.89706 730.881226 7.06159 0.678007 15.95755 14.425442 15.109092 17.064833 15.920691 9.464821 -13.660709 269.082428 215.657455 239.821274 306.043915 267.761963 115.092857 188.519485 1306.813721 845.161377 0.715383 3.222845 2842.234131 -104.676048 28.418276 1308.366577 845.501831 0.721522 3.236959 2842.436523 -99.115463 38.285007 3328.173828 4798.616699 3.951998 19.774744 15779.915039 -9.827665 38.340775"
test_melspec_list = test_melspec_str.split(" ")
test_melspec = np.array(test_melspec_list, dtype=np.float32)
test_melspec = test_melspec.reshape(1, -1)
test_melspec_mm = melspec_mmscaler.transform(test_melspec)
test_melspec_mm_tensor = torch.from_numpy(
    test_melspec_mm).float()
test_melspec_mm_tensor = test_melspec_mm_tensor.to(device)
test_pred = model(test_melspec_mm_tensor)
test_pred = test_pred.cpu().detach().numpy()
test_pred = params_scaled_mmscaler.inverse_transform(test_pred)
print(test_pred)

# %%
