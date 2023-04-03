# %%
# imports

import numpy as np
import shutil
import platform
from pathlib import Path
from scipy.io import wavfile as wav
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils import *
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# %%

# dataset parameters

dataset_folder = "/Volumes/T7/synth_dataset"
# if on Windows, use this path
if platform.system() == "Windows":
    dataset_folder = "D:/synth_dataset"

num_samples = 100000
num_params = 3
sr = 48000

# %%

# load datasets

# read unscaled parameter values
params_unscaled = np.load(os.path.join(
    os.path.dirname(dataset_folder), "params_unscaled.npy"))
# read scaled parameter values
params_scaled = np.load(os.path.join(
    os.path.dirname(dataset_folder), "params_scaled.npy"))
# read spectral shape values
spectral_shape = np.load(os.path.join(
    os.path.dirname(dataset_folder), "spectralshape.npy"))

# %%

# standardize datasets

# standardize unscaled parameters
params_unscaled_stdscaler = StandardScaler().fit(params_unscaled)
params_unscaled_std = params_unscaled_stdscaler.transform(params_unscaled)
# standardize scaled parameters
params_scaled_stdscaler = StandardScaler().fit(params_scaled)
params_scaled_std = params_scaled_stdscaler.transform(params_scaled)
# standardize spectral shape
spectral_shape_stdscaler = StandardScaler().fit(spectral_shape.reshape(-1, 49))
spectral_shape_std = spectral_shape_stdscaler.transform(
    spectral_shape.reshape(-1, 49))

# %%

# minmax scale datasets

# minmax scale unscaled parameters
params_unscaled_mmscaler = MinMaxScaler().fit(params_unscaled)
params_unscaled_mm = params_unscaled_mmscaler.transform(params_unscaled)
# minmax scale scaled parameters
params_scaled_mmscaler = MinMaxScaler().fit(params_scaled)
params_scaled_mm = params_scaled_mmscaler.transform(params_scaled)
# minmax scale spectral shape
spectral_shape_mmscaler = MinMaxScaler().fit(spectral_shape.reshape(-1, 49))
spectral_shape_mm = spectral_shape_mmscaler.transform(
    spectral_shape.reshape(-1, 49))

# %%

# device for training

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# %%

# define model


class FMNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(49, 512)
        self.act1 = nn.Tanh()
        self.fc2 = nn.Linear(512, 512)
        self.act2 = nn.Tanh()
        self.fc3 = nn.Linear(512, 512)
        self.act3 = nn.Tanh()
        self.fc4 = nn.Linear(512, 512)
        self.act4 = nn.Tanh()
        self.fc5 = nn.Linear(512, 3)
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

learning_rate = 1e-4
batch_size = 4096 * 4


# %%

# define loss function

loss_fn = nn.MSELoss()

# %%

# define optimizer

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# %%

# define dataloader

spectral_shape_std_tensor = torch.from_numpy(spectral_shape_std).float().to(device)
params_scaled_std_tensor = torch.from_numpy(params_scaled_std).float().to(device)
# spectral_shape_mm_tensor = torch.from_numpy(spectral_shape_mm).float()
# params_scaled_mm_tensor = torch.from_numpy(params_scaled_mm).float()

train_ds = torch.utils.data.TensorDataset(
    spectral_shape_std_tensor, params_scaled_std_tensor)

# train_ds = torch.utils.data.TensorDataset(
#     spectral_shape_mm_tensor, params_scaled_mm_tensor)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size)


# %%

def train_loop(dataloader, model, loss_fn, optimizer, epoch):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Clear gradient buffers
        optimizer.zero_grad()

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()

        if batch % 1 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(
                f"Epoch {epoch+1} loss: {loss:>7f}  [{current:>5d}/{size:>5d}]", end="\r")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# %%

# train model


epochs = 5000

for t in range(epochs):
    train_loop(train_loader, model, loss_fn, optimizer, t)
    # test_loop(train_loader, model, loss_fn)


# %%

# predict synth parameters using stdscalers

test_spectral_shape_str = "1316.668579 862.922119 0.734027 3.301776 2892.480225 -98.244965 38.244732 113.891312 243.333267 0.189771 0.89706 730.881226 7.06159 0.678007 15.95755 14.425442 15.109092 17.064833 15.920691 9.464821 -13.660709 269.082428 215.657455 239.821274 306.043915 267.761963 115.092857 188.519485 1306.813721 845.161377 0.715383 3.222845 2842.234131 -104.676048 28.418276 1308.366577 845.501831 0.721522 3.236959 2842.436523 -99.115463 38.285007 3328.173828 4798.616699 3.951998 19.774744 15779.915039 -9.827665 38.340775"
test_spectral_shape_list = test_spectral_shape_str.split(" ")
test_spectral_shape = np.array(test_spectral_shape_list, dtype=np.float32)
test_spectral_shape = test_spectral_shape.reshape(1, -1)
test_spectral_shape_std = spectral_shape_stdscaler.transform(
    test_spectral_shape)
test_spectral_shape_std_tensor = torch.from_numpy(
    test_spectral_shape_std).float()
test_spectral_shape_std_tensor = test_spectral_shape_std_tensor.to(device)
test_pred = model(test_spectral_shape_std_tensor)
test_pred = test_pred.cpu().detach().numpy()
test_pred = params_scaled_stdscaler.inverse_transform(test_pred)
print(test_pred)

# %%

# predict synth parameters using mmscalers

test_spectral_shape_str = "1316.668579 862.922119 0.734027 3.301776 2892.480225 -98.244965 38.244732 113.891312 243.333267 0.189771 0.89706 730.881226 7.06159 0.678007 15.95755 14.425442 15.109092 17.064833 15.920691 9.464821 -13.660709 269.082428 215.657455 239.821274 306.043915 267.761963 115.092857 188.519485 1306.813721 845.161377 0.715383 3.222845 2842.234131 -104.676048 28.418276 1308.366577 845.501831 0.721522 3.236959 2842.436523 -99.115463 38.285007 3328.173828 4798.616699 3.951998 19.774744 15779.915039 -9.827665 38.340775"
test_spectral_shape_list = test_spectral_shape_str.split(" ")
test_spectral_shape = np.array(test_spectral_shape_list, dtype=np.float32)
test_spectral_shape = test_spectral_shape.reshape(1, -1)
test_spectral_shape_mm = spectral_shape_mmscaler.transform(test_spectral_shape)
test_spectral_shape_mm_tensor = torch.from_numpy(
    test_spectral_shape_mm).float()
test_spectral_shape_mm_tensor = test_spectral_shape_mm_tensor.to(device)
test_pred = model(test_spectral_shape_mm_tensor)
test_pred = test_pred.cpu().detach().numpy()
test_pred = params_scaled_mmscaler.inverse_transform(test_pred)
print(test_pred)

# %%
