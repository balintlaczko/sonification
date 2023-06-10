# imports
import argparse

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchsynth.config import SynthConfig

from utils import *

from fm_inversynth import FmSynth, FM_Autoencoder, FM_Param_Autoencoder

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def train(train_loader, model, optimizer, synth, epoch, device):
    # set model to train mode
    model.train()

    # create progress bar from dataloader
    train_loader = tqdm(train_loader)

    # create loss function
    criterion = nn.MSELoss()

    # initialize loss variables for the epoch
    mse_sum = 0
    mse_n = 0

    for batch_idx, data in enumerate(train_loader):
        # zero out gradients
        model.zero_grad()

        # move data to device
        data = data.to(device)

        # create mel spectrogram
        # carr_freq = data[:, 0]
        # mod_freq = data[:, 1]
        # mod_idx = data[:, 2]
        # mel_spec = synth(carr_freq, mod_freq, mod_idx)

        # forward pass
        # mel_spec_recon, params = model(mel_spec)
        params_recon = model(data)

        # calculate loss
        # recon_loss = criterion(mel_spec_recon, mel_spec)
        # param_loss = criterion(params, data)
        # loss = 0.0 * recon_loss + 1.0 * param_loss
        loss = criterion(params_recon, data)

        # backpropagate
        loss.backward()
        optimizer.step()

        # update loss variables
        mse_sum += loss.item() * data.shape[0]
        mse_n += data.shape[0]

        # get the learning rate from the optimizer
        lr = optimizer.param_groups[0]['lr']

        # update progress bar
        train_loader.set_description(
            f"Epoch {epoch + 1}/{args.num_epochs} | LR: {lr:.6f} | Loss: {mse_sum / mse_n:.4f}")

    # return average loss for the epoch
    return mse_sum / mse_n


def main(args):
    # set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # load parameter values from npy file
    train_dataset = np.load(args.params_dataset)
    print(
        f"Loaded dataset with {len(train_dataset)} samples and {len(train_dataset[0])} features")

    # convert to tensor
    train_dataset = torch.from_numpy(train_dataset).float().to(device)

    # create dataloader
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # create synth config
    config = SynthConfig(
        batch_size=args.batch_size,
        sample_rate=args.sample_rate,
        buffer_size_seconds=args.buffer_length_s,
        reproducible=False
    )

    # create model and optimizer
    # model = FM_Autoencoder(config, device).to(device)
    model = FM_Param_Autoencoder(config, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    print("Model and optimizer created")

    # create synth
    synth = FmSynth(config, device).to(device)

    # create summary writer
    writer = SummaryWriter(args.log_folder)
    # create log folder
    os.makedirs(args.log_folder, exist_ok=True)
    # create checkpoint folder
    os.makedirs(args.ckpt_folder, exist_ok=True)

    # train model
    for epoch in range(args.num_epochs):
        recon_loss = train(train_loader, model, optimizer,
                           synth, epoch, device)

        # log loss
        writer.add_scalar("Loss/train", recon_loss, epoch)

        # save model at checkpoint interval
        if (epoch + 1) % args.ckpt_interval == 0:
            torch.save(model.state_dict(), os.path.join(
                args.ckpt_folder, f"model_{str(epoch + 1).zfill(4)}.pt"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--params_dataset", type=str,
                        default="/Volumes/T7/synth_dataset_fm_inversynth/params_scaled.npy")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sample_rate", type=int, default=48000)
    parser.add_argument("--buffer_length_s", type=float, default=2.0)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--ckpt_folder", type=str,
                        default="/Volumes/T7/synth_dataset_fm_inversynth/ckpt")
    parser.add_argument("--ckpt_interval", type=int,
                        default=1)
    parser.add_argument("--log_folder", type=str,
                        default="/Volumes/T7/synth_dataset_fm_inversynth/logs")

    args = parser.parse_args()

    main(args)
