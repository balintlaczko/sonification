# imports
import argparse

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchsynth.config import SynthConfig

import auraloss

from utils import *

from fm_inversynth import FmSynth, FmSynth2Wave, FM_Autoencoder, FM_Param_Autoencoder, FM_Autoencoder_Wave, FM_Autoencoder_Wave2

from tqdm import tqdm
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter


def train(train_loader, model, optimizer, loss_fn, synth, epoch, device):
    # set model to train mode
    model.train()

    # create progress bar from dataloader
    train_loader = tqdm(train_loader)

    # create loss function
    # criterion = nn.MSELoss()
    criterion = loss_fn

    # initialize loss variables for the epoch
    mse_sum = 0
    mse_n = 0

    for batch_idx, data in enumerate(train_loader):
        # zero out gradients
        model.zero_grad()

        # move data to device
        data = data.to(device)

        # create mel spectrogram
        carr_freq = data[:, 0]
        mod_freq = data[:, 1]
        mod_idx = data[:, 2]
        fm_wave_in = synth(carr_freq, mod_freq, mod_idx)

        # forward pass
        fm_wave_out = model(fm_wave_in)
        # params_recon = model(data)
        fm_wave_in = fm_wave_in.view(fm_wave_in.shape[0], 1, -1)
        # print(fm_wave_in.shape)
        # print(torch.isfinite(fm_wave_in).all())
        fm_wave_out = fm_wave_out.view(fm_wave_in.shape[0], 1, -1)
        # print(fm_wave_out.shape)
        # print(torch.isfinite(fm_wave_out).all())
        # print(torch.max(fm_wave_out), torch.min(fm_wave_out))
        # print(fm_wave_out)

        # calculate loss
        # recon_loss = criterion(mel_spec_recon, mel_spec)
        # param_loss = criterion(params, data)
        # loss = 0.0 * recon_loss + 1.0 * param_loss
        loss = criterion(fm_wave_out, fm_wave_in)

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
            # f"Epoch {epoch + 1}/{args.num_epochs} | LR: {lr:.6f} | Loss: {mse_sum / mse_n:.4f}")
            f"Epoch {epoch + 1}/{args.num_epochs} | LR: {lr:.6f} | Loss: {loss.item():.4f}")

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
        reproducible=False,
        no_grad=False,
    )

    # create model and optimizer
    # model = FM_Autoencoder(config, device).to(device)
    # model = FM_Param_Autoencoder(config, device).to(device)
    model = FM_Autoencoder_Wave2(config, device, z_dim=512).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    print("Model and optimizer created")

    # print model summary
    # print(config.buffer_size)
    # summary(model, (1, config.buffer_size))

    # create synth
    # synth = FmSynth(config, device).to(device)
    synth = FmSynth2Wave(config, device).to(device)

    # create summary writer
    writer = SummaryWriter(args.log_folder)
    # create log folder
    os.makedirs(args.log_folder, exist_ok=True)
    # create checkpoint folder
    os.makedirs(args.ckpt_folder, exist_ok=True)

    # create loss function
    loss_fn = auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=[1024, 2048],
        hop_sizes=[256, 512],
        win_lengths=[1024, 2048],
        scale="mel",
        n_bins=128,
        sample_rate=args.sample_rate,
        perceptual_weighting=True,
    )

    # train model
    for epoch in range(args.num_epochs):
        recon_loss = train(train_loader, model, optimizer, loss_fn,
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
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--ckpt_folder", type=str,
                        default="/Volumes/T7/synth_dataset_fm_inversynth/ckpt")
    parser.add_argument("--ckpt_interval", type=int,
                        default=1)
    parser.add_argument("--log_folder", type=str,
                        default="/Volumes/T7/synth_dataset_fm_inversynth/logs")

    args = parser.parse_args()

    main(args)
