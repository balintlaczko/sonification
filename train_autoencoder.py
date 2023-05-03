import argparse
import sys
import os
import json

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils import *
from simple_autoencoder import Autoencoder


def train(train_loader, model, optimizer, epoch, device):
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

        # forward pass
        output = model(data)

        # calculate loss
        recon_loss = criterion(output, data)

        # backpropagate
        recon_loss.backward()
        optimizer.step()

        # update loss variables
        mse_sum += recon_loss.item() * data.shape[0]
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

    # load dataset
    train_dataset = None
    with open(args.fluid_dataset, "r") as f:
        train_dataset_json = json.load(f)
        train_dataset = fluid_dataset2array(train_dataset_json)
    print(
        f"Loaded dataset with {len(train_dataset)} samples and {len(train_dataset[0])} features")

    # convert to tensor
    train_dataset = torch.from_numpy(train_dataset).float().to(device)

    # create dataloader
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)

    # create model and optimizer
    model = Autoencoder(input_size=len(
        train_dataset[0]), hidden_size=2, output_size=len(train_dataset[0])).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # create summary writer
    writer = SummaryWriter(args.log_folder)
    # create log folder
    os.makedirs(args.log_folder, exist_ok=True)
    # create checkpoint folder
    os.makedirs(args.ckpt_folder, exist_ok=True)

    # train model
    for epoch in tqdm(range(args.num_epochs)):
        recon_loss = train(train_loader, model, optimizer, epoch, device)

        # log loss
        writer.add_scalar("Loss/train", recon_loss, epoch)

        # save model
        torch.save(model.state_dict(), os.path.join(
            args.ckpt_folder, f"model_{str(epoch + 1).zfill(4)}.pt"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--fluid_dataset", type=str,
                        default="/Volumes/T7/synth_dataset_4/fm_descriptors_merged.json")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--ckpt_folder", type=str,
                        default="ckpt/simple_autoencoder")
    parser.add_argument("--log_folder", type=str,
                        default="logs/simple_autoencoder")

    args = parser.parse_args()

    main(args)
