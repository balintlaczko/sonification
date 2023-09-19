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
from simple_autoencoder import VAE

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


def train(train_loader, model, optimizer, epoch, device):
    # set model to train mode
    model.train()

    # create progress bar from dataloader
    train_loader = tqdm(train_loader)

    # create loss function
    criterion = nn.MSELoss()

    # initialize loss variables for the epoch
    loss_sum = 0
    loss_n = 0
    epoch_embeddings = None

    for batch_idx, data in enumerate(train_loader):
        # zero out gradients
        model.zero_grad()

        # move data to device
        data = data.to(device)

        # forward pass
        output, mean, logvar, z = model(data)

        # print("data.shape", data.shape)
        # print("output.shape", output.shape)
        # print("output", output)

        # print("z.shape", z.shape)

        if batch_idx == 0:
            epoch_embeddings = z
        else:
            epoch_embeddings = torch.cat((epoch_embeddings, z), dim=0)

        # calculate loss
        recon_loss = criterion(output, data)
        # print("recon_loss.shape", recon_loss.shape)
        # print("recon_loss", recon_loss)
        KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        # print("KLD.shape", KLD.shape)
        # print("KLD", KLD)
        combined_loss = recon_loss + KLD

        # backpropagate
        combined_loss.backward()
        optimizer.step()

        # update loss variables
        loss_sum += combined_loss.item() * data.shape[0]
        loss_n += data.shape[0]

        # get the learning rate from the optimizer
        lr = optimizer.param_groups[0]['lr']

        # update progress bar
        train_loader.set_description(
            f"Epoch {epoch + 1}/{args.num_epochs} | LR: {lr:.6f} | Loss: {loss_sum / loss_n:.4f}")

    # return average loss for the epoch
    return loss_sum / loss_n, epoch_embeddings


def main(args):
    # set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # load dataset
    train_dataset = None
    train_dataset = np.load(args.dataset)
    print(
        f"Loaded dataset with {len(train_dataset)} samples and {len(train_dataset[0])} features")

    print("train_dataset.shape", train_dataset.shape)
    train_dataset = train_dataset[..., 0]
    print("train_dataset.shape", train_dataset.shape)

    # normalize
    scaler = MinMaxScaler()
    train_dataset = scaler.fit_transform(train_dataset)
    print("train_dataset.shape", train_dataset.shape)

    # convert to tensor
    train_dataset = torch.from_numpy(train_dataset).float().to(device)

    # create dataloader
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)

    # create model and optimizer
    model = VAE(input_size=len(
        train_dataset[0]), hidden_size=args.model_hidden_size, latent_size=args.model_latent_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # create summary writer
    writer = SummaryWriter(args.log_folder)
    # create log folder
    os.makedirs(args.log_folder, exist_ok=True)
    # create checkpoint folder
    os.makedirs(args.ckpt_folder, exist_ok=True)
    # create plots folder
    os.makedirs(args.plots_folder, exist_ok=True)

    # save args to file
    with open(f"{args.log_folder}/args.json", "w") as f:
        json.dump(args.__dict__, f, indent=2)

    pca = PCA(n_components=2)

    # train model
    for epoch in tqdm(range(args.num_epochs)):
        loss, embeddings = train(train_loader, model, optimizer, epoch, device)

        # log loss
        writer.add_scalar("Loss/train", loss, epoch)

        # reduce the dimensionality of the vectors to 2
        reduced_vectors = pca.fit_transform(embeddings.detach().cpu().numpy())
        # plot the vectors
        plt.scatter(reduced_vectors[:, 0],
                    reduced_vectors[:, 1], s=1, alpha=0.5)
        plt.title('embedding space')
        plot_save_path = f"{args.plots_folder}/embedding_space_{str(epoch).zfill(4)}.png"
        plt.savefig(plot_save_path)
        plt.clf()

        # save model at checkpoint interval
        if (epoch + 1) % args.ckpt_interval == 0:
            torch.save(model.state_dict(), os.path.join(
                args.ckpt_folder, f"model_{str(epoch + 1).zfill(4)}.pt"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="/Volumes/T7/synth_dataset_2/melspec.npy")
    parser.add_argument("--model_hidden_size", type=int, default=512)
    parser.add_argument("--model_latent_size", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ckpt_folder", type=str,
                        default="ckpt/simple_vae")
    parser.add_argument("--ckpt_interval", type=int,
                        default=10)
    parser.add_argument("--log_folder", type=str,
                        default="logs/simple_vae")
    parser.add_argument("--plots_folder", type=str,
                        default="plots/simple_vae")

    args = parser.parse_args()

    main(args)
