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
from models.models import VAE
from loss import MMDloss

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


def train(train_loader, model, optimizer, epoch, device, args, kld_warmpup_factor):
    # set model to train mode
    model.train()

    # create progress bar from dataloader
    train_loader = tqdm(train_loader)

    # create loss function
    criterion = nn.MSELoss()
    mmd = MMDloss(kernel_type=args.mmd_kernel_type, latent_var=args.mmd_latent_var)

    # initialize loss variables for the epoch
    combined_loss_sum = 0
    recon_loss_sum = 0
    scaled_recon_loss_sum = 0
    KLD_sum = 0
    scaled_KLD_sum = 0
    MMD_sum = 0
    scaled_MMD_sum = 0
    loss_n = 0
    epoch_embeddings = None

    for batch_idx, data in enumerate(train_loader):
        # zero out gradients
        model.zero_grad()

        # move data to device
        data = data.to(device)

        # forward pass
        output, mean, logvar, z = model(data)

        if batch_idx == 0:
            epoch_embeddings = z
        else:
            epoch_embeddings = torch.cat((epoch_embeddings, z), dim=0)

        # KLD corrected according to: https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed
        # calculate loss
        recon_loss = criterion(output, data)
        scaled_recon_loss = recon_loss * args.beta

        KLD, scaled_KLD = 0, 0
        compute_kld = args.kld_weight > 0 and args.kld_start_epoch <= epoch and args.alpha < 1
        if compute_kld:
            KLD = torch.mean(-0.5 * torch.sum(1 + logvar -
                            mean.pow(2) - logvar.exp(), dim=1), dim=0)
            scaled_KLD = KLD * args.kld_weight * \
                kld_warmpup_factor * (1. - args.alpha)
            
        mmd_loss, scaled_mmd_loss = 0, 0
        if args.mmd_weight > 0:
            mmd_loss = mmd.compute_mmd(z)
            bias_corr = args.batch_size * (args.batch_size - 1)
            scaled_mmd_loss = (args.alpha + args.reg_weight - 1.) / \
                bias_corr * mmd_loss * args.mmd_weight
            
        combined_loss = scaled_recon_loss + scaled_KLD + scaled_mmd_loss

        # backpropagate
        combined_loss.backward()
        optimizer.step()

        # update loss variables
        combined_loss_sum += combined_loss.item() * data.shape[0]
        recon_loss_sum += recon_loss.item() * data.shape[0]
        scaled_recon_loss_sum += scaled_recon_loss.item() * data.shape[0]
        if compute_kld:
            KLD_sum += KLD.item() * data.shape[0]
            scaled_KLD_sum += scaled_KLD.item() * data.shape[0]
        if args.mmd_weight > 0:
            MMD_sum += mmd_loss.item() * data.shape[0]
            scaled_MMD_sum += scaled_mmd_loss.item() * data.shape[0]
        loss_n += data.shape[0]

        # get the learning rate from the optimizer
        lr = optimizer.param_groups[0]['lr']

        # update progress bar
        train_loader.set_description(
            f"Epoch {epoch + 1}/{args.num_epochs} | LR: {lr:.6f} | Combined Loss: {combined_loss_sum / loss_n:.6f} | Recon Loss: {recon_loss_sum / loss_n:.6f} | Scaled Recon: {scaled_recon_loss_sum / loss_n:.6f} | KLD: {KLD_sum / loss_n:.3f} | Scaled KLD: {scaled_KLD_sum / loss_n:.6f} | MMD: {MMD_sum / loss_n:.6f} | Scaled MMD: {scaled_MMD_sum / loss_n:.6f}")

        # update learning rate at the end of the epoch
        if batch_idx == len(train_loader) - 1:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * args.lr_decay

    # return average loss for the epoch
    return combined_loss_sum / loss_n, recon_loss_sum / loss_n, KLD_sum / loss_n, MMD_sum / loss_n, epoch_embeddings, lr


def main(args):
    # set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # load dataset
    train_dataset = None
    train_dataset = np.load(args.dataset)
    print(
        f"Loaded dataset with {len(train_dataset)} samples and {len(train_dataset[0])} features")

    train_dataset = train_dataset[..., 0]

    # normalize
    scaler = MinMaxScaler()
    train_dataset = scaler.fit_transform(train_dataset)

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

    # create linear warmup for KLD loss
    kld_warmup_curve = np.linspace(0, 1, args.kld_warmup_epochs)

    # train model
    for epoch in tqdm(range(args.num_epochs)):

        kld_warmup_counter = np.clip(
            epoch - args.kld_start_epoch, 0, args.kld_warmup_epochs-1)

        combined_loss, recon_loss, KLD_loss, MMD_loss, embeddings, current_lr = train(
            train_loader, model, optimizer, epoch, device, args, kld_warmup_curve[kld_warmup_counter])

        # log loss
        writer.add_scalar("Loss/train", combined_loss, epoch)
        writer.add_scalar("Recon Loss/train", recon_loss, epoch)
        writer.add_scalar("KLD Loss/train", KLD_loss, epoch)
        writer.add_scalar("MMD Loss/train", MMD_loss, epoch)
        writer.add_scalar("LR", current_lr, epoch)
        writer.add_scalar("KLD Warmup Factor",
                          kld_warmup_curve[kld_warmup_counter], epoch)

        # reduce the dimensionality of the vectors to 2
        if args.model_latent_size > 2:
            reduced_vectors = pca.fit_transform(
                embeddings.detach().cpu().numpy())
        else:
            reduced_vectors = embeddings.detach().cpu().numpy()
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
    parser.add_argument("--model_latent_size", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=-0.5)
    parser.add_argument("--beta", type=float, default=5)
    parser.add_argument("--reg_weight", type=float, default=100)
    parser.add_argument("--kld_weight", type=float, default=1)
    parser.add_argument("--kld_start_epoch", type=int, default=0)
    parser.add_argument("--kld_warmup_epochs", type=int, default=100)
    parser.add_argument("--mmd_weight", type=float, default=1)
    parser.add_argument("--mmd_kernel_type", type=str, default="imq")
    parser.add_argument("--mmd_latent_var", type=float, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_decay", type=float, default=0.99)
    parser.add_argument("--ckpt_folder", type=str,
                        default="ckpt/simple_vae")
    parser.add_argument("--ckpt_interval", type=int,
                        default=10)
    parser.add_argument("--log_folder", type=str,
                        default="logs/simple_vae")
    parser.add_argument("--plots_folder", type=str,
                        default="plots/simple_vae")
    parser.add_argument("--notes", type=str, default="")

    args = parser.parse_args()

    main(args)
