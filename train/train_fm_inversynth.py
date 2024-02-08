# imports
import argparse
import json
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary


from sonification.models.ddsp import Tsynth_FmSynth, Wave2Params
from sonification.utils.misc import scale_linear, midi2frequency
from sonification.utils.tensor import scale

import auraloss
from tqdm import tqdm


def train(train_loader, model, optimizer, loss_fn, synth, epoch, device, args):
    # set model to train mode
    model.train()

    # create progress bar from dataloader
    train_loader = tqdm(train_loader)

    # create loss functions
    criterion_mse = nn.MSELoss()
    criterion = loss_fn

    # initialize loss variables for the epoch
    mse_sum = 0
    mse_n = 0

    for batch_idx, data in enumerate(train_loader):
        # zero out gradients
        model.zero_grad()

        # move data to device
        data = data.to(device)

        # scale synth params from 0-1 to their respective ranges
        carr_freq = data[:, 0]
        carr_freq_midi = scale_linear(carr_freq, 0, 1, 44, 88)
        carr_freq_hz = midi2frequency(carr_freq_midi)
        harm_ratio = data[:, 1]
        harm_ratio_scaled = scale_linear(harm_ratio, 0, 1, 1, 10)
        mod_index = data[:, 2]
        mod_index_scaled = scale_linear(mod_index, 0, 1, 0.1, 10)

        # take each param from all batches and repeat it for the number of samples in the buffer
        carr_freq_array = carr_freq_hz.unsqueeze(-1).repeat(
            1, int(args.buffer_length_s * args.sample_rate))
        harm_ratio_array = harm_ratio_scaled.unsqueeze(
            -1).repeat(1, int(args.buffer_length_s * args.sample_rate))
        mod_index_array = mod_index_scaled.unsqueeze(
            -1).repeat(1, int(args.buffer_length_s * args.sample_rate))

        # create input synth buffers
        y = synth(carr_freq_array, harm_ratio_array, mod_index_array)

        # forward pass
        y_pred, params_pred = model(y)

        # add audio channels dim for loss function
        y = y.view(y.shape[0], 1, -1)
        y_pred = y_pred.view(y.shape[0], 1, -1)

        # calculate param loss + recon loss
        loss = criterion_mse(params_pred, data) * args.param_loss_weight + \
            criterion_mse(y_pred, y) * args.recon_loss_weight

        # backpropagate
        loss.backward()
        optimizer.step()

        # update loss variables
        mse_sum += loss.item() * data.shape[0]
        mse_n += data.shape[0]

        # get the learning rate from the optimizer
        lr = optimizer.param_groups[0]['lr']
        # apply learning rate decay for next round
        if args.lr_decay < 1 and batch_idx == len(train_loader) - 1:
            # decay learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.lr_decay

        # update progress bar
        train_loader.set_description(
            f"Epoch {epoch + 1}/{args.num_epochs} | LR: {lr:.6f} | Loss: {(mse_sum/mse_n):.4f} | Params: MIN: {float(params_pred.min()):.4f} MAX: {float(params_pred.max()):.4f} Uniques: {len(torch.unique(params_pred))}")

    # return average loss for the epoch
    return mse_sum / mse_n


def main(args):
    # set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # # load parameter values from npy file
    # train_dataset = np.load(args.params_dataset)
    # print(
    #     f"Loaded dataset with {len(train_dataset)} samples and {len(train_dataset[0])} features")

    # # convert to tensor
    # train_dataset = torch.from_numpy(train_dataset).float().to(device)

    # generate normalized (0-1) params dataset
    num_samples = args.batch_size * args.steps_per_epoch
    num_params = 3
    train_dataset = torch.randn((num_samples, num_params), device=device)
    train_dataset = scale(
        train_dataset, train_dataset.min(), train_dataset.max(), 0, 1, 1)
    print(
        f"Generated dataset with {len(train_dataset)} samples and {len(train_dataset[0])} features")

    # create dataloader
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # create model and optimizer
    model = Wave2Params(
        sr=args.sample_rate,
        n_mels=args.n_mels,
        n_mfcc=args.n_mfcc,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        f_min=args.f_min,
        f_max=args.f_max,
        use_gru=args.use_gru == 1,
        gru_hidden_dim=args.gru_hidden_dim,
        mlp_in_dim=args.mlp_in_dim,
        mlp_out_dim=args.mlp_out_dim,
        mlp_layers=args.mlp_layers,
        buffer_length_s=args.buffer_length_s,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print("Model and optimizer created")

    # print model summary
    summary(model, (int(args.sample_rate * args.buffer_length_s),))

    # create synth
    synth = Tsynth_FmSynth(sr=args.sample_rate).to(device)

    # create summary writer
    writer = SummaryWriter(args.log_folder)
    # create log folder
    os.makedirs(args.log_folder, exist_ok=True)
    # create checkpoint folder
    os.makedirs(args.ckpt_folder, exist_ok=True)

    # save args to file
    with open(f"{args.log_folder}/args.json", "w") as f:
        json.dump(args.__dict__, f, indent=2)

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
        loss = train(train_loader, model, optimizer, loss_fn,
                     synth, epoch, device, args)

        # log loss
        writer.add_scalar("Loss/train", loss, epoch)

        # save model at checkpoint interval
        if (epoch + 1) % args.ckpt_interval == 0:
            torch.save(model.state_dict(), os.path.join(
                args.ckpt_folder, f"model_{str(epoch + 1).zfill(4)}.pt"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument("--params_dataset", type=str,
    #                     default="/Volumes/T7/synth_dataset_fm_inversynth/params_scaled.npy")

    # train params
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--steps_per_epoch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_decay", type=float, default=0.99)
    parser.add_argument("--param_loss_weight", type=float, default=1.0)
    parser.add_argument("--recon_loss_weight", type=float, default=1.0)

    # model params
    parser.add_argument("--sample_rate", type=int, default=48000)
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--n_mfcc", type=int, default=30)
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--f_min", type=float, default=20.0)
    parser.add_argument("--f_max", type=float, default=20000.0)
    parser.add_argument("--use_gru", type=int, default=1)
    parser.add_argument("--gru_hidden_dim", type=int, default=128)
    parser.add_argument("--mlp_in_dim", type=int, default=16)
    parser.add_argument("--mlp_out_dim", type=int, default=64)
    parser.add_argument("--mlp_layers", type=int, default=3)
    parser.add_argument("--buffer_length_s", type=float, default=2.0)

    # logging params
    parser.add_argument("--ckpt_folder", type=str,
                        default="/Volumes/T7/synth_dataset_fm_inversynth/ckpt")
    parser.add_argument("--ckpt_interval", type=int,
                        default=1)
    parser.add_argument("--log_folder", type=str,
                        default="/Volumes/T7/synth_dataset_fm_inversynth/logs")

    args = parser.parse_args()

    main(args)
