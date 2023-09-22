import argparse
import os
import json

import torch
from torch.utils.data import DataLoader

from utils import *
from simple_autoencoder import VAE

from sklearn.preprocessing import MinMaxScaler


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
        train_dataset, batch_size=args.batch_size, shuffle=False)

    # load model params from train args
    train_args = None
    with open(args.train_args, "r") as f:
        train_args = json.load(f)
    model_hidden_size = train_args["model_hidden_size"]
    model_latent_size = train_args["model_latent_size"]

    # create model and optimizer
    model = VAE(input_size=len(
        train_dataset[0]), hidden_size=model_hidden_size, latent_size=model_latent_size).to(device)
    model.load_state_dict(torch.load(
        args.ckpt, map_location='cpu'))

    # send it to device and set it to eval mode
    model = model.to(device)
    model.eval()

    # create tensor for predictions
    predictions = torch.zeros(len(train_dataset), model_latent_size).to(device)

    # predict
    with torch.no_grad():
        for i, x in enumerate(train_loader):
            x = x.to(device)
            mean, logvar = model.encode(x)
            z = model.reparameterize(mean, logvar)
            predictions[i * args.batch_size: i *
                        args.batch_size + len(x)] = z

    # convert to numpy
    predictions = predictions.cpu().numpy()

    # create folder for predictions
    os.makedirs(args.target_folder, exist_ok=True)

    # save predictions
    with open(os.path.join(args.target_folder, "VAE_predictions.json"), "w") as f:
        json.dump(array2fluid_dataset(predictions), f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="/Volumes/T7/synth_dataset_2/melspec_2_mean_std.npy")
    parser.add_argument("--train_args", type=str,
                        default="logs/simple_vae/run33/args.json")
    parser.add_argument("--ckpt", type=str,
                        default="ckpt/simple_vae/run33_model_2000.pt")
    parser.add_argument("--target_folder", type=str,
                        default="/Volumes/T7/simple_vae_clustering/")
    parser.add_argument("--batch_size", type=int, default=1024)

    args = parser.parse_args()

    main(args)
