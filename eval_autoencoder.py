import argparse
import sys
import os
import json

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from utils import *
from models.models import AE


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

    # create model from checkpoint
    model = AE(input_size=len(
        train_dataset[0]), hidden_size=2, output_size=len(train_dataset[0])).to(device)
    model.load_state_dict(torch.load(
        args.ckpt, map_location='cpu'))

    # send it to device and set it to eval mode
    model = model.to(device)
    model.eval()

    # create tensor for predictions
    predictions = torch.zeros(len(train_dataset), 2).to(device)

    # predict
    with torch.no_grad():
        for i, x in enumerate(train_loader):
            x = x.to(device)
            y = model.encode(x)
            predictions[i * args.batch_size: i *
                        args.batch_size + len(x)] = y

    # convert to numpy
    predictions = predictions.cpu().numpy()

    # create folder for predictions
    os.makedirs(args.target_folder, exist_ok=True)

    # save predictions
    with open(os.path.join(args.target_folder, "AE_predictions.json"), "w") as f:
        json.dump(array2fluid_dataset(predictions), f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--fluid_dataset", type=str,
                        default="/Volumes/T7/synth_dataset_4/fm_descriptors_merged.json")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--ckpt", type=str,
                        default="ckpt/simple_autoencoder/run_03_b1024/model_3000.pt")
    parser.add_argument("--target_folder", type=str,
                        default="/Volumes/T7/synth_dataset_4/")

    args = parser.parse_args()

    main(args)
