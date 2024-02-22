import argparse
import os
import torch
from torch.utils.data import DataLoader
import random
import numpy as np
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from sonification.models.models import PlFactorVAE1D
from sonification.datasets import Sinewave_dataset


def main():
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser()

    # root path
    parser.add_argument('--root_path', type=str,
                        default='./experiments/lsm_paper/', help='root path')

    # dataset
    parser.add_argument('--csv_path', type=str,
                        default='sinewave.csv', help='csv path')
    parser.add_argument('--img_size', type=int, default=64, help='image size')

    # model
    parser.add_argument('--in_channels', type=int,
                        default=1, help='image color channels')
    parser.add_argument('--latent_size', type=int,
                        default=2, help='latent size')
    parser.add_argument('--layers_channels', type=int, nargs='*', default=[1024, 1024, 1024, 1024, 1024],
                        help='channels for the layers')
    parser.add_argument('--d_hidden_size', type=int,
                        default=512, help='mlp hidden size')
    parser.add_argument('--d_num_layers', type=int,
                        default=5, help='mlp number of layers')

    # training
    parser.add_argument('--train_epochs', type=int,
                        default=1000000, help='number of training epochs')
    parser.add_argument('--batch_size', type=int,
                        default=512, help='batch size')
    parser.add_argument('--lr_vae', type=float, default=1e-5,
                        help='learning rate for the vae')
    parser.add_argument('--lr_decay_vae', type=float, default=0.99)
    parser.add_argument('--lr_d', type=float, default=1e-5,
                        help='learning rate for the discriminator')
    parser.add_argument('--lr_decay_d', type=float, default=0.99)
    parser.add_argument('--kld_weight', type=float,
                        default=0.02, help='kld weight')
    parser.add_argument('--tc_weight', type=float,
                        default=0.1, help='tc weight')
    parser.add_argument('--l1_weight', type=float,
                        default=0, help='l1 weight')

    # GPU
    parser.add_argument('--num_devices', type=int, nargs='*', default=[0],
                        help='number of gpus to use. if list it will be the device ids to use')

    # checkpoint & logging
    parser.add_argument('--ckpt_path', type=str,
                        default='./ckpt/sinewave_fvae', help='checkpoint path')
    parser.add_argument('--ckpt_name', type=str,
                        default='test-v1', help='checkpoint name')
    parser.add_argument('--resume_ckpt_path', type=str,
                        default=None,)
    parser.add_argument(
        '--logdir', type=str, default='./logs/sinewave_fvae', help='log directory')
    parser.add_argument('--plot_interval', type=int, default=1)

    # quick comment
    parser.add_argument('--comment', type=str, default='',
                        help='add a comment if needed')

    args = parser.parse_args()

    # create train and val datasets and loaders
    sinewave_ds_train = Sinewave_dataset(
        root_path=args.root_path, csv_path=args.csv_path, flag="train")
    sinewave_ds_val = Sinewave_dataset(
        root_path=args.root_path, csv_path=args.csv_path, flag="val", scaler=sinewave_ds_train.scaler)

    train_loader = DataLoader(
        sinewave_ds_train, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(
        sinewave_ds_val, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True)

    # create the model
    args.train_scaler = sinewave_ds_train.scaler
    model = PlFactorVAE1D(args)

    # checkpoint callbacks
    checkpoint_path = os.path.join(args.ckpt_path, args.ckpt_name)
    best_checkpoint_callback = ModelCheckpoint(
        monitor="val_vae_loss",
        dirpath=checkpoint_path,
        filename=args.ckpt_name + "_val_{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        mode="min",
    )
    last_checkpoint_callback = ModelCheckpoint(
        monitor="epoch",
        dirpath=checkpoint_path,
        filename=args.ckpt_name + "_last_{epoch:02d}",
        save_top_k=1,
        mode="max",
    )

    # logger callbacks
    tensorboard_logger = TensorBoardLogger(
        save_dir=args.logdir, name=args.ckpt_name)

    trainer = Trainer(
        max_epochs=args.train_epochs,
        enable_checkpointing=True,
        callbacks=[best_checkpoint_callback, last_checkpoint_callback],
        logger=tensorboard_logger,
        log_every_n_steps=1,
    )

    # save hyperparameters
    hyperparams = dict(
        in_channels=args.in_channels,
        img_size=args.img_size,
        latent_size=args.latent_size,
        layers_channels=args.layers_channels,
        d_hidden_size=args.d_hidden_size,
        d_num_layers=args.d_num_layers,
        batch_size=args.batch_size,
        lr_vae=args.lr_vae,
        lr_decay_vae=args.lr_decay_vae,
        lr_d=args.lr_d,
        lr_decay_d=args.lr_decay_d,
        kld_weight=args.kld_weight,
        tc_weight=args.tc_weight,
        l1_weight=args.l1_weight,
        comment=args.comment
    )

    trainer.logger.log_hyperparams(hyperparams)

    # train the model
    trainer.fit(model, train_loader, val_loader,
                ckpt_path=args.resume_ckpt_path)


if __name__ == "__main__":
    main()
