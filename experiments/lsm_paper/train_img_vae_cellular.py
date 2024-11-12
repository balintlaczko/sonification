import argparse
import os
import torch
from torch.utils.data import DataLoader
import random
import numpy as np
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from sonification.models.models import PlVAE
from sonification.datasets import CellularDataset


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
                        default=r'C:\Users\Balint Laczko\Desktop\work\Sonification\CELLULAR\images', help='csv path')
    parser.add_argument('--img_size', type=int,
                        default=2048, help='image size')
    parser.add_argument('--patch_size', type=int,
                        default=256, help='size of an image patch')

    # model
    parser.add_argument('--in_channels', type=int,
                        default=2, help='image color channels')
    parser.add_argument('--latent_size', type=int,
                        default=32, help='latent size')
    parser.add_argument('--layers_channels', type=int, nargs='*', default=[64, 128, 256, 512],
                        help='channels for the layers')

    # training
    parser.add_argument('--train_epochs', type=int,
                        default=1000000, help='number of training epochs')
    parser.add_argument('--batch_size', type=int,
                        default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate for the vae')
    parser.add_argument('--lr_decay', type=float, default=0.9999)
    parser.add_argument('--recon_weight', type=float, default=1,
                        help='reconstruction weight')
    parser.add_argument('--kld_weight_max', type=float,
                        default=0.1, help='kld weight at the end of the warmup')
    parser.add_argument('--kld_weight_min', type=float, default=0.01,
                        help='kld weight at the start of the warmup')
    parser.add_argument('--kld_start_epoch', type=int, default=0,
                        help='the epoch at which to start the kld warmup from kld_weight_min to kld_weight_max')
    parser.add_argument('--kld_warmup_epochs', type=int, default=1000,
                        help='the number of epochs to warmup the kld weight')

    # GPU
    parser.add_argument('--num_devices', type=int, nargs='*', default=[0],
                        help='number of gpus to use. if list it will be the device ids to use')

    # checkpoint & logging
    parser.add_argument('--ckpt_path', type=str,
                        default='./ckpt/cellular', help='checkpoint path')
    parser.add_argument('--ckpt_name', type=str,
                        default='cellular-v1', help='checkpoint name')
    parser.add_argument('--resume_ckpt_path', type=str,
                        default=None,)
    parser.add_argument(
        '--logdir', type=str, default='./logs/cellular', help='log directory')
    parser.add_argument('--plot_interval', type=int, default=1)

    # quick comment
    parser.add_argument('--comment', type=str, default='first attempt',
                        help='add a comment if needed')

    args = parser.parse_args()
    # args.onpix_weight = args.img_size // args.square_size
    # args.onpix_weight = 1

    # create train dataset
    train_dataset = CellularDataset(
        csv_path=args.csv_path,
        root_dir=args.root_path,
        img_size=args.img_size,
        kernel_size=args.patch_size,
        flag="train")
    # TODO: implement minmax scaler, add val dataset + loader, continue rewriting from here

    # create train and val dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4, prefetch_factor=2, persistent_workers=True,)

    # create the model
    model = PlVAE(args)

    # checkpoint callbacks
    checkpoint_path = os.path.join(args.ckpt_path, args.ckpt_name)
    best_checkpoint_callback = ModelCheckpoint(
        monitor="vae_loss",
        dirpath=checkpoint_path,
        filename=args.ckpt_name + "_{epoch:02d}-{loss:.4f}",
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
        callbacks=[best_checkpoint_callback,
                   last_checkpoint_callback],
        logger=tensorboard_logger,
        log_every_n_steps=1,
    )

    hyperparameters = dict(
        img_size=args.img_size,
        square_size=args.square_size,
        latent_size=args.latent_size,
        layers_channels=args.layers_channels,
        d_hidden_size=args.d_hidden_size,
        d_num_layers=args.d_num_layers,
        lr_vae=args.lr_vae,
        lr_decay_vae=args.lr_decay_vae,
        lr_d=args.lr_d,
        lr_decay_d=args.lr_decay_d,
        recon_weight=args.recon_weight,
        kld_weight_max=args.kld_weight_max,
        kld_weight_min=args.kld_weight_min,
        kld_start_epoch=args.kld_start_epoch,
        kld_warmup_epochs=args.kld_warmup_epochs,
        tc_weight=args.tc_weight,
        comment=args.comment,
    )
    trainer.logger.log_hyperparams(hyperparameters)

    trainer.fit(model=model, train_dataloaders=train_loader,
                ckpt_path=args.resume_ckpt_path)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    main()
