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
                        default=r'C:\Users\Balint Laczko\Desktop\work\Sonification\CELLULAR\images', help='root path')

    # dataset
    parser.add_argument('--csv_path', type=str,
                        default='./experiments/lsm_paper/cellular.csv', help='csv path')
    parser.add_argument('--img_size', type=int,
                        default=2048, help='image size')
    parser.add_argument('--patch_size', type=int,
                        default=256, help='size of an image patch')

    # model
    parser.add_argument('--in_channels', type=int,
                        default=2, help='image color channels')
    parser.add_argument('--latent_size', type=int,
                        default=32, help='latent size')
    parser.add_argument('--layers_channels', type=int, nargs='*', default=[8, 16, 32, 64],
                        help='channels for the layers')

    # training
    parser.add_argument('--train_epochs', type=int,
                        default=1000000, help='number of training epochs')
    parser.add_argument('--batch_size', type=int,
                        default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate for the vae')
    parser.add_argument('--lr_decay', type=float, default=0.9999999)

    # recon loss
    parser.add_argument('--recon_weight', type=float, default=1,
                        help='reconstruction weight')
    parser.add_argument('--target_recon_loss', type=float, default=0.01,
                        help='target recon loss to keep in case of dynamic kld')
    parser.add_argument('--stop_on_target_recon_loss', type=int, default=0,
                        help='non-zero will stop training if the recon loss is below target_recon_loss')

    # kld loss
    parser.add_argument('--dynamic_kld', type=int, default=0,
                        help='non-zero will use dynamic kld')
    parser.add_argument('--dynamic_kld_increment', type=float, default=0.000005,
                        help="in dynamic kld mode, increment the kld this much after every epoch when recon loss is below target")
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
    val_dataset = CellularDataset(
        csv_path=args.csv_path,
        root_dir=args.root_path,
        img_size=args.img_size,
        kernel_size=args.patch_size,
        flag="val")
    # set the minmax values of val dataset to the train dataset for scaling
    val_dataset.r_min = train_dataset.r_min
    val_dataset.r_max = train_dataset.r_max
    val_dataset.g_min = train_dataset.g_min
    val_dataset.g_max = train_dataset.g_max
    # TODO: add val dataset + loader, continue rewriting from here

    # create train and val dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=2, prefetch_factor=2, persistent_workers=True,)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, pin_memory=True, num_workers=2, prefetch_factor=2, persistent_workers=True,)

    # create the model
    args.img_size = args.patch_size
    model = PlVAE(args)

    # checkpoint callbacks
    checkpoint_path = os.path.join(args.ckpt_path, args.ckpt_name)
    best_checkpoint_callback = ModelCheckpoint(
        monitor="val_vae_loss",
        dirpath=checkpoint_path,
        filename=args.ckpt_name + "_{epoch:02d}-{val_vae_loss:.4f}",
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

    # logger callback
    wandb_logger = WandbLogger(
        name=args.ckpt_name,
        project="cellular_vae",
        save_dir=args.logdir,)

    trainer = Trainer(
        max_epochs=args.train_epochs,
        enable_checkpointing=True,
        callbacks=[best_checkpoint_callback,
                   last_checkpoint_callback],
        logger=wandb_logger,
        log_every_n_steps=1,
        # limit_train_batches=1,
        # limit_val_batches=1,
    )

    hyperparameters = dict(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_channels=args.in_channels,
        latent_size=args.latent_size,
        layers_channels=args.layers_channels,
        lr=args.lr,
        lr_decay=args.lr_decay,
        recon_weight=args.recon_weight,
        target_recon_loss=args.target_recon_loss,
        dynamic_kld=args.dynamic_kld,
        dynamic_kld_increment=args.dynamic_kld_increment,
        kld_weight_max=args.kld_weight_max,
        kld_weight_min=args.kld_weight_min,
        kld_start_epoch=args.kld_start_epoch,
        kld_warmup_epochs=args.kld_warmup_epochs,
        comment=args.comment,
    )
    trainer.logger.log_hyperparams(hyperparameters)

    trainer.fit(model, train_loader, val_loader,
                ckpt_path=args.resume_ckpt_path)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    main()
