import argparse
import os
import torch
from torch.utils.data import DataLoader
import random
import numpy as np
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from sonification.models.models import PlVAE
from torchvision import transforms
from torchvision.datasets import MNIST


def main():
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--in_channels', type=int,
                        default=1, help='image color channels')
    parser.add_argument('--img_size', type=int, default=32, help='image size')
    parser.add_argument('--latent_size', type=int,
                        default=2, help='latent size')
    parser.add_argument('--layers_channels', type=int, nargs='*', default=[128, 256, 512, 1024],
                        help='channels for the layers')

    # training
    parser.add_argument('--train_epochs', type=int,
                        default=10000000, help='number of training epochs')
    parser.add_argument('--batch_size', type=int,
                        default=2048, help='batch size')
    parser.add_argument('--lr_vae', type=float, default=1e-3,
                        help='learning rate for the vae')
    parser.add_argument('--lr_decay_vae', type=float,
                        default=0.9999)
    parser.add_argument('--recon_weight', type=float,
                        default=1, help='recon weight')
    parser.add_argument('--target_recon_loss', type=float, default=0.01,
                        help='target recon loss to keep in case of dynamic kld')
    parser.add_argument('--dynamic_kld', type=int, default=1,
                        help='non-zero will use dynamic kld')
    parser.add_argument('--kld_weight_max', type=float,
                        default=0.01, help='kld weight at the end of the warmup')
    parser.add_argument('--kld_weight_min', type=float, default=0.01,
                        help='kld weight at the start of the warmup')
    parser.add_argument('--kld_start_epoch', type=int, default=0,
                        help='the epoch at which to start the kld warmup from kld_weight_min to kld_weight_max')
    parser.add_argument('--kld_warmup_epochs', type=int, default=1,
                        help='the number of epochs to warmup the kld weight')

    # GPU
    parser.add_argument('--num_devices', type=int, nargs='*', default=[0],
                        help='number of gpus to use. if list it will be the device ids to use')

    # checkpoint & logging
    parser.add_argument('--ckpt_path', type=str,
                        default='./ckpt/mnist_vae', help='checkpoint path')
    parser.add_argument('--ckpt_name', type=str,
                        default='mnist-v3', help='checkpoint name')
    parser.add_argument('--resume_ckpt_path', type=str,
                        default=None,)
    parser.add_argument(
        '--logdir', type=str, default='./logs/mnist_vae', help='log directory')
    parser.add_argument('--plot_interval', type=int, default=10)

    # quick comment
    parser.add_argument('--comment', type=str, default='mnist vae test',
                        help='add a comment if needed')

    args = parser.parse_args()

    # create MNIST dataset
    # create a transform to pad the images to 32x32
    pad = transforms.Pad(padding=2)
    mnist_train = MNIST(root="./", train=True, transform=transforms.Compose(
        [transforms.ToTensor(), pad]), download=True)
    mnist_val = MNIST(root="./", train=False, transform=transforms.Compose(
        [transforms.ToTensor(), pad]), download=True)

    # create dataloaders
    train_loader = DataLoader(
        mnist_train, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(mnist_val, batch_size=args.batch_size,
                            shuffle=False, drop_last=False, num_workers=4, persistent_workers=True)

    # create the model
    model = PlVAE(args)

    # checkpoint callbacks
    checkpoint_path = os.path.join(
        args.ckpt_path, args.ckpt_name)
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

    # logger callbacks
    tensorboard_logger = TensorBoardLogger(
        save_dir=args.logdir, name=args.ckpt_name)

    trainer = Trainer(
        max_epochs=args.train_epochs,
        enable_checkpointing=True,
        callbacks=[best_checkpoint_callback, last_checkpoint_callback],
        logger=tensorboard_logger,
    )

    # save hyperparameters
    hyperparams = dict(
        in_channels=args.in_channels,
        img_size=args.img_size,
        latent_size=args.latent_size,
        layers_channels=args.layers_channels,
        batch_size=args.batch_size,
        lr_vae=args.lr_vae,
        lr_decay_vae=args.lr_decay_vae,
        recon_weight=args.recon_weight,
        target_recon_loss=args.target_recon_loss,
        dynamic_kld=args.dynamic_kld,
        kld_weight_max=args.kld_weight_max,
        kld_weight_min=args.kld_weight_min,
        kld_start_epoch=args.kld_start_epoch,
        kld_warmup_epochs=args.kld_warmup_epochs,
        comment="test"
    )

    trainer.logger.log_hyperparams(hyperparams)

    # train the model
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader,
                ckpt_path=args.resume_ckpt_path)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    main()
