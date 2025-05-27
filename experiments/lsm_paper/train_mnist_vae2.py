import argparse
import os
import torch
import random
import numpy as np
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from sonification.models.models import PlImgFactorVAE
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb

def main():
    parser = argparse.ArgumentParser()

    # model params
    parser.add_argument("--latent_size", type=int, default=16)
    parser.add_argument("--input_width", type=int, default=32)
    parser.add_argument("--output_channels", type=int, default=1)
    parser.add_argument("--encoder_channels", type=int, default=64)
    parser.add_argument("--encoder_kernels", type=int, nargs='*', default=[3, 3])
    parser.add_argument("--encoder_n_res_block", type=int, default=4)
    parser.add_argument("--encoder_n_res_channel", type=int, default=32)
    parser.add_argument("--decoder_channels", type=int, default=64)
    parser.add_argument("--decoder_n_res_block", type=int, default=4)
    parser.add_argument("--decoder_n_res_channel", type=int, default=32)
    parser.add_argument("--d_hidden_size", type=int, default=64)
    parser.add_argument("--d_num_layers", type=int, default=5)
    # training params
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument('--target_recon_loss', type=float, default=0.01, help='target recon loss to keep in case of dynamic kld')
    parser.add_argument('--dynamic_kld', type=int, default=1, help='non-zero will use dynamic kld')
    parser.add_argument('--kld_weight_max', type=float, default=1, help='kld weight at the end of the warmup')
    parser.add_argument('--kld_weight_min', type=float, default=0.1, help='kld weight at the start of the warmup')
    parser.add_argument('--kld_start_epoch', type=int, default=1000, help='the epoch at which to start the kld warmup from kld_weight_min to kld_weight_max')
    parser.add_argument('--kld_warmup_epochs', type=int, default=10000, help='the number of epochs to warmup the kld weight')
    parser.add_argument('--tc_weight_max', type=float, default=10, help='tc weight at the end of the warmup')
    parser.add_argument('--tc_weight_min', type=float, default=0, help='tc weight at the start of the warmup')
    parser.add_argument('--tc_start_epoch', type=int, default=0, help='the epoch at which to start the tc warmup from tc_weight_min to tc_weight_max')
    parser.add_argument('--tc_warmup_epochs', type=int, default=1, help='the number of epochs to warmup the tc weight')
    parser.add_argument("--lr_vae", type=float, default=0.001)
    parser.add_argument("--lr_decay_vae", type=float, default=0.75)
    parser.add_argument("--lr_d", type=float, default=0.0001)
    parser.add_argument("--lr_decay_d", type=float, default=0.75)
    parser.add_argument("--train_epochs", type=int, default=100000)
    # parser.add_argument("--steps_per_epoch", type=int, default=100)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/mnist_vae")
    parser.add_argument("--ckpt_name", type=str, default="testparams")
    parser.add_argument("--logdir", type=str, default="./logs/mnis_vae")
    parser.add_argument("--comment", type=str, default="")
    
    args = parser.parse_args()

    # change relative path to absolute path
    logdir = os.path.abspath(args.logdir)
    logdir = os.path.join(logdir, args.ckpt_name)
    os.makedirs(logdir, exist_ok=True)
    print(f"Logging to {logdir}")
    args.logdir = logdir

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])


    # Download and load training dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Download and load validation dataset
    val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    # create model
    model = PlImgFactorVAE(args)

    # checkpoint callbacks
    checkpoint_path = os.path.join(args.ckpt_path, args.ckpt_name)
    best_checkpoint_callback = ModelCheckpoint(
        monitor="val_vae_loss",
        dirpath=checkpoint_path,
        filename=args.ckpt_name + "_best_{epoch:02d}-{val_vae_loss:.4f}",
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
    callbacks = [best_checkpoint_callback, last_checkpoint_callback]

    # create logger
    logger = WandbLogger(
        name=args.ckpt_name,
        project="mnist_vae",
        save_dir=logdir,
        offline=False,
        settings=wandb.Settings(_disable_stats=True),
        )
    # logger.watch(model, log='all')

    # create trainer
    trainer = Trainer(
        max_epochs=args.train_epochs,
        enable_checkpointing=True,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=20,
        # limit_train_batches=args.steps_per_epoch,
    )

    # save hyperparameters
    hyperparams = dict(
        latent_size=args.latent_size,
        input_width=args.input_width,
        output_channels=args.output_channels,
        encoder_channels=args.encoder_channels,
        encoder_kernels=args.encoder_kernels,
        encoder_n_res_block=args.encoder_n_res_block,
        encoder_n_res_channel=args.encoder_n_res_channel,
        decoder_channels=args.decoder_channels,
        decoder_n_res_block=args.decoder_n_res_block,
        decoder_n_res_channel=args.decoder_n_res_channel,
        d_hidden_size=args.d_hidden_size,
        d_num_layers=args.d_num_layers,
        batch_size=args.batch_size,
        warmup_epochs=args.warmup_epochs,
        target_recon_loss=args.target_recon_loss,
        dynamic_kld=args.dynamic_kld,
        kld_weight_max=args.kld_weight_max,
        kld_weight_min=args.kld_weight_min,
        kld_start_epoch=args.kld_start_epoch,
        kld_warmup_epochs=args.kld_warmup_epochs,
        tc_weight_max=args.tc_weight_max,
        tc_weight_min=args.tc_weight_min,
        tc_start_epoch=args.tc_start_epoch,
        tc_warmup_epochs=args.tc_warmup_epochs,
        lr_vae=args.lr_vae,
        lr_decay_vae=args.lr_decay_vae,
        lr_d=args.lr_d,
        lr_decay_d=args.lr_decay_d,
        train_epochs=args.train_epochs,
        comment=args.comment
    )
    trainer.logger.log_hyperparams(hyperparams)

    # train model
    try:
        resume_path = os.listdir(checkpoint_path)
    except FileNotFoundError:
        resume_path = None
    if resume_path:
        resume_path = sorted(resume_path)
        resume_path = os.path.join(checkpoint_path, resume_path[-1])
        print(f"Resuming from {resume_path}")
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=resume_path)
    else:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    fix_seed = 2025
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    torch.set_float32_matmul_precision('high')
    # os.environ["WANDB_START_METHOD"] = "thread"
    main()