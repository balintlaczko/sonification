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
from sonification.datasets import White_Square_dataset_2 as White_Square_dataset
import wandb


def main():
    parser = argparse.ArgumentParser()

    # data params
    parser.add_argument('--img_size', type=int, default=128, help='image size')
    parser.add_argument('--square_size', type=int, default=2, help='size of the square')

    # model params
    parser.add_argument("--latent_size", type=int, default=2)
    parser.add_argument("--encoder_channels", type=int, default=128)
    parser.add_argument("--encoder_kernels", type=int, nargs='*', default=[3, 3])
    parser.add_argument("--encoder_n_res_block", type=int, default=32)
    parser.add_argument("--encoder_n_res_channel", type=int, default=64)
    parser.add_argument("--decoder_channels", type=int, default=128)
    parser.add_argument("--decoder_n_res_block", type=int, default=8)
    parser.add_argument("--decoder_n_res_channel", type=int, default=64)
    parser.add_argument("--d_hidden_size", type=int, default=128)
    parser.add_argument("--d_num_layers", type=int, default=5)

    # training params
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    # reconstruction loss params
    parser.add_argument('--recon_loss_type', type=str, default='l1', help='reconstruction loss type: l1 or mse')
    parser.add_argument("--recon_loss_weight_start", type=float, default=100)
    parser.add_argument("--recon_loss_weight_end", type=float, default=100)
    parser.add_argument("--recon_loss_weight_ramp_start_epoch", type=int, default=0)
    parser.add_argument("--recon_loss_weight_ramp_end_epoch", type=int, default=1)
    parser.add_argument('--target_recon_loss', type=float, default=0.01, help='target recon loss to keep in case of dynamic kld')
    # kld loss params
    parser.add_argument('--dynamic_kld', type=int, default=1, help='non-zero will use dynamic kld')
    parser.add_argument('--kld_weight_max', type=float, default=100, help='kld weight at the end of the warmup')
    parser.add_argument('--kld_weight_min', type=float, default=0.3, help='kld weight at the start of the warmup')
    parser.add_argument('--kld_start_epoch', type=int, default=0, help='the epoch at which to start the kld warmup from kld_weight_min to kld_weight_max')
    parser.add_argument('--kld_warmup_epochs', type=int, default=1, help='the number of epochs to warmup the kld weight')
    # tc loss params
    parser.add_argument('--tc_weight_max', type=float, default=10, help='tc weight at the end of the warmup')
    parser.add_argument('--tc_weight_min', type=float, default=10, help='tc weight at the start of the warmup')
    parser.add_argument('--tc_start_epoch', type=int, default=0, help='the epoch at which to start the tc warmup from tc_weight_min to tc_weight_max')
    parser.add_argument('--tc_warmup_epochs', type=int, default=1, help='the number of epochs to warmup the tc weight')
    # optimizer params
    parser.add_argument("--lr_vae", type=float, default=0.0002)
    parser.add_argument("--lr_decay_vae", type=float, default=0.85)
    parser.add_argument("--lr_d", type=float, default=0.0001)
    parser.add_argument("--lr_decay_d", type=float, default=0.85)
    parser.add_argument("--train_epochs", type=int, default=100000)
    parser.add_argument("--steps_per_epoch", type=int, default=400)
    # checkpointing & logging
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/squares_vae")
    parser.add_argument("--ckpt_name", type=str, default="imv_new_v1")
    parser.add_argument("--logdir", type=str, default="./logs/squares_vae")
    parser.add_argument("--comment", type=str, default="")

    args = parser.parse_args()

    # change relative path to absolute path
    logdir = os.path.abspath(args.logdir)
    logdir = os.path.join(logdir, args.ckpt_name)
    os.makedirs(logdir, exist_ok=True)
    print(f"Logging to {logdir}")
    args.logdir = logdir

    # create dataset
    dataset = White_Square_dataset(img_size=args.img_size, square_size=args.square_size)

    # create dataloader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4, prefetch_factor=2, persistent_workers=True)

    # create the model
    model = PlImgFactorVAE(args)

    # checkpoint callbacks
    checkpoint_path = os.path.join(args.ckpt_path, args.ckpt_name)
    best_checkpoint_callback = ModelCheckpoint(
        monitor="vae_loss",
        dirpath=checkpoint_path,
        filename=args.ckpt_name + "_best_{epoch:02d}-{vae_loss:.4f}",
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
        project="squares_fvae",
        save_dir=logdir,
        offline=False,
        settings=wandb.Settings(_disable_stats=True),
        )

    # create trainer
    trainer = Trainer(
        max_epochs=args.train_epochs,
        enable_checkpointing=True,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=100,
        limit_train_batches=args.steps_per_epoch,
    )

    hyperparams = vars(args).copy()
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
        trainer.fit(model, train_dataloaders=dataloader, ckpt_path=resume_path)
    else:
        trainer.fit(model, train_dataloaders=dataloader)


if __name__ == "__main__":
    fix_seed = 2025
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    torch.set_float32_matmul_precision('high')
    main()
