import argparse
import os
import torch
import random
import numpy as np
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from sonification.models.models import PlFMFactorVAE
from sonification.utils.misc import midi2frequency
from torch.utils.data import DataLoader
import wandb

def main():
    parser = argparse.ArgumentParser()

    # audio params
    parser.add_argument("--sr", type=int, default=48000)
    parser.add_argument("--length_samps", type=int, default=8192)
    parser.add_argument("--n_fft", type=int, default=4096)
    parser.add_argument("--f_min", type=float, default=20)
    parser.add_argument("--f_max", type=float, default=16000)
    parser.add_argument("--n_mels", type=int, default=512)
    parser.add_argument("--power", type=float, default=1)
    parser.add_argument("--normalized", type=int, default=1)
    parser.add_argument("--max_harm_ratio", type=int, default=6)
    parser.add_argument("--max_mod_idx", type=int, default=6)
    # model params
    parser.add_argument("--latent_size", type=int, default=8)
    parser.add_argument("--encoder_channels", type=int, default=128)
    parser.add_argument("--encoder_kernels", type=int, nargs='*', default=[3, 5])
    parser.add_argument("--encoder_n_res_block", type=int, default=24)
    parser.add_argument("--encoder_n_res_channel", type=int, default=64)
    parser.add_argument("--decoder_features", type=int, default=128)
    parser.add_argument("--decoder_n_res_block", type=int, default=32)
    parser.add_argument("--decoder_n_res_features", type=int, default=64)
    parser.add_argument("--d_hidden_size", type=int, default=64)
    parser.add_argument("--d_num_layers", type=int, default=2)
    # training params
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--param_loss_weight_start", type=int, default=150)
    parser.add_argument("--param_loss_weight_end", type=int, default=150)
    parser.add_argument("--param_loss_weight_ramp_start_epoch", type=int, default=0)
    parser.add_argument("--param_loss_weight_ramp_end_epoch", type=int, default=1)
    parser.add_argument("--recon_weight", type=int, default=1)
    parser.add_argument('--target_recon_loss', type=float, default=0.02, help='target recon loss to keep in case of dynamic kld')
    parser.add_argument('--dynamic_kld', type=int, default=0, help='non-zero will use dynamic kld')
    parser.add_argument('--kld_weight_max', type=float, default=1, help='kld weight at the end of the warmup')
    parser.add_argument('--kld_weight_min', type=float, default=0.1, help='kld weight at the start of the warmup')
    parser.add_argument('--kld_start_epoch', type=int, default=2000, help='the epoch at which to start the kld warmup from kld_weight_min to kld_weight_max')
    parser.add_argument('--kld_warmup_epochs', type=int, default=2000, help='the number of epochs to warmup the kld weight')
    parser.add_argument('--tc_weight_max', type=float, default=10, help='tc weight at the end of the warmup')
    parser.add_argument('--tc_weight_min', type=float, default=0, help='tc weight at the start of the warmup')
    parser.add_argument('--tc_start_epoch', type=int, default=4000, help='the epoch at which to start the tc warmup from tc_weight_min to tc_weight_max')
    parser.add_argument('--tc_warmup_epochs', type=int, default=2000, help='the number of epochs to warmup the tc weight')
    parser.add_argument("--lr_vae", type=float, default=0.0001)
    parser.add_argument("--lr_decay_vae", type=float, default=0.75)
    parser.add_argument("--lr_d", type=float, default=0.00001)
    parser.add_argument("--lr_decay_d", type=float, default=0.75)
    parser.add_argument("--train_epochs", type=int, default=20000)
    parser.add_argument("--steps_per_epoch", type=int, default=100)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/fm_vae")
    parser.add_argument("--ckpt_name", type=str, default="imv_v1.7")
    parser.add_argument("--logdir", type=str, default="./logs/fm_vae")
    parser.add_argument("--comment", type=str, default="delayed fade in, kld1")
    
    args = parser.parse_args()

    # change relative path to absolute path
    logdir = os.path.abspath(args.logdir)
    logdir = os.path.join(logdir, args.ckpt_name)
    os.makedirs(logdir, exist_ok=True)
    print(f"Logging to {logdir}")
    args.logdir = logdir

    # a dummy dataloader
    dataloader = DataLoader(
        range(args.steps_per_epoch * args.batch_size),
        batch_size=args.batch_size,
        shuffle=True
    )

    # create model
    model = PlFMFactorVAE(args)

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
    # swa = StochasticWeightAveraging(swa_lrs=1e-2)
    callbacks = [best_checkpoint_callback, last_checkpoint_callback] #, swa]

    # create logger
    logger = WandbLogger(
        name=args.ckpt_name,
        project="fm_vae",
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
        limit_train_batches=args.steps_per_epoch,
    )

    # save hyperparameters
    hyperparams = dict(
        sr=args.sr,
        length_samps=args.length_samps,
        n_fft=args.n_fft,
        f_min=args.f_min,
        f_max=args.f_max,
        n_mels=args.n_mels,
        power=args.power,
        normalized=args.normalized,
        max_harm_ratio=args.max_harm_ratio,
        max_mod_idx=args.max_mod_idx,
        latent_size=args.latent_size,
        encoder_channels=args.encoder_channels,
        encoder_kernels=args.encoder_kernels,
        encoder_n_res_block=args.encoder_n_res_block,
        encoder_n_res_channel=args.encoder_n_res_channel,
        decoder_features=args.decoder_features,
        decoder_n_res_block=args.decoder_n_res_block,
        decoder_n_res_features=args.decoder_n_res_features,
        d_hidden_size=args.d_hidden_size,
        d_num_layers=args.d_num_layers,
        batch_size=args.batch_size,
        warmup_epochs=args.warmup_epochs,
        param_loss_weight_start=args.param_loss_weight_start,
        param_loss_weight_end=args.param_loss_weight_end,
        param_loss_weight_ramp_start_epoch=args.param_loss_weight_ramp_start_epoch,
        param_loss_weight_ramp_end_epoch=args.param_loss_weight_ramp_end_epoch,
        recon_weight=args.recon_weight,
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
        steps_per_epoch=args.steps_per_epoch,
        ckpt_path=args.ckpt_path,
        ckpt_name=args.ckpt_name,
        logdir=logdir,
        comment=args.comment,
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
        trainer.fit(model, train_dataloaders=dataloader, ckpt_path=resume_path)
    else:
        trainer.fit(model, train_dataloaders=dataloader)


if __name__ == "__main__":
    fix_seed = 2025
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    torch.set_float32_matmul_precision('high')
    # os.environ["WANDB_START_METHOD"] = "thread"
    main()