import argparse
import os
import torch
import random
import numpy as np
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from sonification.models.models import PlFMEmbedder
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
    parser.add_argument("--apply_transposition", type=int, default=1)
    # model params
    parser.add_argument("--latent_size", type=int, default=64)
    parser.add_argument("--center_momentum", type=float, default=0.9)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--student_temperature", type=float, default=0.1)
    parser.add_argument("--teacher_temperature", type=float, default=0.1)
    parser.add_argument("--encoder_channels", type=int, default=128)
    parser.add_argument("--encoder_kernels", type=int, nargs='*', default=[3, 5])
    parser.add_argument("--encoder_n_res_block", type=int, default=24)
    parser.add_argument("--encoder_n_res_channel", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    # training params
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_decay", type=float, default=0.75)
    parser.add_argument("--train_epochs", type=int, default=100000)
    parser.add_argument("--steps_per_epoch", type=int, default=100)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/fm_embedder")
    parser.add_argument("--ckpt_name", type=str, default="home_v2")
    parser.add_argument("--logdir", type=str, default="./logs/fm_embedder")
    parser.add_argument("--comment", type=str, default="discard logvar instead of adding to mu")
    
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

    # model
    model = PlFMEmbedder(args)
    model.create_shadow_model()

    # checkpoint callbacks
    checkpoint_path = os.path.join(args.ckpt_path, args.ckpt_name)
    best_checkpoint_callback = ModelCheckpoint(
        monitor="loss",
        dirpath=checkpoint_path,
        filename=args.ckpt_name + "_best_{epoch:02d}-{loss:.4f}",
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
        project="fm_embedder",
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
        apply_transposition=args.apply_transposition,
        center_momentum=args.center_momentum,
        ema_decay=args.ema_decay,
        student_temperature=args.student_temperature,
        teacher_temperature=args.teacher_temperature,
        latent_size=args.latent_size,
        encoder_channels=args.encoder_channels,
        encoder_kernels=args.encoder_kernels,
        encoder_n_res_block=args.encoder_n_res_block,
        encoder_n_res_channel=args.encoder_n_res_channel,
        dropout=args.dropout,
        lr=args.lr,
        lr_decay=args.lr_decay,
        warmup_epochs=args.warmup_epochs,
        train_epochs=args.train_epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        ckpt_path=args.ckpt_path,
        ckpt_name=args.ckpt_name,
        logdir=args.logdir,
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
    main()