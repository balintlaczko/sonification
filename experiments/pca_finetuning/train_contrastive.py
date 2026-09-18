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
    parser.add_argument("--resample_base", type=int, default=960)  # divides 48k well
    parser.add_argument("--length_samps", type=int, default=8192)
    parser.add_argument("--n_fft", type=int, default=4096)
    parser.add_argument("--f_min", type=float, default=20)
    parser.add_argument("--f_max", type=float, default=12000)
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--power", type=float, default=1) # not used now
    parser.add_argument("--normalized", type=int, default=1)
    parser.add_argument("--max_harm_ratio", type=int, default=14)
    parser.add_argument("--max_mod_idx", type=int, default=14)
    parser.add_argument("--num_views", type=int, default=8)  # number of views for contrastive learning
    parser.add_argument("--apply_transposition", type=int, default=1)
    parser.add_argument("--transposition_range", type=float, default=2.0)  # range for pitch transposition: 3 == -3 —> +3
    parser.add_argument("--noise_max_amp", type=float, default=0.01)  # max amplitude for noise augmentation
    # model params
    parser.add_argument("--mode", type=str, default="byol")  # dino or byol
    parser.add_argument("--latent_size", type=int, default=8)
    parser.add_argument("--predictor_hidden_layers_features", type=int, nargs='*', default=[32, 64, 32])
    parser.add_argument("--center_momentum", type=float, default=0.996)
    parser.add_argument("--ema_decay_min", type=float, default=0.99)
    parser.add_argument("--ema_decay_max", type=float, default=0.999)
    parser.add_argument("--ema_decay_ramp_start_epoch", type=int, default=1000)
    parser.add_argument("--ema_decay_ramp_num_epochs", type=int, default=3000)
    parser.add_argument("--student_temperature", type=float, default=0.1)
    parser.add_argument("--teacher_temperature_min", type=float, default=0.04)
    parser.add_argument("--teacher_temperature_max", type=float, default=0.07)
    parser.add_argument("--teacher_temperature_ramp_start_epoch", type=int, default=0)
    parser.add_argument("--teacher_temperature_ramp_num_epochs", type=int, default=3000)
    parser.add_argument("--encoder_channels", type=int, default=64)
    parser.add_argument("--encoder_kernels", type=int, nargs='*', default=[3, 5])
    parser.add_argument("--encoder_n_res_block", type=int, default=8)
    parser.add_argument("--encoder_n_res_channel", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    # training params
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--lr_decay", type=float, default=0.75)
    parser.add_argument("--train_epochs", type=int, default=100000)
    parser.add_argument("--steps_per_epoch", type=int, default=100)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/fm_embedder")
    parser.add_argument("--ckpt_name", type=str, default="byol_v1.0")
    parser.add_argument("--logdir", type=str, default="./logs/fm_embedder")
    parser.add_argument("--comment", type=str, default="checking implementation")
    
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