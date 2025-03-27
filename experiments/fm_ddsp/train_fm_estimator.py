import argparse
import os
import torch
import random
import numpy as np
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging
from lightning.pytorch.loggers import WandbLogger
from sonification.models.models import PlFMParamEstimator
from sonification.utils.misc import midi2frequency
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser()

    # audio params
    parser.add_argument("--sr", type=int, default=48000)
    parser.add_argument("--length_s", type=float, default=0.25)
    parser.add_argument("--n_fft", type=int, default=4096)
    parser.add_argument("--f_min", type=float, default=midi2frequency(38))
    parser.add_argument("--f_max", type=float, default=midi2frequency(86))
    parser.add_argument("--n_mels", type=int, default=512)
    parser.add_argument("--power", type=float, default=1)
    parser.add_argument("--normalized", type=int, default=1)
    parser.add_argument("--max_harm_ratio", type=int, default=6)
    parser.add_argument("--max_mod_idx", type=int, default=6)
    # model params
    parser.add_argument("--latent_size", type=int, default=128)
    parser.add_argument("--encoder_kernels", type=int, nargs='*', default=[4, 16])
    parser.add_argument("--n_res_block", type=int, default=24)
    parser.add_argument("--n_res_channel", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=3)
    # training params
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--lr_decay", type=float, default=0.75)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--train_epochs", type=int, default=10000)
    parser.add_argument("--steps_per_epoch", type=int, default=1000)
    parser.add_argument("--param_loss_weight_start", type=int, default=100)
    parser.add_argument("--param_loss_weight_end", type=int, default=100)
    parser.add_argument("--param_loss_weight_ramp_start_epoch", type=int, default=0)
    parser.add_argument("--param_loss_weight_ramp_end_epoch", type=int, default=1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/fm_ddsp")
    parser.add_argument("--ckpt_name", type=str, default="grad_test_19")
    parser.add_argument("--logdir", type=str, default="./logs/fm_ddsp")
    parser.add_argument("--comment", type=str, default="warmup10, param100, res64, groupnorm")
    
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
    model = PlFMParamEstimator(args)

    # checkpoint callbacks
    checkpoint_path = os.path.join(args.ckpt_path, args.ckpt_name)
    best_checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        dirpath=checkpoint_path,
        filename=args.ckpt_name + "_best_{epoch:02d}-{train_loss:.4f}",
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
        project="fm_ddsp",
        save_dir=logdir,
        offline=True,
        )
    # logger.watch(model, log='all')

    # create trainer
    trainer = Trainer(
        max_epochs=args.train_epochs,
        enable_checkpointing=True,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=250,
        limit_train_batches=args.steps_per_epoch,
    )

    # save hyperparameters
    hyperparams = dict(
        sr=args.sr,
        length_s=args.length_s,
        n_fft=args.n_fft,
        f_min=args.f_min,
        f_max=args.f_max,
        n_mels=args.n_mels,
        power=args.power,
        normalized=args.normalized,
        latent_size=args.latent_size,
        encoder_kernels=args.encoder_kernels,
        n_res_block=args.n_res_block,
        n_res_channel=args.n_res_channel,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_decay=args.lr_decay,
        warmup_epochs=args.warmup_epochs,
        train_epochs=args.train_epochs,
        steps_per_epoch=args.steps_per_epoch,
        param_loss_weight_start=args.param_loss_weight_start,
        param_loss_weight_end=args.param_loss_weight_end,
        param_loss_weight_ramp_start_epoch=args.param_loss_weight_ramp_start_epoch,
        param_loss_weight_ramp_end_epoch=args.param_loss_weight_ramp_end_epoch,
        max_harm_ratio=args.max_harm_ratio,
        max_mod_idx=args.max_mod_idx,
        ckpt_path=args.ckpt_path,
        ckpt_name=args.ckpt_name,
        logdir=logdir,
        comment=args.comment
    )
    trainer.logger.log_hyperparams(hyperparams)

    # train model
    trainer.fit(model, train_dataloaders=dataloader)


if __name__ == "__main__":
    fix_seed = 2025
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    torch.set_float32_matmul_precision('high')
    # os.environ["WANDB_START_METHOD"] = "thread"
    main()