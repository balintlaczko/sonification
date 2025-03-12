import argparse
import os
import torch
import random
import numpy as np
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from sonification.models.models import PlFMParamEstimator
from sonification.utils.misc import midi2frequency
from torch.utils.data import DataLoader
import optuna
from optuna.integration import PyTorchLightningPruningCallback

def objective(trial: optuna.trial.Trial) -> float:

    # audio params
    length_s = trial.suggest_categorical("length_s", [0.25, 0.5, 1.0])
    n_fft = trial.suggest_categorical("n_fft", [1024, 2048, 4096])
    n_mels = trial.suggest_categorical("n_mels", [128, 256, 512])
    power = trial.suggest_categorical("power", [1, 0.5, 2])
    # model params
    latent_size = trial.suggest_categorical("latent_size", [32, 64, 128, 256])
    encoder_kernel_1 = trial.suggest_int("encoder_kernel_1", 1, 8)
    encoder_kernel_2 = trial.suggest_int("encoder_kernel_2", 8, 16)
    n_res_block = trial.suggest_int("n_res_block", 1, 32)
    n_res_channel = trial.suggest_categorical("n_res_channel", [8, 16, 32, 64])
    hidden_dim = trial.suggest_categorical("hidden_dim", [16, 32, 64, 128])
    num_layers = trial.suggest_int("num_layers", 3, 8)
    # training params
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    param_loss_weight = trial.suggest_float("param_loss_weight", 0, 100)

    class Args:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    args = Args(
        sr=48000,
        length_s=length_s,
        n_fft=n_fft,
        f_min=midi2frequency(38),
        f_max=midi2frequency(86),
        n_mels=n_mels,
        power=power,
        normalized=1,
        max_harm_ratio=6,
        max_mod_idx=6,
        latent_size=latent_size,
        encoder_kernels=[encoder_kernel_1, encoder_kernel_2],
        n_res_block=n_res_block,
        n_res_channel=n_res_channel,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        batch_size=batch_size,
        lr=lr,
        lr_decay=0.5,
        train_epochs=50,
        steps_per_epoch=1000,
        param_loss_weight=param_loss_weight,
        ckpt_path="./ckpt/fm_ddsp-optuna",
        ckpt_name=f"optuna_trial_{str(trial.number).zfill(4)}",
        logdir="./logs/fm_ddsp-optuna",
        comment=""
    )

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
        filename=args.ckpt_name + "_val_{epoch:02d}-{train_loss:.4f}",
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
    optuna_callback = PyTorchLightningPruningCallback(
        trial,
        monitor="mss_loss",
    )
    callbacks = [best_checkpoint_callback, last_checkpoint_callback, optuna_callback]

    # create logger
    logger = WandbLogger(
        name=args.ckpt_name,
        project="fm_ddsp",
        save_dir=args.logdir,
        )

    # create trainer
    trainer = Trainer(
        max_epochs=args.train_epochs,
        enable_checkpointing=True,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=1,
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
        train_epochs=args.train_epochs,
        steps_per_epoch=args.steps_per_epoch,
        param_loss_weight=args.param_loss_weight,
        max_harm_ratio=args.max_harm_ratio,
        max_mod_idx=args.max_mod_idx,
        ckpt_path=args.ckpt_path,
        ckpt_name=args.ckpt_name,
        logdir=args.logdir,
        comment=args.comment
    )
    trainer.logger.log_hyperparams(hyperparams)

    # train model
    trainer.fit(model, train_dataloaders=dataloader)

    return trainer.callback_metrics["mss_loss"].item()


if __name__ == "__main__":
    fix_seed = 2025
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    torch.set_float32_matmul_precision("high")

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=11)

    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=200)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))