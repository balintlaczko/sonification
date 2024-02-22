import argparse
import os
import torch
from torch.utils.data import DataLoader
import random
import numpy as np
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from sonification.models.models import PlFactorVAE1D
from sonification.datasets import Sinewave_dataset
import optuna
from optuna.integration import PyTorchLightningPruningCallback


def objective(trial: optuna.trial.Trial) -> float:

    class Args:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    # We optimize the number of layers, hidden units in each layer and dropouts.
    # n_layers = trial.suggest_int("n_layers", 3, 5)
    n_layers = 5
    layers_channels = [
        trial.suggest_categorical("layers_channels_l{}".format(i), [256, 512, 1024, 2048]) for i in range(n_layers)
    ]
    d_hidden_size = trial.suggest_categorical(
        "d_hidden_size", [256, 512, 1024])
    d_num_layers = trial.suggest_int("d_num_layers", 3, 6)
    lr_vae = trial.suggest_float("lr_vae", 1e-6, 1e-2, log=True)
    lr_decay_vae = trial.suggest_float("lr_decay_vae", 0.9, 0.999)
    # lr_d = trial.suggest_float("lr_d", 1e-6, 1e-2, log=True)
    lr_d = lr_vae
    # lr_decay_d = trial.suggest_float("lr_decay_d", 0.9, 0.999)
    lr_decay_d = lr_decay_vae
    batch_size = trial.suggest_categorical(
        "batch_size", [256, 512, 1024])

    args = Args(
        # root path
        root_path='./experiments/lsm_paper/',
        # dataset
        csv_path='sinewave.csv',
        img_size=64,
        # model
        in_channels=1,
        latent_size=2,
        layers_channels=layers_channels,
        d_hidden_size=d_hidden_size,
        d_num_layers=d_num_layers,
        # training
        train_epochs=31,
        batch_size=batch_size,
        lr_vae=lr_vae,
        lr_decay_vae=lr_decay_vae,
        lr_d=lr_d,
        lr_decay_d=lr_decay_d,
        kld_weight=0.02,
        tc_weight=0.02,
        l1_weight=0.0,
        # checkpoint & logging
        ckpt_path='./ckpt/sinewave_fvae-optuna-kld_0_02-tc_0_02',
        ckpt_name=f'sinewave_fvae-optuna-kld_0_02-tc_0_02_{str(trial.number).zfill(3)}',
        logdir='./logs/sinewave_fvae-optuna-kld_0_02-tc_0_02',
        plot_interval=15,
    )

    # create train and val datasets and loaders
    sinewave_ds_train = Sinewave_dataset(
        root_path=args.root_path, csv_path=args.csv_path, flag="train")
    sinewave_ds_val = Sinewave_dataset(
        root_path=args.root_path, csv_path=args.csv_path, flag="val", scaler=sinewave_ds_train.scaler)

    train_loader = DataLoader(
        sinewave_ds_train, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(
        sinewave_ds_val, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True)

    # create the model
    args.train_scaler = sinewave_ds_train.scaler
    model = PlFactorVAE1D(args)

    # logger callbacks
    tensorboard_logger = TensorBoardLogger(
        save_dir=args.logdir, name=args.ckpt_name)

    trainer = Trainer(
        max_epochs=args.train_epochs,
        enable_checkpointing=False,
        callbacks=[PyTorchLightningPruningCallback(
            trial, monitor="val_vae_recon_loss")],
        logger=tensorboard_logger,
        log_every_n_steps=1,
    )

    hyperparameters = dict(
        n_layers=n_layers,
        layers_channels=layers_channels,
        d_hidden_size=d_hidden_size,
        d_num_layers=d_num_layers,
        lr_vae=lr_vae,
        lr_decay_vae=lr_decay_vae,
        lr_d=lr_d,
        lr_decay_d=lr_decay_d,
        batch_size=batch_size,
    )
    trainer.logger.log_hyperparams(hyperparameters)

    trainer.fit(model=model, train_dataloaders=train_loader,
                val_dataloaders=val_loader)

    return trainer.callback_metrics["val_vae_recon_loss"].item()


if __name__ == "__main__":
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    torch.set_float32_matmul_precision("high")

    pruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=1000)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
