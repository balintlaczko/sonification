import argparse
import os
import torch
from torch.utils.data import DataLoader
import random
import numpy as np
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from sonification.models.models import PlFactorVAE
from sonification.datasets import White_Square_dataset
import optuna
from optuna.integration import PyTorchLightningPruningCallback


def objective(trial: optuna.trial.Trial) -> float:

    class Args:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    # We optimize the number of layers, hidden units in each layer and dropouts.
    n_layers = 3
    # n_layers = trial.suggest_int("n_layers", 1, 3)
    layers_channels = [
        trial.suggest_categorical("layers_channels_l{}".format(i), [256, 512, 1024, 2048]) for i in range(n_layers)
    ]
    d_hidden_size = trial.suggest_categorical(
        "d_hidden_size", [512, 1024])
    d_num_layers = trial.suggest_int("d_num_layers", 4, 6)
    lr_vae = trial.suggest_float("lr_vae", 1e-3, 1e-1, log=True)
    lr_decay_vae = trial.suggest_float("lr_decay_vae", 0.9, 0.999)
    lr_d = trial.suggest_float("lr_d", 1e-3, 1e-1, log=True)
    lr_decay_d = trial.suggest_float("lr_decay_d", 0.9, 0.999)

    args = Args(
        # root path
        root_path='./experiments/lsm_paper/',
        # dataset
        csv_path='white_squares_xy_16_4.csv',
        img_size=16,
        square_size=4,
        # model
        in_channels=1,
        latent_size=2,
        layers_channels=layers_channels,
        d_hidden_size=d_hidden_size,
        d_num_layers=d_num_layers,
        # training
        train_epochs=1001,
        batch_size=144,
        dataset_size=144,
        lr_vae=lr_vae,
        lr_decay_vae=lr_decay_vae,
        lr_d=lr_d,
        lr_decay_d=lr_decay_d,
        kld_weight=0.1,
        tc_weight=10.0,
        l1_weight=0.0,
        mmd_prior_distribution="gaussian",
        mmd_weight=0,
        # checkpoint & logging
        ckpt_path='./ckpt/factorvae-optuna-kld_0_1',
        ckpt_name=f'factorvae-optuna-kld_0_1_{str(trial.number).zfill(3)}',
        logdir='./logs/factorvae-optuna-kld_0_1',
        plot_interval=500,
    )
    # manually calc & add weight for active pixels (assuming a sparse binary image)
    args.onpix_weight = args.img_size / args.square_size

    # create train dataset
    train_dataset = White_Square_dataset(
        root_path=args.root_path,
        csv_path=args.csv_path,
        img_size=args.img_size,
        square_size=args.square_size,
        flag="all")

    # create train and val dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # create the model
    model = PlFactorVAE(args)

    # logger callbacks
    tensorboard_logger = TensorBoardLogger(
        save_dir=args.logdir, name=args.ckpt_name)

    trainer = Trainer(
        max_epochs=args.train_epochs,
        enable_checkpointing=False,
        callbacks=[PyTorchLightningPruningCallback(
            trial, monitor="vae_recon_loss")],
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
    )
    trainer.logger.log_hyperparams(hyperparameters)

    trainer.fit(model=model, train_dataloaders=train_loader)

    return trainer.callback_metrics["vae_recon_loss"].item()


if __name__ == "__main__":
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    torch.set_float32_matmul_precision("high")

    pruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
