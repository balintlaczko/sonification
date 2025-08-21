import argparse
import os
import torch
import random
import numpy as np
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from sonification.models.models import PlFMFactorVAE
from sonification.datasets import FMTripletDataset
from torch.utils.data import DataLoader
import wandb
import torch.multiprocessing as mp

def main():
    parser = argparse.ArgumentParser()

    # audio/data params
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
    parser.add_argument("--contrastive_dataset_path", type=str, default="data/fm_triplet/fm-synth-study-2025-08-20_15_46_36.json")

    # model params
    parser.add_argument("--latent_size", type=int, default=16)
    parser.add_argument("--encoder_channels", type=int, default=128)
    parser.add_argument("--encoder_kernels", type=int, nargs='*', default=[3, 5])
    parser.add_argument("--encoder_n_res_block", type=int, default=24)
    parser.add_argument("--encoder_n_res_channel", type=int, default=64)
    parser.add_argument("--decoder_features", type=int, default=256)
    parser.add_argument("--decoder_n_res_block", type=int, default=8)
    parser.add_argument("--decoder_n_res_features", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--d_hidden_size", type=int, default=128)
    parser.add_argument("--d_num_layers", type=int, default=5)

    # training params
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    # reconstruction loss params
    parser.add_argument("--param_loss_weight_start", type=int, default=150)
    parser.add_argument("--param_loss_weight_end", type=int, default=150)
    parser.add_argument("--param_loss_weight_ramp_start_epoch", type=int, default=0)
    parser.add_argument("--param_loss_weight_ramp_end_epoch", type=int, default=1)
    parser.add_argument("--recon_weight", type=int, default=1)
    parser.add_argument('--target_recon_loss', type=float, default=0.01, help='target recon loss to keep in case of dynamic kld')
    # kld loss params
    parser.add_argument('--dynamic_kld', type=int, default=1, help='non-zero will use dynamic kld')
    parser.add_argument('--kld_weight_max', type=float, default=1, help='kld weight at the end of the warmup')
    parser.add_argument('--kld_weight_min', type=float, default=0.1, help='kld weight at the start of the warmup')
    parser.add_argument('--kld_start_epoch', type=int, default=1000, help='the epoch at which to start the kld warmup from kld_weight_min to kld_weight_max')
    parser.add_argument('--kld_warmup_epochs', type=int, default=10000, help='the number of epochs to warmup the kld weight')
    # tc loss params
    parser.add_argument('--tc_weight_max', type=float, default=10, help='tc weight at the end of the warmup')
    parser.add_argument('--tc_weight_min', type=float, default=0, help='tc weight at the start of the warmup')
    parser.add_argument('--tc_start_epoch', type=int, default=0, help='the epoch at which to start the tc warmup from tc_weight_min to tc_weight_max')
    parser.add_argument('--tc_warmup_epochs', type=int, default=1, help='the number of epochs to warmup the tc weight')
    # contrastive loss params
    parser.add_argument("--contrastive_regularization", type=int, default=1, help="Use contrastive regularization (default: 0 = no contrastive regularization)")
    parser.add_argument("--contrastive_weight_max", type=float, default=0.1, help="Maximum weight for contrastive loss")
    parser.add_argument("--contrastive_weight_min", type=float, default=0.01, help="Minimum weight for contrastive loss")
    parser.add_argument("--contrastive_start_epoch", type=int, default=0, help="The epoch at which to start the contrastive warmup from contrastive_weight_min to contrastive_weight_max")
    parser.add_argument("--contrastive_warmup_epochs", type=int, default=100, help="The number of epochs to warmup the contrastive weight")
    # optimizer params
    parser.add_argument("--lr_vae", type=float, default=0.001)
    parser.add_argument("--lr_decay_vae", type=float, default=0.75)
    parser.add_argument("--lr_d", type=float, default=0.0001)
    parser.add_argument("--lr_decay_d", type=float, default=0.75)
    parser.add_argument("--train_epochs", type=int, default=100000)
    parser.add_argument("--steps_per_epoch", type=int, default=100)
    # checkpointing & logging
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/fm_vae")
    parser.add_argument("--ckpt_name", type=str, default="imv_v4")
    parser.add_argument("--logdir", type=str, default="./logs/fm_vae")
    parser.add_argument("--comment", type=str, default="like v3.2 but with contrastive regularization")
    
    args = parser.parse_args()

    # change relative path to absolute path
    logdir = os.path.abspath(args.logdir)
    logdir = os.path.join(logdir, args.ckpt_name)
    os.makedirs(logdir, exist_ok=True)
    print(f"Logging to {logdir}")
    args.logdir = logdir

    # a dummy dataloader for the fm synth data
    dataloader = DataLoader(
        range(args.steps_per_epoch * args.batch_size),
        batch_size=args.batch_size,
        shuffle=True
    )
    if args.contrastive_regularization:
        # the triplet data for contrastive regularization
        fm_triplet_dataset = FMTripletDataset(
            json_path=args.contrastive_dataset_path,
            # sr=args.sr,
            # n_samples=args.length_samps,
            # n_fft=args.n_fft,
            # f_min=args.f_min,
            # f_max=args.f_max,
            # n_mels=args.n_mels,
            # power=args.power,
            # normalized=args.normalized > 0,
            device='cpu'
            )
        fm_triplet_dataloader = DataLoader(
            fm_triplet_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=24,
            persistent_workers=True,
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
    hyperparams = vars(args).copy()
    trainer.logger.log_hyperparams(hyperparams)

    # train model
    train_dataloaders = [dataloader, fm_triplet_dataloader] if args.contrastive_regularization > 0 else dataloader
    try:
        resume_path = os.listdir(checkpoint_path)
    except FileNotFoundError:
        resume_path = None
    if resume_path:
        resume_path = sorted(resume_path)
        resume_path = os.path.join(checkpoint_path, resume_path[-1])
        print(f"Resuming from {resume_path}")
        trainer.fit(model, train_dataloaders=train_dataloaders, ckpt_path=resume_path)
    else:
        trainer.fit(model, train_dataloaders=train_dataloaders)


if __name__ == "__main__":
    # mp.set_start_method('spawn', force=True)
    fix_seed = 2025
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    torch.set_float32_matmul_precision('high')
    # os.environ["WANDB_START_METHOD"] = "thread"
    main()