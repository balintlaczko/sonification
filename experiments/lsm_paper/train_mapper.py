import argparse
import os
import torch
from torch.utils.data import DataLoader
import random
import numpy as np
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from sonification.models.models import PlImgFactorVAE, PlSineFactorVAE, PlMapper
from sonification.datasets import White_Square_dataset_2 as White_Square_dataset
import wandb


def main():
    parser = argparse.ArgumentParser()

    # model params
    parser.add_argument('--in_features', type=int, default=2, help='input size')
    parser.add_argument('--out_features', type=int, default=2, help='output size')
    parser.add_argument('--hidden_layers_features', type=int, nargs='*', default=[8, 16, 32, 64, 128, 256, 128, 64, 32, 16, 8], help='the size of the hidden layers')
    parser.add_argument("--d_hidden_size", type=int, default=16)
    parser.add_argument("--d_num_layers", type=int, default=3)

    # training params
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument('--train_epochs', type=int, default=10000000, help='number of training epochs')
    # locality loss params
    parser.add_argument('--locality_loss_type', type=str, default='l1', help='locality loss type: l1 or mse')
    parser.add_argument('--locality_weight', type=float, default=1)
    # mmd loss params
    parser.add_argument('--mmd_weight', type=float, default=50)
    # cycle consistency loss params
    parser.add_argument('--cycle_consistency_loss_type', type=str, default='l1', help='cycle consistency loss type: l1 or mse')
    parser.add_argument('--cycle_consistency_weight_start', type=float, default=0)
    parser.add_argument('--cycle_consistency_weight_end', type=float, default=1)
    parser.add_argument('--cycle_consistency_ramp_start_epoch', type=int, default=1000, help='cycle consistency start epoch')
    parser.add_argument('--cycle_consistency_ramp_end_epoch', type=int, default=2000)
    # tc loss params
    parser.add_argument('--tc_weight_max', type=float, default=2, help='tc weight at the end of the warmup')
    parser.add_argument('--tc_weight_min', type=float, default=2, help='tc weight at the start of the warmup')
    parser.add_argument('--tc_start_epoch', type=int, default=0, help='the epoch at which to start the tc warmup from tc_weight_min to tc_weight_max')
    parser.add_argument('--tc_warmup_epochs', type=int, default=1, help='the number of epochs to warmup the tc weight')

    # optimizer params
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--lr_decay", type=float, default=0.85)
    parser.add_argument("--lr_d", type=float, default=0.000005)
    parser.add_argument("--lr_decay_d", type=float, default=0.85)

    # image model
    parser.add_argument('--img_model_ckpt_path', type=str, default='./ckpt/squares_vae/imv_new_v21_bkp/imv_new_v21_last_epoch=203.ckpt', help='image model checkpoint path')

    # audio model
    parser.add_argument('--audio_model_ckpt_path', type=str, default='./ckpt/sine_vae/imv_new_v35/imv_new_v35_last_epoch=1171.ckpt', help='sound model checkpoint path')

    # checkpoint & logging
    parser.add_argument('--ckpt_path', type=str, default='./ckpt/mapper', help='checkpoint path')
    parser.add_argument('--ckpt_name', type=str, default='imv_new_v7', help='checkpoint name')
    parser.add_argument('--logdir', type=str, default='./logs/mapper', help='log directory')

    # quick comment
    parser.add_argument('--comment', type=str, default='adding tc loss to mapped', help='add a comment if needed')

    args = parser.parse_args()

    # change relative path to absolute path
    logdir = os.path.abspath(args.logdir)
    logdir = os.path.join(logdir, args.ckpt_name)
    os.makedirs(logdir, exist_ok=True)
    print(f"Logging to {logdir}")
    args.logdir = logdir

    # load in model checkpoint and extract saved args
    ckpt = torch.load(args.img_model_ckpt_path, map_location='cpu')
    in_model_args = ckpt["hyper_parameters"]['args']

    # create in model with args and load state dict
    in_model = PlImgFactorVAE(in_model_args)
    in_model.load_state_dict(ckpt['state_dict'])
    in_model.eval()
    print("In model loaded")

    # load out model checkpoint and extract saved args
    ckpt = torch.load(args.audio_model_ckpt_path, map_location='cpu')
    out_model_args = ckpt["hyper_parameters"]['args']

    # create out model with args and load state dict
    out_model = PlSineFactorVAE(out_model_args)
    out_model.load_state_dict(ckpt['state_dict'])
    out_model.eval()
    print("Out model loaded")

    # create mapper model where args also references the in and out models
    args.in_model = in_model
    args.out_model = out_model
    model = PlMapper(args)
    model.in_model.requires_grad_(False)
    model.out_model.requires_grad_(False)
    print("Mapper model created")

    # create image dataset
    dataset = White_Square_dataset(img_size=in_model_args.img_size, square_size=in_model_args.square_size)

    # create image dataloader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=16, persistent_workers=True)
    print("Image dataloader created")

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
        project="mapper",
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
        log_every_n_steps=10,
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
