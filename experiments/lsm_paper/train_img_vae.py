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

def main():
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser()

    # root path
    parser.add_argument('--root_path', type=str, default='./experiments/lsm_paper/', help='root path')

    # dataset
    parser.add_argument('--csv_path', type=str, default='white_squares_xy_64_4.csv', help='csv path')
    parser.add_argument('--img_size', type=int, default=64, help='image size')
    parser.add_argument('--square_size', type=int, default=4, help='size of the square')

    # model
    parser.add_argument('--latent_size', type=int, default=2, help='latent size')
    parser.add_argument('--layers_channels', type=int, nargs='*', default=[32, 32, 64, 64],
                        help='channels for the layers')
    parser.add_argument('--d_hidden_size', type=int, default=1000, help='mlp hidden size')
    parser.add_argument('--d_num_layers', type=int, default=6, help='mlp number of layers')

    # training
    parser.add_argument('--train_epochs', type=int, default=10000, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr_vae', type=float, default=1e-4, help='learning rate for the vae')
    parser.add_argument('--lr_d', type=float, default=1e-4, help='learning rate for the discriminator')
    parser.add_argument('--kld_weight', type=float, default=1, help='kld weight')
    parser.add_argument('--tc_weight', type=float, default=40, help='tc weight')
    parser.add_argument('--train_steps_limit', type=int, default=-1, help='train steps limit. -1 means no limit')
    parser.add_argument('--val_steps_limit', type=int, default=-1, help='validation steps limit. -1 means no limit')

    # GPU
    parser.add_argument('--num_devices', type=int, nargs='*', default=[0],
                        help='number of gpus to use. if list it will be the device ids to use')
    
    # checkpoint & logging
    parser.add_argument('--ckpt_path', type=str, default='./ckpt/white_squares_fvae', help='checkpoint path')
    parser.add_argument('--ckpt_name', type=str, default='v1', help='checkpoint name')
    parser.add_argument('--resume_ckpt_path', type=str, default=None,)
    parser.add_argument('--logdir', type=str, default='./logs/white_squares_fvae', help='log directory')

    # quick comment
    parser.add_argument('--comment', type=str, default='', help='add a comment if needed')

    args = parser.parse_args()

    # create train and val datasets
    train_dataset = White_Square_dataset(
        root_path=args.root_path, 
        csv_path=args.csv_path,
        img_size=args.img_size,
        square_size=args.square_size,
        flag="train")
    val_dataset = White_Square_dataset(
        root_path=args.root_path, 
        csv_path=args.csv_path,
        img_size=args.img_size,
        square_size=args.square_size,
        flag="val")
    
    # create train and val dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # create the model
    model = PlFactorVAE(
        in_channels=1,
        latent_size=args.latent_size,
        layers_channels=args.layers_channels,
        input_size=args.img_size,
        d_hidden_size=args.d_hidden_size,
        d_num_layers=args.d_num_layers,
        lr_vae=args.lr_vae,
        lr_d=args.lr_d,
        kld_weight=args.kld_weight,
        tc_weight=args.tc_weight
    )

    # checkpoint callbacks
    checkpoint_path = os.path.join(args.ckpt_path, args.ckpt_name)
    val_checkpoint_callback = ModelCheckpoint(
        monitor="val_vae_loss",
        dirpath=checkpoint_path,
        filename=args.ckpt_name + "_val_{epoch:02d}-{val_loss:.4f}",
        save_top_k=10,
        mode="min",
    )
    last_checkpoint_callback = ModelCheckpoint(
        monitor="epoch",
        dirpath=checkpoint_path,
        filename=args.ckpt_name + "_last_{epoch:02d}",
        save_top_k=1,
        mode="max",
    )

    # logger callbacks
    csv_logger = CSVLogger(save_dir=args.logdir, name=args.ckpt_name)
    tensorboard_logger = TensorBoardLogger(save_dir=args.logdir, name=args.ckpt_name)

    train_steps_limit = args.train_steps_limit if args.train_steps_limit > 0 else None
    val_steps_limit = args.val_steps_limit if args.val_steps_limit > 0 else None

    # trainer_strategy = "ddp_find_unused_parameters_true" if len(args.num_devices) != 1 else "auto"
    trainer_strategy = "auto"
    trainer = Trainer(
        strategy=trainer_strategy,
        # devices=args.num_devices, 
        # accelerator="gpu",
        max_epochs=args.train_epochs, 
        enable_checkpointing=True, 
        limit_train_batches=train_steps_limit, 
        limit_val_batches=val_steps_limit, 
        callbacks=[val_checkpoint_callback, last_checkpoint_callback],
        logger=[csv_logger, tensorboard_logger],
    )
    
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.resume_ckpt_path)



if __name__ == "__main__":
    main()