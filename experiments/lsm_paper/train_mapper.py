import argparse
import os
import torch
from torch.utils.data import DataLoader
import random
import numpy as np
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from sonification.models.models import PlFactorVAE, PlFactorVAE1D, PlMapper
from sonification.datasets import White_Square_dataset, Sinewave_dataset


def main():
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--in_features', type=int,
                        default=2, help='input size')
    parser.add_argument('--out_features', type=int,
                        default=2, help='output size')
    parser.add_argument('--hidden_layers_features', type=int, nargs='*', default=[64, 128, 256, 512, 256, 128, 64],
                        help='the size of the hidden layers')

    # training
    parser.add_argument('--train_epochs', type=int,
                        default=50001, help='number of training epochs')
    # batch size is determined by the image dataset size, since it's small
    # parser.add_argument('--batch_size', type=int,
    #                     default=144, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.999)
    parser.add_argument('--locality_weight', type=float, default=1)
    parser.add_argument('--cycle_consistency_weight', type=float, default=4)
    parser.add_argument('--cycle_consistency_start', type=int,
                        default=300, help='cycle consistency start epoch')
    parser.add_argument('--cycle_consistency_warmup_epochs',
                        type=int, default=700)

    # image model
    parser.add_argument('--img_model_ckpt_path', type=str,
                        default='./ckpt/white_squares_fvae_opt/with_falloff-v12/with_falloff-v12_epoch=4807-val_loss=0.0000.ckpt', help='image model checkpoint path')

    # audio model
    parser.add_argument('--audio_model_ckpt_path', type=str,  # v27 is the previous best
                        default='./ckpt/sinewave_fvae-opt/opt-v41/opt-v41_last_epoch=50001.ckpt', help='sound model checkpoint path')

    # checkpoint & logging
    parser.add_argument('--ckpt_path', type=str,
                        default='./ckpt/mapper', help='checkpoint path')
    parser.add_argument('--ckpt_name', type=str,
                        default='mapper-64x2-v5', help='checkpoint name')
    parser.add_argument('--resume_ckpt_path', type=str,
                        default=None,)
    parser.add_argument('--logdir', type=str,
                        default='./logs/mapper', help='log directory')
    parser.add_argument('--plot_interval', type=int, default=10)

    # quick comment
    parser.add_argument('--comment', type=str, default='double cycle consistency weight, shorter ramp',
                        help='add a comment if needed')

    mapper_args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class Args:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    # load image model
    img_model_args = Args(
        # root path
        root_path='./experiments/lsm_paper/',
        # dataset
        csv_path='white_squares_xy_64_2.csv',
        img_size=64,
        square_size=2,
        # model
        in_channels=1,
        latent_size=2,
        layers_channels=[64, 128, 256, 512],
        d_hidden_size=512,
        d_num_layers=5,
        # training
        dataset_size=3844,
        recon_weight=20,
        kld_weight_max=0.02,
        kld_weight_min=0.002,
        kld_start_epoch=5000,
        kld_warmup_epochs=10000,
        tc_weight=6,
        onpix_weight=1,
        lr_vae=1e-2,
        lr_decay_vae=0.9999,
        lr_d=1e-2,
        lr_decay_d=0.9999,
        plot_interval=10,
    )

    ckpt_path = mapper_args.img_model_ckpt_path
    ckpt = torch.load(ckpt_path, map_location=device)
    img_model = PlFactorVAE(img_model_args).to(device)
    img_model.load_state_dict(ckpt['state_dict'])
    img_model.eval()

    # load audio model
    audio_model_args = Args(
        # root path
        root_path='./experiments/lsm_paper/',
        # dataset
        csv_path='sinewave.csv',
        img_size=64,
        # model
        in_channels=1,
        latent_size=2,
        layers_channels=[64, 128, 256, 512, 1024],
        d_hidden_size=512,
        d_num_layers=5,
        # training
        recon_weight=1,
        target_recon_loss=1e-3,
        dynamic_kld=1,
        kld_weight_max=0.01,
        kld_weight_min=0.001,
        kld_start_epoch=0,
        kld_warmup_epochs=1,
        tc_weight=6,
        tc_start=0,
        tc_warmup_epochs=1,
        l1_weight=0.0,
        lr_d=1e-2,
        lr_decay_d=0.9999,
        lr_decay_vae=0.9999,
        lr_vae=1e-2,
        # checkpoint & logging
        ckpt_path='./ckpt/sinewave_fvae-opt',
        ckpt_name='opt-v41',
        logdir='./logs/sinewave_fvae-opt',
        plot_interval=1,
    )

    # create train and val datasets and loaders
    sinewave_ds_train = Sinewave_dataset(
        root_path=audio_model_args.root_path, csv_path=audio_model_args.csv_path, flag="train")
    sinewave_ds_val = Sinewave_dataset(
        root_path=audio_model_args.root_path, csv_path=audio_model_args.csv_path, flag="val", scaler=sinewave_ds_train.scaler)

    # load model
    ckpt_path = mapper_args.audio_model_ckpt_path
    ckpt = torch.load(ckpt_path, map_location=device)
    audio_model_args.train_scaler = sinewave_ds_train.scaler
    audio_model = PlFactorVAE1D(audio_model_args).to(device)
    audio_model.load_state_dict(ckpt['state_dict'])
    audio_model.eval()

    # pre-render the audio model latent space (purely for plotting during training, not used in the model itself)
    batch_size = 256
    dataset = sinewave_ds_val
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, drop_last=False)
    audio_model_latent_space = torch.zeros(
        len(dataset), audio_model.args.latent_size).to(audio_model.device)
    for batch_idx, data in enumerate(loader):
        x, y = data
        x_recon, mean, logvar, z = audio_model.VAE(x.to(audio_model.device))
        z = z.detach()
        audio_model_latent_space[batch_idx *
                                 batch_size: batch_idx*batch_size + batch_size] = z

    # add missing args
    mapper_args.batch_size = img_model_args.dataset_size
    mapper_args.in_model = img_model
    mapper_args.out_model = audio_model
    mapper_args.out_latent_space = audio_model_latent_space

    # create model
    model = PlMapper(mapper_args)
    model.in_model.requires_grad_(False)
    model.out_model.requires_grad_(False)

    # create train dataset
    train_dataset = White_Square_dataset(
        root_path=img_model_args.root_path,
        csv_path=img_model_args.csv_path,
        img_size=img_model_args.img_size,
        square_size=img_model_args.square_size,
        flag="all")

    # create train dataloader
    train_loader = DataLoader(
        train_dataset, batch_size=mapper_args.batch_size, shuffle=True, drop_last=True, pin_memory=True)

    # checkpoint callbacks
    checkpoint_path = os.path.join(
        mapper_args.ckpt_path, mapper_args.ckpt_name)
    best_checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        dirpath=checkpoint_path,
        filename=mapper_args.ckpt_name + "_{epoch:02d}-{train_loss:.4f}",
        save_top_k=1,
        mode="min",
    )
    last_checkpoint_callback = ModelCheckpoint(
        monitor="epoch",
        dirpath=checkpoint_path,
        filename=mapper_args.ckpt_name + "_last_{epoch:02d}",
        save_top_k=1,
        mode="max",
    )

    # logger callbacks
    tensorboard_logger = TensorBoardLogger(
        save_dir=mapper_args.logdir, name=mapper_args.ckpt_name)

    trainer = Trainer(
        max_epochs=mapper_args.train_epochs,
        enable_checkpointing=True,
        callbacks=[best_checkpoint_callback, last_checkpoint_callback],
        logger=tensorboard_logger,
        log_every_n_steps=1,
    )

    # save hyperparameters
    hyperparams = dict(
        in_features=mapper_args.in_features,
        out_features=mapper_args.out_features,
        hidden_layers_features=mapper_args.hidden_layers_features,
        batch_size=mapper_args.batch_size,
        lr=mapper_args.lr,
        lr_decay=mapper_args.lr_decay,
        locality_weight=mapper_args.locality_weight,
        cycle_consistency_weight=mapper_args.cycle_consistency_weight,
        cycle_consistency_start=mapper_args.cycle_consistency_start,
        cycle_consistency_warmup_epochs=mapper_args.cycle_consistency_warmup_epochs,
        img_model_ckpt_path=mapper_args.img_model_ckpt_path,
        audio_model_ckpt_path=mapper_args.audio_model_ckpt_path,
        comment=mapper_args.comment
    )

    trainer.logger.log_hyperparams(hyperparams)

    # train the model
    trainer.fit(model=model, train_dataloaders=train_loader,
                ckpt_path=mapper_args.resume_ckpt_path)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    main()
