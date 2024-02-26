# %%
# imports
from pythonosc import udp_client
from sonification.utils.matrix import square_over_bg
from pythonosc import osc_server
from pythonosc import dispatcher
from torchsummary import summary
import librosa
from torchaudio.functional import amplitude_to_DB, DB_to_amplitude
import torch
import numpy as np
import cv2
from sonification.utils import matrix
from sonification.utils.dsp import midi2frequency, db2amp
from torchaudio.transforms import MelSpectrogram, Loudness, InverseMelScale, GriffinLim
from sonification.models.ddsp import Sinewave
from scipy.io import wavfile as wav
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import torchyin
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from torch import nn
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import Trainer
from sonification.utils.tensor import permute_dims
from sonification.models.loss import kld_loss, recon_loss, MMDloss
import torch.nn.functional as F
from sonification.models.layers import LinearDiscriminator
from sonification.datasets import Sinewave_dataset, White_Square_dataset
from sonification.models.models import ConvVAE1D, PlFactorVAE1D, PlFactorVAE
from sklearn.neighbors import KDTree

# %%
root_path = "./experiments/lsm_paper"
csv_path = "sinewave.csv"
sr = 44100
samps = 16384*2
n_fft = 8192
f_min = 60
f_max = 1200
pad = 1
n_mels = 64
power = 1
norm = "slaney"
mel_scale = "slaney"


# %%
sinewave_gen = Sinewave(sr=sr)
# %%
in_tensor = torch.ones(1, samps) * 830.61
in_tensor.shape
# %%
sinewave = sinewave_gen(in_tensor)
sinewave.shape
# %%
# save to disk
path = "sinewave.wav"
wav.write(path, sr, sinewave[0].numpy())
# %%
# create mel spectrogram
mel_spectrogram = MelSpectrogram(
    sample_rate=sr,
    n_fft=n_fft,
    f_min=f_min,
    f_max=f_max,
    pad=pad,
    n_mels=n_mels,
    power=2,
    norm=norm,
    mel_scale=mel_scale
    # mel_scale="htk"
)
# %%
melspec = mel_spectrogram(sinewave)
# average time dimension
melspec_avg = melspec.mean(dim=2)
melspec_avg

# %%

melspec_avg_db = amplitude_to_DB(
    melspec_avg, multiplier=10, amin=1e-5, db_multiplier=20, top_db=80)
melspec_avg_db

# %%
melspec_avg_np = melspec_avg[0].numpy()
librosa_melspec_avg = librosa.power_to_db(
    melspec_avg_np, ref=1.0, amin=1e-5, top_db=80)
librosa_melspec_avg
# %%
# normalize tensor
melspec_avg_db_norm = (melspec_avg_db - melspec_avg_db.min()) / \
    (melspec_avg_db.max() - melspec_avg_db.min())
melspec_avg_db_norm
# %%
# plot the mel spectrogram
# melspec_avg_np = melspec_avg_db[0].unsqueeze(-1).numpy()
melspec_avg_np = melspec_avg[0].unsqueeze(-1).numpy()
# reverse the mel scale
# melspec_avg_np = np.flip(melspec_avg_np, axis=0)
# repeat the mel spectrogram to make it more visible
# melspec_avg_np = np.repeat(melspec_avg_np, 10, axis=1)
# transpose
melspec_avg_np = melspec_avg_np.T
fig, ax = plt.subplots()
cax = ax.matshow(melspec_avg_np, interpolation='nearest', cmap='viridis')
fig.colorbar(cax)
plt.show()
# %%
loud = Loudness(sample_rate=sr)
# %%
loud(sinewave*0.5)
# %%
for multiplier in [1, 0.5, 0.25]:
    loudness = loud(sinewave*multiplier)
    print(loudness)

# %%
for multiplier in [1, 0.5, 0.25]:
    mspec = mel_spectrogram(sinewave*multiplier)
    mspec_avg = mspec.mean(dim=2)
    mspec_avg_db = amplitude_to_DB(
        mspec_avg, multiplier=10, amin=1e-5, db_multiplier=20, top_db=80)
    print(mspec_avg_db.mean())
    print(mspec_avg_db.min(), mspec_avg_db.max())
# %%

# resynth chain: repeat -> inverse mel -> griffin lim -> waveform

melspec_avg_rep = melspec_avg.unsqueeze(-1).repeat(1, 1, 9)
print(melspec_avg_rep.shape)

inverse_mel = InverseMelScale(
    n_stft=n_fft//2+1,
    n_mels=n_mels,
    sample_rate=sr,
    f_min=f_min,
    f_max=f_max,
    norm=norm,
    mel_scale=mel_scale,
)

spec = inverse_mel(melspec_avg_rep)
# spec = inverse_mel(melspec)
print(spec.shape)

# %%
gl = GriffinLim(
    n_fft=n_fft,
    n_iter=32,
    power=2,
    momentum=0.99)

waveform = gl(spec)
print(waveform.shape)

# %%
# get pitch with torchyin
pitch = torchyin.estimate(waveform, sr)
pitch.mean()

# resynthesize sinewave based on pitch
in_tensor = torch.ones(1, samps) * pitch.mean()
sinewave_recon = sinewave_gen(in_tensor)
sinewave_recon.shape

# %%
# save reconstructed waveform to disk
path = "reconstructed.wav"
wav.write(path, sr, sinewave_recon[0].numpy())

# %%
sinewave_ds_train = Sinewave_dataset(flag="train")
sinewave_ds_val = Sinewave_dataset(flag="val", scaler=sinewave_ds_train.scaler)

# %%
x_1, x_2 = sinewave_ds_val[:12]
print(x_1.shape, x_2.shape)
# %%
# plot the two mel spectrograms
specs, _ = sinewave_ds_train[:100]
print(specs.shape)
specs_np = specs.squeeze(1).T.numpy()

fig, ax = plt.subplots()
cax = ax.matshow(specs_np, interpolation='nearest', cmap='viridis')
fig.colorbar(cax)
plt.show()

# %%
# test that shapes are correct
conv_vae = ConvVAE1D(
    in_channels=1, latent_size=2, layers_channels=[16, 32, 64, 128, 256], input_size=64)
# %%
test_input = torch.rand(128, 1, 64)
recon, mean, logvar, z = conv_vae(test_input)
recon.shape, mean.shape, logvar.shape, z.shape

# %%
conv_vae.eval()
specs, _ = sinewave_ds_train[:10]
print(specs.shape)
recon, mean, logvar, z = conv_vae(specs)
recon.shape, mean.shape, logvar.shape, z.shape
# %%
# plot reconstructions
recon_np = recon.detach().squeeze(1).numpy()
print(recon_np.shape)
# inverse transform with scaler
recon_np = sinewave_ds_train.scaler.inverse_transform(recon_np).T
fig, ax = plt.subplots()
cax = ax.matshow(recon_np, interpolation='nearest', cmap='viridis')
fig.colorbar(cax)
plt.show()

# %%
# set up test training cycle


class Args:
    def __init__(self):
        self.in_channels = 1
        self.img_size = 64
        self.latent_size = 2
        self.layers_channels = [1024, 1024, 1024, 1024, 1024]
        self.d_hidden_size = 512
        self.d_num_layers = 5
        self.train_epochs = 10
        self.batch_size = 512
        self.lr_vae = 1e-5
        self.lr_decay_vae = 0.99
        self.lr_d = 1e-5
        self.lr_decay_d = 0.99
        self.kld_weight = 0.01
        self.tc_weight = 0.01
        self.l1_weight = 0
        self.ckpt_path = "./ckpt/sinewave_fvae_test"
        self.ckpt_name = "test-v1"
        self.plot_interval = 1
        self.logdir = "./logs/sinewave_fvae_test"
        self.comment = "test"


args = Args()

# %%
# create train and val datasets and loaders
sinewave_ds_train = Sinewave_dataset(flag="train")
sinewave_ds_val = Sinewave_dataset(flag="val", scaler=sinewave_ds_train.scaler)

train_loader = DataLoader(
    sinewave_ds_train, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)
val_loader = DataLoader(
    sinewave_ds_val, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True)

# %%
# create the model
args.train_scaler = sinewave_ds_train.scaler
model = PlFactorVAE1D(args)

# %%
# checkpoint callbacks
checkpoint_path = os.path.join(args.ckpt_path, args.ckpt_name)
best_checkpoint_callback = ModelCheckpoint(
    monitor="val_vae_loss",
    dirpath=checkpoint_path,
    filename=args.ckpt_name + "_val_{epoch:02d}-{val_loss:.4f}",
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

# logger callbacks
tensorboard_logger = TensorBoardLogger(
    save_dir=args.logdir, name=args.ckpt_name)

# %%
trainer = Trainer(
    max_epochs=args.train_epochs,
    enable_checkpointing=True,
    callbacks=[best_checkpoint_callback, last_checkpoint_callback],
    logger=tensorboard_logger,
    log_every_n_steps=1,
)

# %%
# save hyperparameters
hyperparams = dict(
    in_channels=args.in_channels,
    img_size=args.img_size,
    latent_size=args.latent_size,
    layers_channels=args.layers_channels,
    d_hidden_size=args.d_hidden_size,
    d_num_layers=args.d_num_layers,
    batch_size=args.batch_size,
    lr_vae=args.lr_vae,
    lr_decay_vae=args.lr_decay_vae,
    lr_d=args.lr_d,
    lr_decay_d=args.lr_decay_d,
    kld_weight=args.kld_weight,
    tc_weight=args.tc_weight,
    l1_weight=args.l1_weight,
    comment=args.comment
)

trainer.logger.log_hyperparams(hyperparams)
# %%
# train the model
trainer.fit(model, train_loader, val_loader)

# %%


class LinearProjector(nn.Module):
    def __init__(self, in_features, out_features, hidden_layers_features=[64, 128, 256, 128, 64]):
        super(LinearProjector, self).__init__()

        layers = []

        # add first layer
        layers.extend([
            nn.Linear(in_features, hidden_layers_features[0]),
            nn.LeakyReLU(0.2, inplace=True),
        ])

        # add hidden layers
        for i in range(len(hidden_layers_features)-1):
            layers.extend([
                nn.Linear(hidden_layers_features[i],
                          hidden_layers_features[i+1]),
                nn.LeakyReLU(0.2, inplace=True),
            ])

        # add output layer
        layers.append(nn.Linear(hidden_layers_features[-1], out_features))

        self.linear_projector = nn.Sequential(*layers)

    def forward(self, x):
        return self.linear_projector(x)


# %%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LinearProjector(in_features=2, out_features=2, hidden_layers_features=[
                        64, 128, 256, 128, 64]).to(device)
summary(model, (42, 2))

# %%
# test input
x = torch.rand(10, 2).to(device)
output = model(x)
output.shape

# %%
# get normalized distance matrix between a batch of 2D points
test_latents = torch.rand(4, 2)
dist = torch.cdist(test_latents, test_latents)
print(dist)
dist_norm = dist / dist.max()
print(dist_norm)

# %%
# create Lightning module


class PlMapper(LightningModule):
    def __init__(self, args):
        super(PlMapper, self).__init__()
        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        # self.save_hyperparameters()

        # data params
        self.in_features = args.in_features
        self.out_features = args.out_features
        self.hidden_layers_features = args.hidden_layers_features

        # losses
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.locality_weight = args.locality_weight
        self.bounding_box_weight = args.bounding_box_weight
        self.bounding_box_weight_decay = args.bounding_box_weight_decay
        self.target_matching_weight = args.target_matching_weight
        self.target_matching_start = args.target_matching_start
        self.target_matching_warmup_epochs = args.target_matching_warmup_epochs
        self.cycle_consistency_weight = args.cycle_consistency_weight
        self.cycle_consistency_start = args.cycle_consistency_start
        self.cycle_consistency_warmup_epochs = args.cycle_consistency_warmup_epochs

        # learning rate
        self.lr = args.lr
        self.lr_decay = args.lr_decay

        # logging
        self.plot_interval = args.plot_interval
        self.args = args

        # models
        self.model = LinearProjector(
            in_features=self.in_features, out_features=self.out_features, hidden_layers_features=self.hidden_layers_features)
        self.in_model = args.in_model
        self.out_model = args.out_model
        self.in_model.eval()
        self.out_model.eval()
        self.out_latent_space = args.out_latent_space

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        self.model.train()
        self.in_model.eval()
        self.out_model.eval()
        epoch_idx = self.trainer.current_epoch

        # get the optimizer and scheduler
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()

        # get the batch
        x, _ = batch
        batch_size = x.shape[0]

        # encode with input model
        z_1 = self.in_model.encode(x)

        # project to output space
        z_2 = self.model(z_1)

        # decode with output model
        x_hat = self.out_model.decode(z_2)

        # LOSSES
        # locality loss
        # TODO: have to do it per axis, otherwise it just aligns them on a diagonal
        # or add another term that prevents collapsing into a single point
        # no, have to do per axis anyway since the axes have different meanings
        # dist_1 = torch.cdist(z_1, z_1)
        # dist_1_norm = dist_1 / dist_1.max()
        # dist_2 = torch.cdist(z_2, z_2)
        # dist_2_norm = dist_2 / dist_2.max()
        # locality_loss = self.l1(dist_2_norm, dist_1_norm)
        z_1_x = z_1[:, 0]
        z_1_y = z_1[:, 1]
        z_1_x_norm = (z_1_x - z_1_x.min()) / (z_1_x.max() - z_1_x.min())
        z_1_y_norm = (z_1_y - z_1_y.min()) / (z_1_y.max() - z_1_y.min())
        z_2_x = z_2[:, 0]
        z_2_y = z_2[:, 1]
        z_2_x_norm = (z_2_x - z_2_x.min()) / (z_2_x.max() - z_2_x.min())
        z_2_y_norm = (z_2_y - z_2_y.min()) / (z_2_y.max() - z_2_y.min())
        locality_loss_x = self.l1(z_2_x_norm, z_1_x_norm)
        locality_loss_y = self.l1(z_2_y_norm, z_1_y_norm)
        locality_loss = locality_loss_x + locality_loss_y
        scaled_locality_loss = self.locality_weight * locality_loss

        # compare bounding boxes
        # TODO: use l1 or mse?
        # TODO: use 10 and 90 percentiles?
        # z_2_min10 = z_2.min(dim=0).values + (z_2.max(dim=0).values - z_2.min(dim=0).values) * 0.1
        # z_2_max90 = z_2.min(dim=0).values + (z_2.max(dim=0).values - z_2.min(dim=0).values) * 0.9
        out_latent_space_min10 = self.out_latent_space.min(dim=0).values + \
            (self.out_latent_space.max(dim=0).values -
             self.out_latent_space.min(dim=0).values) * 0.1
        out_latent_space_max90 = self.out_latent_space.min(dim=0).values + \
            (self.out_latent_space.max(dim=0).values -
             self.out_latent_space.min(dim=0).values) * 0.9
        bounding_box_loss = self.l1(z_2.min(dim=0).values, out_latent_space_min10) + \
            self.l1(z_2.max(dim=0).values, out_latent_space_max90)
        scaled_bounding_box_loss = self.bounding_box_weight * bounding_box_loss

        # target matching loss
        dist_from_out = torch.cdist(z_2, self.out_latent_space)
        min_dist_from_out = torch.min(dist_from_out, dim=1).values
        # TODO: use mean or max or sum?
        target_matching_loss = min_dist_from_out.mean()
        target_matching_scale = self.target_matching_weight * \
            min(1.0, (epoch_idx - self.target_matching_start) /
                self.target_matching_warmup_epochs) if epoch_idx > self.target_matching_start else 0
        scaled_target_matching_loss = target_matching_scale * target_matching_loss

        # cycle consistency loss
        z_3 = self.out_model.encode(x_hat.detach())
        cycle_consistency_loss = self.mse(z_3, z_2)
        cycle_consistency_scale = self.cycle_consistency_weight * \
            min(1.0, (epoch_idx - self.cycle_consistency_start) /
                self.cycle_consistency_warmup_epochs) if epoch_idx > self.cycle_consistency_start else 0
        scaled_cycle_consistency_loss = cycle_consistency_scale * cycle_consistency_loss

        # total loss
        loss = scaled_locality_loss + scaled_bounding_box_loss + scaled_target_matching_loss + \
            scaled_cycle_consistency_loss

        # backward pass
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        scheduler.step()

        # log losses
        self.log_dict({
            "train_locality_loss": locality_loss,
            "train_bounding_box_loss": bounding_box_loss,
            "train_target_matching_loss": target_matching_loss,
            "train_cycle_consistency_loss": cycle_consistency_loss,
            "train_loss": loss,
        })

    def on_train_epoch_end(self) -> None:
        self.bounding_box_weight *= self.bounding_box_weight_decay
        self.model.eval()
        # get the epoch number from trainer
        epoch = self.trainer.current_epoch

        if epoch % self.plot_interval != 0 and epoch != 0:
            return

        self.save_latent_space_plot()

    def save_latent_space_plot(self, batch_size=64):
        """Save a figure of the latent space"""
        # get the length of the training dataset
        dataset = self.trainer.train_dataloader.dataset
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, drop_last=False)
        z_all = torch.zeros(
            len(dataset), self.out_features).to(self.device)
        for batch_idx, data in enumerate(loader):
            x, y = data
            x_recon, mean, logvar, z = self.in_model(x.to(self.device))
            z = self.model(z)
            z = z.detach()
            z_all[batch_idx*batch_size: batch_idx*batch_size + batch_size] = z
        z_all = z_all.cpu().numpy()
        # create the figure
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        out_latent_space = self.out_latent_space.cpu().numpy()
        ax.scatter(out_latent_space[:, 0],
                   out_latent_space[:, 1], c="blue")
        ax.scatter(z_all[:, 0], z_all[:, 1], c="red")
        ax.set_title(
            f"Latent space at epoch {self.trainer.current_epoch}")
        # save figure to checkpoint folder/latent
        save_dir = os.path.join(self.args.ckpt_path,
                                self.args.ckpt_name, "latent")
        os.makedirs(save_dir, exist_ok=True)
        fig_name = f"latent_{str(self.trainer.current_epoch).zfill(5)}.png"
        plt.savefig(os.path.join(save_dir, fig_name))
        plt.close(fig)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.lr_decay)
        return [optimizer], [scheduler]

# %%
# create args class


class Args:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# %%
# load image model
img_model_args = Args(
    # root path
    root_path='',
    # dataset
    csv_path='white_squares_xy_16_4.csv',
    img_size=16,
    square_size=4,
    # model
    in_channels=1,
    latent_size=2,
    layers_channels=[512, 256, 1024],
    d_hidden_size=512,
    d_num_layers=5,
    # training
    dataset_size=144,
    mmd_prior_distribution="gaussian",
    kld_weight=0.02,
    mmd_weight=0.02,
    tc_weight=2,
    l1_weight=0,
    onpix_weight=1,
    lr_vae=0.03,
    lr_decay_vae=0.99,
    lr_d=1e-4,
    lr_decay_d=0.99,
    plot_interval=1000,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt_path = '../../ckpt/white_squares_fvae_opt/factorvae-opt-v3/factorvae-opt-v3_last_epoch=258706.ckpt'
ckpt = torch.load(ckpt_path, map_location=device)
img_model = PlFactorVAE(img_model_args).to(device)
img_model.load_state_dict(ckpt['state_dict'])
img_model.eval()

# %%
# load audio model
audio_model_args = Args(
    # root path
    root_path='',
    # dataset
    csv_path='sinewave.csv',
    img_size=64,
    # model
    in_channels=1,
    latent_size=2,
    layers_channels=[64, 128, 256, 512],
    d_hidden_size=512,
    d_num_layers=5,
    # training
    recon_weight=200,
    kld_weight=0.1,
    kld_start=0,
    kld_warmup_epochs=1,
    tc_weight=4,
    tc_start=0,
    tc_warmup_epochs=1,
    l1_weight=0.0,
    lr_d=0.01,
    lr_decay_d=0.999,
    lr_decay_vae=0.999,
    lr_vae=0.01,
    # checkpoint & logging
    ckpt_path='./ckpt/sinewave_fvae-opt',
    ckpt_name='opt-v33',
    logdir='./logs/sinewave_fvae-opt',
    plot_interval=1,
)

# create train and val datasets and loaders
sinewave_ds_train = Sinewave_dataset(
    root_path=audio_model_args.root_path, csv_path=audio_model_args.csv_path, flag="train")
sinewave_ds_val = Sinewave_dataset(
    root_path=audio_model_args.root_path, csv_path=audio_model_args.csv_path, flag="val", scaler=sinewave_ds_train.scaler)

# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt_path = '../../ckpt/sinewave_fvae-opt/opt-v27/opt-v27_last_epoch=10405.ckpt'
ckpt = torch.load(ckpt_path, map_location=device)
audio_model_args.train_scaler = sinewave_ds_train.scaler
audio_model = PlFactorVAE1D(audio_model_args).to(device)
audio_model.load_state_dict(ckpt['state_dict'])
audio_model.eval()

# %%
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
print(audio_model_latent_space.shape)

# %%
# create mapper
mapper_args = Args(
    # model
    in_features=2,
    out_features=2,
    hidden_layers_features=[64, 128, 256, 512, 256, 128, 64],
    # training
    train_epochs=3001,
    batch_size=img_model_args.dataset_size,
    lr=1e-2,
    lr_decay=0.99,
    locality_weight=100,
    bounding_box_weight=10,
    bounding_box_weight_decay=0.9,
    target_matching_weight=20,
    target_matching_start=0,
    target_matching_warmup_epochs=1000,
    cycle_consistency_weight=10,
    cycle_consistency_start=0,
    cycle_consistency_warmup_epochs=1000,
    in_model=img_model,
    out_model=audio_model,
    out_latent_space=audio_model_latent_space,
    # logging
    plot_interval=10,
    ckpt_path="./ckpt/mapper_test",
    ckpt_name="test-v1",
    logdir="./logs/mapper_test",
    comment="test",
)

model = PlMapper(mapper_args)
model.in_model.requires_grad_(False)
model.out_model.requires_grad_(False)

# %%
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

# %%
# checkpoint callbacks
checkpoint_path = os.path.join(mapper_args.ckpt_path, mapper_args.ckpt_name)
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

# %%
trainer = Trainer(
    max_epochs=mapper_args.train_epochs,
    enable_checkpointing=True,
    callbacks=[best_checkpoint_callback, last_checkpoint_callback],
    logger=tensorboard_logger,
    log_every_n_steps=1,
)

# %%
# save hyperparameters
hyperparams = dict(
    in_features=mapper_args.in_features,
    out_features=mapper_args.out_features,
    hidden_layers_features=mapper_args.hidden_layers_features,
    batch_size=mapper_args.batch_size,
    lr=mapper_args.lr,
    lr_decay=mapper_args.lr_decay,
    locality_weight=mapper_args.locality_weight,
    target_matching_weight=mapper_args.target_matching_weight,
    cycle_consistency_weight=mapper_args.cycle_consistency_weight,
    comment=mapper_args.comment
)

trainer.logger.log_hyperparams(hyperparams)

# %%
# train the model
trainer.fit(model=model, train_dataloaders=train_loader)

# %%
# load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt_path = 'ckpt/mapper_test/test-v1/test-v1_last_epoch=3000.ckpt'
ckpt = torch.load(ckpt_path, map_location=device)
model = PlMapper(mapper_args).to(device)
model.load_state_dict(ckpt['state_dict'])
model.eval()

# %%
# fit a KD tree to the scaled sinewave dataset
sinewave_ds_all = Sinewave_dataset(flag="all", scaler=sinewave_ds_train.scaler)

# %%
melbands = sinewave_ds_all.all_tensors.squeeze(-1).cpu().numpy()
tree = KDTree(melbands)

# %%
# create osc client
ip = "127.0.0.1"
port = 12347
client = udp_client.SimpleUDPClient(ip, port)

# %%
# create a function to generate an input image based on xy coordinates


def handle_pictslider(unused_addr, x, y):
    # create the image
    img = square_over_bg(x, y, img_model_args.img_size,
                         img_model_args.square_size)
    # add a channel dimension
    img = img.unsqueeze(0).unsqueeze(0)
    # encode the image
    z_1 = img_model.encode(img.to(device))
    # project to audio latent space
    z_2 = model(z_1.to(device))
    # decode the audio
    mels_norm = audio_model.decode(z_2)
    # convert to numpy
    mels_norm = mels_norm.squeeze(1).detach().cpu().numpy()
    # query the KD tree
    _, idx = tree.query(mels_norm, k=1)
    idx = idx[0][0]
    # look up pitch and loudness from dataset
    row = sinewave_ds_all.df.iloc[idx]
    pitch = row["pitch"]
    loudness = row["loudness"]
    # send pitch and loudness to Max
    client.send_message("/sineparams", [pitch, loudness])


# %%
# create an OSC receiver and start it
# create a dispatcher
d = dispatcher.Dispatcher()
d.map("/pictslider", handle_pictslider)
# create a server
ip = "127.0.0.1"
port = 12346
server = osc_server.ThreadingOSCUDPServer(
    (ip, port), d)

# %%
print("Serving on {}".format(server.server_address))
server.serve_forever()

# %%
server.server_close()
print("server closed")
# %%
