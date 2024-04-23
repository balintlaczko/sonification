# imports
# from torchsummary import summary
from lightning.pytorch import LightningModule
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from sonification.models.ddsp import FMSynth
from sonification.models.layers import ConvEncoder1D, LinearProjector
from librosa import resample
# from sonification.datasets import FmSynthDataset
from sonification.utils.dsp import transposition2duration
from torchaudio.functional import amplitude_to_DB
from torchaudio.transforms import MelSpectrogram
from sonification.utils.array import fluid_dataset2array
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
from torch import nn
import os
from matplotlib import pyplot as plt
import pandas as pd
import json


class Args:
    pass


class FmSynthDataset(Dataset):
    def __init__(self, csv_path=None, dataframe=None, sr=48000, dur=1):
        if dataframe is not None:
            self.df = dataframe
        else:
            self.df = pd.read_csv(csv_path)
        self.sr = sr
        self.dur = dur
        self.synth = FMSynth(sr)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        nsamps = int(self.dur * self.sr)
        row = self.df.iloc[idx]
        row_tensor = torch.tensor(row.values, dtype=torch.float32)
        row_tensor = row_tensor.unsqueeze(0) if len(
            row_tensor.shape) < 2 else row_tensor
        # find the column id for "freq", "harm_ratio", "mod_index"
        freq_col_id = self.df.columns.get_loc("freq")
        harm_ratio_col_id = self.df.columns.get_loc("harm_ratio")
        mod_index_col_id = self.df.columns.get_loc("mod_index")
        # extract the carrier frequency, harm_ratio, mod_index, repeat for nsamps
        carr_freq = row_tensor[:,
                               freq_col_id].unsqueeze(-1).repeat(1, nsamps)
        harm_ratio = row_tensor[:,
                                harm_ratio_col_id].unsqueeze(-1).repeat(1, nsamps)
        mod_index = row_tensor[:,
                               mod_index_col_id].unsqueeze(-1).repeat(1, nsamps)
        fm_synth = self.synth(carr_freq, harm_ratio, mod_index)
        return fm_synth


class FmMel2PCADataset(Dataset):
    def __init__(
            self,
            fm_params_dataframe,
            pca_array,
            sr=48000,
            dur=1,
            n_mels=200,
            mel_scaler=None,
            pca_scaler=None):
        self.fm_synth = FmSynthDataset(
            dataframe=fm_params_dataframe, sr=sr, dur=dur)
        self.pca = torch.tensor(pca_array).float()
        self.sr = sr
        self.dur = dur
        self.n_mels = n_mels
        self.mel_spec = MelSpectrogram(
            sample_rate=self.sr,
            n_fft=4096,
            f_min=20,
            f_max=10000,
            pad=1,
            n_mels=self.n_mels,
            power=2,
            norm='slaney',
            mel_scale='slaney')

        # hold mel tensors in memory
        self.all_mels = torch.zeros(len(self.fm_synth), 1, self.n_mels)
        self.load_all_mels()

        # scalers
        self.mel_scaler = mel_scaler
        self.pca_scaler = pca_scaler

    def fit_scalers(self):
        # fit mel scaler
        if self.mel_scaler is None:
            self.mel_scaler = MinMaxScaler()
        self.fit_mel_scaler()
        # fit pca scaler
        if self.pca_scaler is None:
            self.pca_scaler = MinMaxScaler()
        print("Fitting PCA scaler...")
        self.pca_scaler.fit(self.pca.numpy())
        print("PCA scaler fit")

    def __len__(self):
        return len(self.fm_synth)

    def __getitem__(self, idx):
        # print(idx)
        mel = self.all_mels[idx]  # (B, 1, n_mels)
        if isinstance(idx, int):
            mel = mel.unsqueeze(0)
        mel_scaled = self.mel_scaler.transform(
            mel.squeeze(1).numpy())  # (B, n_mels)
        if not isinstance(idx, int):
            mel_scaled = torch.tensor(
                mel_scaled).reshape(-1, 1, self.n_mels)  # (B, 1, n_mels)
        pca = self.pca[idx]  # (B, 2)
        if isinstance(idx, int):
            pca = pca.unsqueeze(0)
        pca_scaled = self.pca_scaler.transform(pca.numpy())
        pca_scaled = torch.tensor(pca_scaled)
        if isinstance(idx, int):
            pca_scaled = pca_scaled.squeeze(0)
        return mel_scaled, pca_scaled

    def get_mel(self, idx):
        fm_synth = self.fm_synth[idx]  # (B, T)
        mel = self.mel_spec(fm_synth)  # (B, n_mels, T)
        mel_avg = mel.mean(dim=-1, keepdim=True)  # (B, n_mels, 1)
        mel_avg_db = amplitude_to_DB(
            mel_avg, multiplier=10, amin=1e-5, db_multiplier=20, top_db=80)  # (B, n_mels, 1)
        return mel_avg_db

    def load_all_mels(self):
        batch_size = 256
        # create batches of indices
        indices = torch.arange(len(self.fm_synth))
        batches = torch.split(indices, batch_size)
        pbar = tqdm(batches)
        pbar.set_description("Loading mel tensors")
        # iterate through all fm synths in batches
        for idx in pbar:
            mel = self.get_mel(idx)  # (B, n_mels, 1)
            # swap channels and n_mels dims
            mel = mel.permute(0, 2, 1)  # (B, 1, n_mels)
            self.all_mels[idx] = mel
        print("All mel tensors loaded to memory")

    def fit_mel_scaler(self):
        all_mels = self.all_mels.squeeze(1).numpy()  # (B, n_mels)
        print("Fitting mel scaler...")
        self.mel_scaler.fit(all_mels)
        print("Mel scaler fit")


class FmMelContrastiveDataset(Dataset):
    def __init__(
            self,
            fm_params_dataframe,
            sr=48000,
            dur=1,
            n_mels=200,
            mel_scaler=None):
        self.fm_synth = FmSynthDataset(
            dataframe=fm_params_dataframe, sr=sr, dur=dur)
        self.sr = sr
        self.dur = dur
        self.n_mels = n_mels
        self.n_transpositions = 2
        self.mel_spec = MelSpectrogram(
            sample_rate=self.sr,
            n_fft=4096,
            f_min=20,
            f_max=10000,
            pad=1,
            n_mels=self.n_mels,
            power=2,
            norm='slaney',
            mel_scale='slaney')

        # scaler
        self.mel_scaler = mel_scaler

    def __len__(self):
        return len(self.fm_synth)

    def __getitem__(self, idx):
        # print(idx)
        assert self.mel_scaler is not None
        # get random numbers between -2 and 2 for transposition
        transpositions = torch.rand(self.n_transpositions) * 4 - 2
        # append 0 to the front, so the first element is the original
        transpositions = torch.cat((torch.tensor([0]), transpositions))
        output = []
        for transpose in transpositions:
            mel = self.get_mel(idx, transpose)  # (B, n_mels, 1)
            mel_scaled = self.mel_scaler.transform(
                mel.squeeze(-1).numpy())  # (B, n_mels)
            if not isinstance(idx, int):
                mel_scaled = torch.tensor(
                    mel_scaled).reshape(-1, 1, self.n_mels)  # (B, 1, n_mels)
            output.append(mel_scaled)
        # a list of tensors of shape (B, n_mels)
        return output

    def get_mel(self, idx, transpose=0):
        fm_synth = self.fm_synth[idx]  # (B, T)
        if transpose != 0:
            target_dur = transposition2duration(transpose)
            target_sr = int(self.sr * target_dur)
            fm_synth = resample(
                fm_synth.numpy(), orig_sr=self.sr, target_sr=target_sr)  # (B, T')
            fm_synth = torch.tensor(fm_synth)
        mel = self.mel_spec(fm_synth)  # (B, n_mels, T')
        mel_avg = mel.mean(dim=-1, keepdim=True)  # (B, n_mels, 1)
        mel_avg_db = amplitude_to_DB(
            mel_avg, multiplier=10, amin=1e-5, db_multiplier=20, top_db=80)  # (B, n_mels, 1)
        return mel_avg_db


class PlMelEncoder(LightningModule):
    def __init__(self, args):
        super().__init__()
        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        # self.save_hyperparameters()

        # model params
        # ConvEncoder1D
        self.conv_in_channels = args.conv_in_channels
        self.conv_layers_channels = args.conv_layers_channels
        self.conv_in_size = args.conv_in_size
        self.conv_out_features = args.conv_out_features
        # LinearProjector
        self.proj_in_features = self.conv_out_features
        self.proj_out_features = args.out_features
        self.proj_hidden_layers_features = args.proj_hidden_layers_features

        # losses
        self.mse = nn.MSELoss()
        self.contrastive_diff_loss_scaler = args.contrastive_diff_loss_scaler

        # learning rate
        self.lr = args.lr
        self.lr_decay = args.lr_decay

        # logging
        self.plot_interval = args.plot_interval
        self.args = args

        # models
        conv = ConvEncoder1D(
            in_channels=self.conv_in_channels, output_size=self.conv_out_features, layers_channels=self.conv_layers_channels, input_size=self.conv_in_size)
        proj = LinearProjector(
            in_features=self.proj_in_features, out_features=self.proj_out_features, hidden_layers_features=self.proj_hidden_layers_features)
        self.model = nn.Sequential(conv, proj)

        # mode: "supervised" or "contrastive"
        self.mode = args.mode

    def forward(self, x):
        return self.model(x)

    def supervised_loss(self, batch):
        # get sample and label
        x, y = batch
        # predict label
        y_hat = self.model(x)
        # calculate loss
        loss = self.mse(y_hat, y)
        return loss

    def contrastive_loss(self, batch):
        # get sample and two transformations
        x, x_a, x_b = batch
        # get representations for each
        s_x = self.model(x)
        s_x_a = self.model(x_a)
        s_x_b = self.model(x_b)

        # # vector shifts for the same transformation should be the same
        # v_a = s_x_a - s_x
        # v_b = s_x_b - s_x
        # loss_for_same = torch.std(v_a) + torch.std(v_b)
        # # vector shifts for different transformations should be different
        # loss_for_diff = torch.nn.functional.l1_loss(v_a, v_b)
        # # total loss
        # loss = loss_for_same - \
        #     (self.contrastive_diff_loss_scaler * loss_for_diff)

        # the same samples should have the same representation
        loss_same = self.mse(s_x, s_x_a) + self.mse(s_x, s_x_b)
        # different samples should have different representations
        loss_diff = self.mse(s_x_a, s_x_b)
        # total loss
        loss = loss_same - (self.contrastive_diff_loss_scaler * loss_diff)

        return loss

    def training_step(self, batch, batch_idx):
        self.model.train()
        # epoch_idx = self.trainer.current_epoch

        # get the optimizer and scheduler
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()

        # in supervised mode, just map to the labels
        if self.mode == "supervised":
            loss = self.supervised_loss(batch)
        # in contrastive mode:
        # 1. the SAME transformation on DIFFERENT samples should result in the same vector shift between their latent representations
        # 2. DIFFERENT transformations on the SAME sample should NOT result in the same vector shift between their latent representations
        elif self.mode == "contrastive":
            loss = self.contrastive_loss(batch)

        # backward pass
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        scheduler.step()

        # log losses
        self.log_dict({
            "train_loss": loss,
        })

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        # epoch_idx = self.trainer.current_epoch

        # in supervised mode, just map to the labels
        if self.mode == "supervised":
            loss = self.supervised_loss(batch)
        # in contrastive mode:
        # 1. the SAME transformation on DIFFERENT samples should result in the same vector shift between their latent representations
        # 2. DIFFERENT transformations on the SAME sample should NOT result in the same vector shift between their latent representations
        elif self.mode == "contrastive":
            loss = self.contrastive_loss(batch)

        # log losses
        self.log_dict({
            "val_loss": loss,
        })

    def on_validation_epoch_end(self) -> None:
        self.model.eval()
        # get the epoch number from trainer
        epoch = self.trainer.current_epoch

        if epoch % self.plot_interval != 0 and epoch != 0:
            return

        self.save_latent_space_plot()

    def save_latent_space_plot(self, batch_size=512):
        """Save a figure of the latent space"""
        # get the length of the training dataset
        dataset = self.trainer.val_dataloaders.dataset
        # batchsampler = torch.utils.data.BatchSampler(
        #     range(len(dataset)), batch_size=batch_size, drop_last=False)
        # loader = DataLoader(dataset, batch_size=None, sampler=batchsampler)
        loader = self.supervised_val_loader
        z_all = torch.zeros(
            len(dataset), self.proj_out_features).to(self.device)
        for batch_idx, data in enumerate(loader):
            if self.mode == "supervised":
                x, _ = data
            elif self.mode == "contrastive":
                # x, _, _ = data
                x, _ = data
            z = self.model(x.to(self.device))
            z = z.detach()
            z_all[batch_idx*batch_size: batch_idx *
                  batch_size + batch_size] = z
        z_all = z_all.cpu().numpy()
        # create the figure
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        ax.scatter(z_all[:, 0], z_all[:, 1])
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


csv_abspath = "./experiments/pca_finetuning/fm_synth_params.csv"
pca_abspath = "./experiments/pca_finetuning/pca_mels_mean.json"


def main():
    # read fm synth dataframe from csv
    df_fm = pd.read_csv(csv_abspath, index_col=0)
    # read pca array from json
    pca_array = fluid_dataset2array(json.load(open(pca_abspath, "r")))
    # total number of samples
    n_samples = len(df_fm)
    n_samples = 10000  # for testing
    # generate dataset splits on indices
    indices = torch.arange(len(df_fm))
    # shuffle
    indices = torch.randperm(len(df_fm))
    indices = indices[:n_samples]
    # splits = torch.utils.data.random_split(list(range(n_samples)), [0.8, 0.2])
    splits = torch.utils.data.random_split(list(indices), [0.8, 0.2])
    train_split, val_split = splits

    # create train and val datasets for fm params and pca
    df_fm_train, df_fm_val = df_fm.loc[list(
        train_split)], df_fm.loc[list(val_split)]
    pca_train, pca_val = pca_array[list(
        train_split)], pca_array[list(val_split)]

    # create train and val datasets for mel spectrograms and pca
    fm_mel_pca_train = FmMel2PCADataset(
        df_fm_train, pca_train, sr=48000, dur=1, n_mels=200)
    fm_mel_pca_val = FmMel2PCADataset(
        df_fm_val, pca_val, sr=48000, dur=1, n_mels=200)

    # fit scalers in the train dataset, pass fit scalers to val dataset
    fm_mel_pca_train.fit_scalers()
    fm_mel_pca_val.mel_scaler = fm_mel_pca_train.mel_scaler
    fm_mel_pca_val.pca_scaler = fm_mel_pca_train.pca_scaler

    # create contrastive datasets based on the same splits
    fm_contrastive_train = FmMelContrastiveDataset(
        df_fm_train, sr=48000, dur=1, mel_scaler=fm_mel_pca_train.mel_scaler)
    fm_contrastive_val = FmMelContrastiveDataset(
        df_fm_val, sr=48000, dur=1, mel_scaler=fm_mel_pca_train.mel_scaler)  # use train scaler

    # create samplers & dataloaders
    batch_size = 512
    # num_workers = 0

    # supervised train
    fm_mel_pca_train_randomsampler = torch.utils.data.RandomSampler(
        fm_mel_pca_train, replacement=False)
    fm_mel_pca_train_batchsampler = torch.utils.data.BatchSampler(
        fm_mel_pca_train_randomsampler, batch_size=batch_size, drop_last=True)
    supervised_train_loader = DataLoader(
        fm_mel_pca_train, batch_size=None, sampler=fm_mel_pca_train_batchsampler)

    # supervised val
    # fm_mel_pca_val_randomsampler = torch.utils.data.RandomSampler(
    #     fm_mel_pca_val, replacement=False)
    fm_mel_pca_val_batchsampler = torch.utils.data.BatchSampler(
        range(len(fm_mel_pca_val)), batch_size=batch_size, drop_last=False)
    supervised_val_loader = DataLoader(
        fm_mel_pca_val, batch_size=None, sampler=fm_mel_pca_val_batchsampler)

    # contrastive train
    fm_contrastive_train_randomsampler = torch.utils.data.RandomSampler(
        fm_contrastive_train, replacement=False)
    fm_contrastive_train_batchsampler = torch.utils.data.BatchSampler(
        fm_contrastive_train_randomsampler, batch_size=batch_size, drop_last=True)
    contrastive_train_loader = DataLoader(
        fm_contrastive_train, batch_size=None, sampler=fm_contrastive_train_batchsampler)

    # contrastive val
    # fm_contrastive_val_randomsampler = torch.utils.data.RandomSampler(
    #     fm_contrastive_val, replacement=False)
    fm_contrastive_val_batchsampler = torch.utils.data.BatchSampler(
        range(len(fm_contrastive_val)), batch_size=batch_size, drop_last=False)
    contrastive_val_loader = DataLoader(
        fm_contrastive_val, batch_size=None, sampler=fm_contrastive_val_batchsampler)

    args = Args()
    args.conv_in_channels = 1
    args.conv_layers_channels = [32, 64, 128]
    args.conv_in_size = 200
    args.conv_out_features = 128
    args.out_features = 2
    args.proj_hidden_layers_features = [64, 32]
    args.contrastive_diff_loss_scaler = 1
    args.lr = 1e-3
    args.lr_decay = 0.99
    args.plot_interval = 1
    args.mode = "supervised"
    args.ckpt_path = "ckpt"
    args.ckpt_name = "test_finetuning"
    args.resume_ckpt_path = None
    args.logdir = "logs"
    args.train_epochs = 21

    # create model
    model = PlMelEncoder(args)
    # show model summary via torch summary
    # summary(model, (1, 200))

    # checkpoint callbacks
    checkpoint_path = os.path.join(args.ckpt_path, args.ckpt_name)
    best_checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=checkpoint_path,
        filename=args.ckpt_name + "_val_{epoch:02d}-{val_vae_loss:.4f}",
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

    # create trainer
    trainer = Trainer(
        max_epochs=args.train_epochs,
        enable_checkpointing=True,
        callbacks=[best_checkpoint_callback, last_checkpoint_callback],
        logger=tensorboard_logger,
        log_every_n_steps=1,
    )

    # save hyperparameters
    hyperparams = dict(
        conv_in_channels=args.conv_in_channels,
        conv_layers_channels=args.conv_layers_channels,
        conv_in_size=args.conv_in_size,
        conv_out_features=args.conv_out_features,
        out_features=args.out_features,
        proj_hidden_layers_features=args.proj_hidden_layers_features,
        contrastive_diff_loss_scaler=args.contrastive_diff_loss_scaler,
        mode=args.mode,
        lr=args.lr,
        lr_decay=args.lr_decay,
        plot_interval=args.plot_interval,
        ckpt_path=args.ckpt_path,
        ckpt_name=args.ckpt_name,
        resume_ckpt_path=args.resume_ckpt_path,
        logdir=args.logdir,
        train_epochs=args.train_epochs
    )
    trainer.logger.log_hyperparams(hyperparams)

    # train the model in supervised mode
    model.mode = "supervised"
    model.supervised_val_loader = supervised_val_loader
    print("Training in supervised mode")
    trainer.fit(model, supervised_train_loader, supervised_val_loader,
                ckpt_path=args.resume_ckpt_path)

    # checkpoint callbacks
    checkpoint_path = os.path.join(args.ckpt_path, args.ckpt_name)
    best_checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=checkpoint_path,
        filename=args.ckpt_name + "_val_{epoch:02d}-{val_vae_loss:.4f}",
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

    # create trainer
    args.train_epochs = 31
    trainer_contrastive = Trainer(
        max_epochs=args.train_epochs,
        enable_checkpointing=True,
        callbacks=[best_checkpoint_callback, last_checkpoint_callback],
        logger=tensorboard_logger,
        log_every_n_steps=1,
    )

    # train the model in contrastive mode
    model.mode = "contrastive"
    # model.lr = 1e-5
    model.supervised_val_loader = supervised_val_loader
    model.contrastive_diff_loss_scaler = 0.5
    args.resume_ckpt_path = "ckpt/test_finetuning/test_finetuning_last_epoch=20.ckpt"
    print("Training in contrastive mode")
    trainer_contrastive.fit(
        model, contrastive_train_loader, contrastive_val_loader, ckpt_path=args.resume_ckpt_path)


if __name__ == "__main__":
    main()
