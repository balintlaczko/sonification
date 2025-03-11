# %%
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

# %%
fix_seed = 2025
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# %%
class Args:
    pass

args = Args()
args.sr = 48000
args.length_s = 0.25
args.n_fft = 2048
args.f_min = midi2frequency(38)
args.f_max = midi2frequency(86)
args.n_mels = 512
args.power = 0.5
args.normalized = True
args.latent_size = 128
args.n_res_block = 12
args.n_res_channel = 32
args.hidden_dim = 64
args.num_layers = 3
args.batch_size = 512
args.lr = 1e-4
args.lr_decay = 0.5
args.train_epochs = 100
args.steps_per_epoch = 1000
args.param_loss_weight = 8
args.max_harm_ratio = 6
args.max_mod_idx = 6
args.ckpt_name = "wandb_test"
# args.ckpt_path = "experiments/fm_ddsp/checkpoints"
args.ckpt_path = "checkpoints"
# args.logdir = "experiments/fm_ddsp/logs"
args.logdir = "logs"

# %%
# a dummy dataloader
dataloader = DataLoader(
    range(args.steps_per_epoch * args.batch_size),
    batch_size=args.batch_size,
    shuffle=True
)

# %%
model = PlFMParamEstimator(args)

# %%
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
callbacks = [best_checkpoint_callback, last_checkpoint_callback]

# %%
logger = WandbLogger(
    name=args.ckpt_name,
    project="fm_ddsp",
    save_dir=args.logdir,
    )

# %%
trainer = Trainer(
    max_epochs=args.train_epochs,
    enable_checkpointing=True,
    callbacks=callbacks,
    logger=logger,
    log_every_n_steps=1,
    limit_train_batches=args.steps_per_epoch,
)

# %%
trainer.fit(model, train_dataloaders=dataloader)
# %%