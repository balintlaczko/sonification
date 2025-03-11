# %%
import torch
import random
import numpy as np
from lightning.pytorch import Trainer
# from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
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
args.length_s = 0.5
args.n_fft = 2048
args.f_min = midi2frequency(38)
args.f_max = midi2frequency(86)
args.n_mels = 512
args.power = 2.0
args.normalized = True
args.latent_size = 64
args.n_res_block = 8
args.n_res_channel = 32
args.hidden_dim = 64
args.num_layers = 8
args.batch_size = 512
args.lr = 1e-4
args.lr_decay = 0.5
args.train_epochs = 100
args.steps_per_epoch = 1000
args.param_loss_weight = 8
args.max_harm_ratio = 6
args.max_mod_idx = 6

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
# logger = TensorBoardLogger("experiments/fm_ddsp/logs", name="fm_ddsp")
logger = TensorBoardLogger("logs", name="fm_ddsp")

# %%
trainer = Trainer(
    max_epochs=args.train_epochs,
    logger=logger,
    log_every_n_steps=1,
    limit_train_batches=args.steps_per_epoch,
)

# %%
trainer.fit(model, train_dataloaders=dataloader)
# %%
