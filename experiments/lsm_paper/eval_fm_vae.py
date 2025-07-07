# %%
# imports

import argparse
import os
import torch
import random
import numpy as np
import pandas as pd
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from sonification.models.models import PlFMFactorVAE
from sonification.utils.misc import midi2frequency
from sonification.utils.tensor import scale
from torch.utils.data import DataLoader
from tqdm import tqdm

# %%
ckpt_path = '../../ckpt/fm_vae'
ckpt_name = 'imv_v3.2'
ckpt_path = os.path.join(ckpt_path, ckpt_name)
# list files, find the one that has "last" in it
ckpt_files = [f for f in os.listdir(ckpt_path) if 'last' in f]
if len(ckpt_files) == 0:
    raise ValueError(f"No checkpoint file found in {ckpt_path} with 'last' in the name")
ckpt_file = ckpt_files[0]
ckpt_path = os.path.join(ckpt_path, ckpt_file)
print(f"Checkpoint file found: {ckpt_path}")

# %%
# load checkpoint and extract saved args
ckpt = torch.load(ckpt_path, map_location='cpu')
args = ckpt["hyper_parameters"]['args']
print(args)

# %%
# create model with args and load state dict
model = PlFMFactorVAE(args)
model.load_state_dict(ckpt['state_dict'])
model.eval()

# %%
# test decode random latent
z = torch.randn(1, args.latent_size)
print(f"Random latent z: {z}")
norm_predicted_params = model.model.decoder(z)
print(f"Predicted parameters: {norm_predicted_params}")
predicted_params = model.scale_predicted_params(norm_predicted_params)
print(f"Predicted parameters (scaled): {predicted_params}")

# %%
# create a meshgrid of parameters to synthesize (SynthMaps style)

# create ranges for each parameter
pitch_steps = 51
harm_ratio_steps = 51
mod_idx_steps = 51
pitches = np.linspace(38, 86, pitch_steps)
freqs = midi2frequency(pitches)  # x
ratios = np.linspace(0, 1, harm_ratio_steps) * args.max_harm_ratio  # y
indices = np.linspace(0, 1, mod_idx_steps) * args.max_mod_idx  # z

# make into 3D mesh
freqs, ratios, indices = np.meshgrid(freqs, ratios, indices)  # y, x, z!
print(freqs.shape, ratios.shape, indices.shape)

# Create a dictionary where each key-value pair corresponds to a column in the dataframe
data = {
    "x": np.tile(np.repeat(np.arange(pitch_steps), mod_idx_steps), harm_ratio_steps),
    "y": np.repeat(np.arange(harm_ratio_steps), pitch_steps * mod_idx_steps),
    "z": np.tile(np.arange(mod_idx_steps), harm_ratio_steps * pitch_steps),
    "freq": freqs.flatten(),
    "harm_ratio": ratios.flatten(),
    "mod_index": indices.flatten()
}

# Create the dataframe
df = pd.DataFrame(data)
df.head()

# %%
# get the cuda/mps device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
# move model to device
model = model.to(device)
model.mel_spectrogram.to(device)  # move mel spectrogram to device

# %%
# iterate the dataframe in batches and synthesize
batch_size = 32
n_batches = len(df) // batch_size + 1
print(f"Number of batches: {n_batches}, batch size: {batch_size}")
z_min_all = torch.ones(args.latent_size) * 1000
z_max_all = torch.ones(args.latent_size) * -1000
with torch.no_grad():
    for i in tqdm(range(n_batches)):
        batch_df = df.iloc[i * batch_size: (i + 1) * batch_size]
        # convert to tensor
        batch_z = torch.tensor(batch_df[['freq', 'harm_ratio', 'mod_index']].values, dtype=torch.float32)
        freqs = batch_z[:, 0]
        ratios = batch_z[:, 1]
        indices = batch_z[:, 2]
        # repeat in samples dimension
        freqs = freqs.unsqueeze(1).repeat(1, args.length_samps).to(device)
        ratios = ratios.unsqueeze(1).repeat(1, args.length_samps).to(device)
        indices = indices.unsqueeze(1).repeat(1, args.length_samps).to(device)
        # synthesize
        x = model.input_synth(freqs, ratios, indices)
        # add channel dimension
        in_wf = x.unsqueeze(1)
        # get the mel spectrogram
        in_spec = model.mel_spectrogram(in_wf)
        # normalize it
        in_spec = scale(in_spec, in_spec.min(), in_spec.max(), 0, 1)
        # encode
        mu, logvar = model.model.encode(in_spec)
        z = model.model.reparameterize(mu, logvar)
        # update min and max
        z_min_all = torch.min(z_min_all, z.min(dim=0).values.cpu())
        z_max_all = torch.max(z_max_all, z.max(dim=0).values.cpu())


# %%
z_min_all, z_max_all

# %%
# set up an OSC server to receive latent codes from Max and send back the synth parameters
from pythonosc import dispatcher, osc_server, udp_client

client = udp_client.SimpleUDPClient("127.0.0.1", 12345)

def handle_latent_code(unused_addr, *args):
    # args is a tuple of the latent code
    latent_code = torch.tensor(args, dtype=torch.float32)
    # decode the latent code
    with torch.no_grad():
        latent_code_scaled = scale(latent_code, -1, 1, z_min_all, z_max_all)
        norm_predicted_params = model.model.decoder(latent_code.unsqueeze(0).to(device))
        predicted_params = model.scale_predicted_params(norm_predicted_params)
    # send back the parameters to Max
    client.send_message("/fmparams", predicted_params.squeeze(0).cpu().numpy().tolist())

# create a dispatcher
d = dispatcher.Dispatcher()
# map the OSC address to the function
d.map("/latent_code", handle_latent_code)

# create an OSC server
ip = "127.0.0.1"
port = 12346
server = osc_server.ThreadingOSCUDPServer((ip, port), d)
print(f"Serving on {server.server_address}")
# start the server
try:
    server.serve_forever()
except KeyboardInterrupt:
    print("Server stopped by user")
finally:
    server.shutdown()
    print("Server shutdown")
    server.server_close()
    print("Server closed")

# %%
