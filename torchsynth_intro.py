# %%
# imports

import numpy as np
import torch
from torch import tensor

from torchsynth.config import SynthConfig
from torchsynth.module import (
    SineVCO,
    FmVCO,
)

import soundfile as sf

from utils import *

# %%
# set torch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
# set up synth config

batch_size = 1
sample_rate = 48000
buffer_length_s = 4.0

config = SynthConfig(
    batch_size=batch_size,
    sample_rate=sample_rate,
    buffer_size_seconds=buffer_length_s,
    reproducible=False
)

# %%
# function to save audio files


def save_audio(audio, path):
    audio = audio.cpu().detach().numpy()
    audio = audio.reshape(-1)
    sf.write(path, audio, sample_rate)

# %%
# set up FmVCO


fm_vco = FmVCO(
    tuning=tensor([0.0]),
    mod_depth=tensor([frequency2midi(np.array(100))]),
    synthconfig=config,
    device=device,
)

# %%
# set up modulator SineVCO

modulator = SineVCO(
    tuning=tensor([0.0]),
    mod_depth=tensor([0.0]),
    initial_phase=tensor([torch.pi / 2]),
    synthconfig=config,
    device=device,
)

# %%
# generate FM synth buffer

carrier_freq = 400.0
carrier_freq_midi = frequency2midi(np.array(carrier_freq))
modulator_freq = 358.0
modulator_freq_midi = frequency2midi(np.array(modulator_freq))
modulator_depth = 1.0
modulator_depth_midi = frequency2midi(np.array(modulator_freq))

# generate modulator buffer
mod_signal = modulator(tensor([modulator_freq_midi]))

# set fm modulation index
fm_vco.set_parameter("mod_depth", tensor([40]))

# generate fm buffer
fm_signal = fm_vco(tensor([carrier_freq_midi]), mod_signal)

# %%
# save audio
target_path = "/Users/balintl/Desktop/torchsynth_test_fm.wav"
save_audio(fm_signal, target_path)

# save sine wave
target_path = "/Users/balintl/Desktop/torchsynth_test_sine.wav"
save_audio(mod_signal, target_path)
# %%
