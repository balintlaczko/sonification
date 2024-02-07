# %%
# imports
import os
import numpy as np
from utils.array import scale_array, scale_array_exp
from utils.dsp import midi2frequency
import platform

# %%
# dataset parameters

dataset_folder = "/Volumes/T7/synth_dataset_fm_inversynth"
# if on Windows, use this path
if platform.system() == "Windows":
    dataset_folder = "D:/synth_dataset_fm_inversynth"
num_samples = 1000000
num_params = 3
sr = 48000

# %%
# generate uniform random parameter values

params = np.random.uniform(0, 1, (num_samples, num_params))
carrfreq = midi2frequency(scale_array(params[:, 0], 0, 1, 44, 88))[..., None]
harm_ratio = scale_array_exp(params[:, 1], 0, 1, 1, 10, 1)[..., None]
mod_idx = scale_array_exp(params[:, 2], 0, 1, 0.1, 10, 0.5)[..., None]

# %%

# save parameter values to npy file
np.save(os.path.join(dataset_folder, "params_unscaled.npy"), params)

# save scaled parameter values to npy file

params_scaled = np.transpose(
    np.array([carrfreq[:, 0], harm_ratio[:, 0], mod_idx[:, 0]]))
np.save(os.path.join(dataset_folder, "params_scaled.npy"), params_scaled)

# %%
print("DONE!")
