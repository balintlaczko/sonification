# %%
# imports

import numpy as np
from utils import *
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
carrfreq = midi2frequency(scale_array(params[:, 0], 0, 1, 20, 100))[..., None]
modfreq = scale_array_exp(params[:, 1], 0, 1, 1, 1000, 2)[..., None]
mod_idx = scale_array_exp(params[:, 2], 0, 1, 0.1, 20, 1)[..., None]

# %%

# save parameter values to npy file
np.save(os.path.join(dataset_folder, "params_unscaled.npy"), params)

# save scaled parameter values to npy file

params_scaled = np.transpose(
    np.array([carrfreq[:, 0], modfreq[:, 0], mod_idx[:, 0]]))
np.save(os.path.join(dataset_folder, "params_scaled.npy"), params_scaled)

# %%
print("DONE!")