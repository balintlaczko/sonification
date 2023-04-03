# %%
# imports

import numpy as np
import shutil
from pathlib import Path
from flucoma import fluid
from flucoma.utils import cleanup, get_buffer
from scipy.io import wavfile as wav
from utils import *

# %%

# dataset parameters

dataset_folder = "/Volumes/T7/synth_dataset"
num_samples = 100000
num_params = 3
sr = 48000

# %%

# generate uniform random parameter values

params = np.random.uniform(0, 1, (num_samples, num_params))
carrfreq = midi2frequency(scale_array(params[:, 0], 0, 1, 20, 100))[..., None]
modfreq = scale_array_exp(params[:, 1], 0, 1, 1, 1000, 2)[..., None]
modamp = scale_array_exp(params[:, 2], 0, 1, 1, 1000, 2)[..., None]

# %%

# save parameter values to npy file
np.save(os.path.join(os.path.dirname(dataset_folder),
        "params_unscaled.npy"), params)

# %%

params_scaled = np.transpose(
    np.array([carrfreq[:, 0], modfreq[:, 0], modamp[:, 0]]))
np.save(os.path.join(os.path.dirname(dataset_folder),
        "params_scaled.npy"), params_scaled)

# %%

# render a 4 second audio file for each parameter set

# first clear the dataset folder
if os.path.exists(dataset_folder):
    shutil.rmtree(dataset_folder)
    os.mkdir(dataset_folder)
else:
    os.mkdir(dataset_folder)

for i in range(num_samples):
    outname = Path(dataset_folder) / \
        f"{str(i).zfill(len(str(num_samples)))}.wav"
    fm_buffer = fm_synth(sr * 4, sr, carrfreq[i], modfreq[i], modamp[i])
    wav.write(outname, sr, fm_buffer.astype(np.float32))


# %%

# take the first sample as a test and prototype audio analysis

test_sample = 0
test_file = Path(dataset_folder) / \
    f"{str(test_sample).zfill(len(str(num_samples)))}.wav"

spec_shape = get_buffer(fluid.stats(fluid.spectralshape(
    test_file), numderivs=0), output="numpy")

spec_shape


# %%

# generate a dataset of spectral shape features

# create dataset container
ds_descriptors = np.zeros((num_samples,) + spec_shape.shape)

# iterate over all samples
for i in range(num_samples):
    # get the file path
    filename = Path(dataset_folder) / \
        f"{str(i).zfill(len(str(num_samples)))}.wav"
    # get the spectral shape
    spec_shape = get_buffer(fluid.stats(fluid.spectralshape(
        filename), numderivs=0), output="numpy")
    # save the spectral shape to the dataset container
    ds_descriptors[i] = spec_shape

# save the dataset to a npy file
np.save(os.path.join(os.path.dirname(dataset_folder),
        "spectralshape.npy"), ds_descriptors)

# delete temporary files
cleanup()


# %%
