# %%
# imports

import numpy as np
from pathlib import Path
from flucoma import fluid
from flucoma.utils import cleanup, get_buffer
from scipy.io import wavfile as wav
from utils import *

# %%

# dataset parameters

dataset_folder = "/Volumes/T7/synth_dataset"
num_samples = 100
num_params = 3

# %%

# generate uniform random parameter values

params = np.random.uniform(0, 1, (num_samples, num_params))


# %%
