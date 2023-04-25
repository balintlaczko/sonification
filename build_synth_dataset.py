# %%
# imports

import json
import matplotlib.pyplot as plt
import numpy as np
import shutil
import platform
from pathlib import Path
from flucoma import fluid
from flucoma.utils import cleanup, get_buffer
from scipy.io import wavfile as wav
from utils import *
import torch
import torchaudio
import umap

# %%

# dataset parameters

dataset_folder = "/Volumes/T7/synth_dataset"
# if on Windows, use this path
if platform.system() == "Windows":
    dataset_folder = "D:/synth_dataset"
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

##########################################################
# create new dataset without file I/O and using torchaudio

# %%

# dataset parameters

dataset_folder = "/Volumes/T7/synth_dataset_2"
# if on Windows, use this path
if platform.system() == "Windows":
    dataset_folder = "D:/synth_dataset_2"
num_samples = 1000000
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
np.save(os.path.join(dataset_folder, "params_unscaled.npy"), params)

# save scaled parameter values to npy file

params_scaled = np.transpose(
    np.array([carrfreq[:, 0], modfreq[:, 0], modamp[:, 0]]))
np.save(os.path.join(dataset_folder, "params_scaled.npy"), params_scaled)

# %%

# initialize torch device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# %%

# for each sample generate a 1 second audio buffer, save it to a tensor and extract MFCCs

# initialize MFCC transform
mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=40, melkwargs={
                                            "n_fft": 2048, "hop_length": 512, "n_mels": 200}).to(device)

# initialize MelSpectrogram transform
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sr, n_fft=2048, hop_length=512, n_mels=200).to(device)

# create tensor container for mfccs
ds_mfccs = torch.zeros((num_samples, 40, 1)).to(device)

# create tensor container for mel spectrograms
ds_melspec = torch.zeros((num_samples, 200, 1)).to(device)

for i in range(num_samples):
    # generate the audio buffer
    fm_buffer = fm_synth(sr * 1, sr, carrfreq[i], modfreq[i], modamp[i])
    # save the audio buffer to a tensor
    audio_tensor = torch.from_numpy(fm_buffer).float().to(device)
    # extract MFCCs
    # mfccs = mfcc_transform(audio_tensor)
    # take the mean across the time dimension
    # mfccs = torch.mean(mfccs, dim=1, keepdim=True)
    # save the MFCCs to the tensor container
    # ds_mfccs[i] = mfccs
    # extract mel spectrogram
    melspec = mel_transform(audio_tensor)
    # take the mean across the time dimension
    melspec = torch.mean(melspec, dim=1, keepdim=True)
    # save the mel spectrogram to the tensor container
    ds_melspec[i] = melspec

# %%

# convert the tensor container to a numpy array
ds_melspec_np = ds_melspec.cpu().numpy()

# save the dataset to a npy file
np.save(os.path.join(dataset_folder, "melspec.npy"), ds_melspec_np)

# %%

# convert the tensor container to a numpy array
ds_mfccs_np = ds_mfccs.cpu().numpy()

# save the dataset to a npy file
np.save(os.path.join(dataset_folder, "mfccs.npy"), ds_mfccs_np)

# %%

ds_mfccs_np.shape

# %%
# dataset parameters

dataset_folder = "/Volumes/T7/synth_dataset_2"
# if on Windows, use this path
if platform.system() == "Windows":
    dataset_folder = "D:/synth_dataset_2"
num_samples = 100000
num_params = 3
sr = 48000

# %%

# load the dataset from the npy file

melspec = np.load(os.path.join(dataset_folder, "melspec.npy"))

# %%
melspec[..., 0].shape

# %%

# fit umap to the dataset

reducer = umap.UMAP(n_neighbors=400, min_dist=0.1,
                    n_components=2, metric="euclidean", random_state=42)
embedding = reducer.fit_transform(melspec[..., 0])

# %%
embedding.shape

# %%
embedding[0]

# %%

# save the embedding to a npy file
np.save(os.path.join(dataset_folder, "melspec_umap_400.npy"), embedding)

# %%

# load the embedding from the npy file
embedding = np.load(os.path.join(dataset_folder, "melspec_umap_400.npy"))

# %%

# plot the embedding
plt.scatter(embedding[:, 0], embedding[:, 1], s=0.1)
plt.show()

# %%

# function: array2fluid_dataset


def array2fluid_dataset(
        array: np.ndarray,
) -> dict:
    """
    Convert a numpy array to a json format that's compatible with fluid.dataset~.

    Args:
        array (np.ndarray): The numpy array to convert. Should be a 2D array of (num_samples, num_features).

    Returns:
        dict: The json dataset.
    """
    num_cols = array.shape[1]
    out_dict = {}
    out_dict["cols"] = num_cols
    out_dict["data"] = {}
    for i in range(len(array)):
        out_dict["data"][str(i)] = array[i].tolist()
    return out_dict

# %%


# create json dataset with umap embedding
umap_ds = array2fluid_dataset(embedding)

# save the umap_ds dataset to a json file
with open(os.path.join(dataset_folder, "melspec_umap_200.json"), "w") as f:
    json.dump(umap_ds, f)

# %%

# load the scaled parameters from the npy file
params_scaled = np.load(os.path.join(dataset_folder, "params_scaled.npy"))
# create json dataset with scaled parameters
params_ds = array2fluid_dataset(params_scaled)

# save the params_ds dataset to a json file
with open(os.path.join(dataset_folder, "params_scaled.json"), "w") as f:
    json.dump(params_ds, f)

# %%
