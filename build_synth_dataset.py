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
modfreq = scale_array_exp(params[:, 1], 0, 1, 10, 1000, 2)[..., None]
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
np.save(os.path.join(dataset_folder, "params_unscaled_2.npy"), params)

# save scaled parameter values to npy file

params_scaled = np.transpose(
    np.array([carrfreq[:, 0], modfreq[:, 0], modamp[:, 0]]))
np.save(os.path.join(dataset_folder, "params_scaled_2.npy"), params_scaled)

# %%

# initialize torch device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# %%

# test shape of mean and std

test_tensor = torch.rand((100, 10)).to(device)
test_mean = torch.mean(test_tensor, dim=1, keepdim=True)
print(test_mean.shape)
test_std = torch.std(test_tensor, dim=1, keepdim=True)
print(test_std.shape)
test_concat = torch.cat((test_mean, test_std), dim=0)
print(test_concat.shape)


# %%

# for each sample generate a 1 second audio buffer, save it to a tensor and extract MFCCs

# initialize MFCC transform
n_mels = 200
mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=40, melkwargs={
                                            "n_fft": 2048, "hop_length": 512, "n_mels": n_mels}).to(device)

# initialize MelSpectrogram transform
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sr, n_fft=2048, hop_length=512, n_mels=n_mels, normalized=True).to(device)

# create tensor container for mfccs
ds_mfccs = torch.zeros((num_samples, 40, 1)).to(device)

# create tensor container for mel spectrograms (mean + std)
ds_melspec = torch.zeros((num_samples, n_mels * 2, 1)).to(device)

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
    melspec_mean = torch.mean(melspec, dim=1, keepdim=True)
    # take the standard deviation across the time dimension
    melspec_std = torch.std(melspec, dim=1, keepdim=True)
    # save the mean + std to the tensor container
    ds_melspec[i] = torch.cat((melspec_mean, melspec_std), dim=0)

# %%

# convert the tensor container to a numpy array
ds_melspec_np = ds_melspec.cpu().numpy()

# save the dataset to a npy file
np.save(os.path.join(dataset_folder, "melspec_2_mean_std.npy"), ds_melspec_np)

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


# create json dataset with umap embedding
umap_ds = array2fluid_dataset(embedding)

# save the umap_ds dataset to a json file
with open(os.path.join(dataset_folder, "melspec_umap_200.json"), "w") as f:
    json.dump(umap_ds, f)

# %%

# load the scaled parameters from the npy file
params_scaled = np.load(os.path.join(dataset_folder, "params_scaled_2.npy"))
# create json dataset with scaled parameters
params_ds = array2fluid_dataset(params_scaled)

# save the params_ds dataset to a json file
with open(os.path.join(dataset_folder, "params_scaled_2.json"), "w") as f:
    json.dump(params_ds, f)

# %%
############################################################################################################
# read dataset from Max, compute UMAP, save to json

# %%

# dataset parameters

dataset_folder = "/Volumes/T7/synth_dataset_4"
# if on Windows, use this path
if platform.system() == "Windows":
    dataset_folder = "D:/synth_dataset_4"

# %%

# load the dataset from the json

fm_descriptors_merged_json = os.path.join(
    dataset_folder, "fm_descriptors_merged.json")

fm_decriptors_merged_dict = dict()
with open(fm_descriptors_merged_json, "r") as f:
    fm_decriptors_merged_dict = json.load(f)

fm_decriptors_merged_array = fluid_dataset2array(fm_decriptors_merged_dict)
fm_decriptors_merged_array.shape

# %%

# fit umap to the dataset

reducer = umap.UMAP(n_neighbors=400, min_dist=0.1,
                    n_components=2, metric="euclidean", random_state=42)
embedding = reducer.fit_transform(fm_decriptors_merged_array)

# %%

# save the embedding to a npy file
np.save(os.path.join(dataset_folder, "fm_descriptors_umap_400.npy"), embedding)

# %%

# load the embedding from the npy file
embedding = np.load(os.path.join(
    dataset_folder, "fm_descriptors_umap_400.npy"))

# %%

# plot the embedding
plt.scatter(embedding[:, 0], embedding[:, 1], s=0.1)
plt.show()

# %%


# create json dataset with umap embedding
umap_ds = array2fluid_dataset(embedding)

# save the umap_ds dataset to a json file
with open(os.path.join(dataset_folder, "fm_descriptors_umap_400.json"), "w") as f:
    json.dump(umap_ds, f)

# %%
