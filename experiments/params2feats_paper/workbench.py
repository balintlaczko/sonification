# %%
# imports
from sonification.utils.array import array2fluid_dataset
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sonification.utils.dsp import frequency2midi
import importlib
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
import timbral_models
from pytimbre.spectral.spectra import SpectrumByFFT, Spectrum
from pytimbre.audio_files.wavefile import WaveFile
from pytimbre.waveform import Waveform
import soundfile as sf
import resampy
import torchaudio
import torch.nn as nn
import torch
from sonification.utils.video import video_from_images
from encodec import EncodecModel
import os
import numpy as np
from sonification.utils.dsp import fm_synth_2, midi2frequency
from scipy.io import wavfile as wav
from matplotlib import pyplot as plt
import pandas as pd
from frechet_audio_distance import FrechetAudioDistance
import json

# %%
# test generate some fm synth examples
sr = 48000
dur = 2
f_carrier = np.array([932.94])
harm_ratio = np.array([0.314])
mod_idx = np.array([0.54])

# %%
# generate fm synth
fm_synth = fm_synth_2(dur * sr, sr, f_carrier, harm_ratio, mod_idx)

# %%
# save to disk
path = "fm_synth.wav"
wav.write(path, sr, fm_synth.astype(np.float32))

# %%
# create ranges for each parameter
pitch_steps = 51
harm_ratio_steps = 51
mod_idx_steps = 51
pitches = np.linspace(38, 86, pitch_steps)
freqs = midi2frequency(pitches)  # x
ratios = np.linspace(0, 1, harm_ratio_steps) * 10  # y
indices = np.linspace(0, 1, mod_idx_steps) * 10  # z
sr = 48000
dur = 1

# make into 3D mesh
freqs, ratios, indices = np.meshgrid(freqs, ratios, indices)  # y, x, z!
print(freqs.shape, ratios.shape, indices.shape)

# %%
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
# save to disk
# df.to_csv("fm_synth_params.csv", index=True)

# %%
# create a Dataset that reads the csv and generates the fm synth buffer


class FmSynthDataset(Dataset):
    def __init__(self, csv_path, sr=48000, dur=1):
        self.df = pd.read_csv(csv_path)
        self.sr = sr
        self.dur = dur

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        f_carrier = np.array([row.freq])
        harm_ratio = np.array([row.harm_ratio])
        mod_idx = np.array([row.mod_index])
        fm_synth = fm_synth_2(int(self.dur * self.sr), self.sr,
                              f_carrier, harm_ratio, mod_idx)
        return fm_synth, row.freq, row.harm_ratio, row.mod_index


# %%
# create the dataset
fm_synth_ds = FmSynthDataset("fm_synth_params.csv", sr=sr, dur=0.5)
print(len(fm_synth_ds))

# %%
# test the dataset
fm_synth, freq, harm_ratio, mod_idx = fm_synth_ds[0]
print(fm_synth.shape, freq, harm_ratio, mod_idx)

# %%
# same but create one folder per one change in the index
folder = f"audio/fm_synth_sr16k_harm_ratio_{harm_ratio_steps}steps"
os.makedirs(folder, exist_ok=True)

# traverse through the meshgrid
for y in range(harm_ratio_steps):
    # create subfolder
    subfolder = f"{folder}/group_{y}"
    os.makedirs(subfolder, exist_ok=True)
    for x in range(pitch_steps):
        for z in range(mod_idx_steps):
            # generate fm synth
            fm_synth = fm_synth_2(dur * sr, sr, np.array([freqs[y, x, z]]), np.array(
                [ratios[y, x, z]]), np.array([indices[y, x, z]]))
            # save to disk
            path = f"{subfolder}/fm_synth_f_{str(freqs[y, x, z]).zfill(3)}_r_{str(ratios[y, x, z]).zfill(3)}_i_{str(indices[y, x, z]).zfill(3)}.wav"
            wav.write(path, sr, fm_synth.astype(np.float32))


# %%
# to use `vggish`
frechet = FrechetAudioDistance(
    ckpt_dir="./checkpoints/vggish",
    model_name="vggish",
    sample_rate=16000,
    use_pca=False,
    use_activation=False,
    verbose=True
)
# frechet.model.device = "mps"
# frechet.model.to("mps")

# %%
# use encodec
frechet = FrechetAudioDistance(
    ckpt_dir="./checkpoints/encodec",
    model_name="encodec",
    sample_rate=48000,
    channels=2,
    verbose=False
)
# frechet_encodec.model.device = "mps"
# frechet_encodec.model.to("mps")

# %%
# use pann
frechet = FrechetAudioDistance(
    ckpt_dir="./checkpoints/pann",
    model_name="pann",
    sample_rate=16000,
    verbose=False
)

# %%
# use clap
frechet = FrechetAudioDistance(
    ckpt_dir="./checkpoints/clap",
    model_name="clap",
    sample_rate=48000,
    verbose=True
)

# %%
# evaluate FAD for all steps
root_folder = f"audio/fm_synth_sr16k_harm_ratio_{harm_ratio_steps}steps"
group_folder = f"fm_synth_sr16k_harm_ratio_{harm_ratio_steps}steps"

scores = []
for i in range(harm_ratio_steps - 1):
    print(f"Step {i}")
    baseline_folder_name = f"group_{i}"
    test_folder_name = f"group_{i+1}"
    print(baseline_folder_name, test_folder_name)
    baseline = os.path.join(root_folder, baseline_folder_name)
    test = os.path.join(root_folder, test_folder_name)
    baseline_embeddings = f"./embeddings/{group_folder}/{baseline_folder_name}_embeddings.npy"
    test_embeddings = f"./embeddings/{group_folder}/{test_folder_name}_embeddings.npy"
    score = frechet.score(
        baseline, test, baseline_embeddings, test_embeddings)
    scores.append(score)
    print("FAD score:", score)

# %%
# plot scores
plt.plot(scores)
plt.xlabel("Step")
plt.ylabel("FAD score")
plt.title("FAD score for each step")
plt.show()

# %%
# evaluate FAD for all steps compared to first step
root_folder = f"audio/fm_synth_sr16k_harm_ratio_{harm_ratio_steps}steps"
group_folder = f"fm_synth_sr16k_harm_ratio_{harm_ratio_steps}steps"

scores = []
for i in range(harm_ratio_steps):
    print(f"Step {i}")
    baseline_folder_name = "group_0"
    test_folder_name = f"group_{i}"
    print(baseline_folder_name, test_folder_name)
    baseline = os.path.join(root_folder, baseline_folder_name)
    test = os.path.join(root_folder, test_folder_name)
    baseline_embeddings = f"./embeddings/{group_folder}/{baseline_folder_name}_embeddings.npy"
    test_embeddings = f"./embeddings/{group_folder}/{test_folder_name}_embeddings.npy"
    # score = frechet_encodec.score(
    #     baseline, test, baseline_embeddings, test_embeddings)
    score = frechet.score(
        baseline, test)
    scores.append(score)
    print("FAD score:", score)

# %%
# plot scores
plt.plot(scores)
plt.xlabel("Step")
plt.ylabel("FAD score")
plt.title("FAD score compared to first step")
plt.show()


# %%


# %%
filename = "/Users/balintl/Documents/GitHub/sonification/experiments/params2feats_paper/audio/fm_synth_sr16k_harm_ratio_50steps/group_40/fm_synth_f_466.1637615180899_r_3.265306122448979_i_2.0.wav"
wfm = WaveFile(filename)
spectrum = SpectrumByFFT(wfm, 4096)
print(spectrum.harmonic_energy, spectrum.noise_energy, spectrum.roughness)

# %%
spectrum.spe
# %%
filename = "/Users/balintl/Documents/GitHub/sonification/experiments/params2feats_paper/audio/fm_synth_sr16k_harm_ratio_50steps/group_46/fm_synth_f_99.90401989638436_r_3.7551020408163263_i_0.4444444444444444.wav"
timbre = timbral_models.timbral_extractor(filename)
timbre

# %%
# traverse through the meshgrid
synths = []
for y in range(harm_ratio_steps):
    for x in range(pitch_steps):
        for z in range(mod_idx_steps):
            # generate fm synth
            fm_synth = fm_synth_2(sr * 1, sr, np.array([freqs[y, x, z]]), np.array(
                [ratios[y, x, z]]), np.array([indices[y, x, z]]))
            synths.append(fm_synth)
        break
    break

# %%
# reload timbral models package
importlib.reload(timbral_models)

# %%
# iterate rows from the dataset, extract features, save them in a dataframe
timbre_list = []
for i in tqdm(range(100)):
    y, freq, ratio, index = fm_synth_ds[i]
    # print(y.shape, freq, ratio, index)
    # timbre = timbral_models.timbral_extractor(y, fs=sr, verbose=False)
    timbral_hardness = timbral_models.timbral_hardness(y, fs=sr)
    timbral_depth = timbral_models.timbral_depth(y, fs=sr)
    timbral_brightness = timbral_models.timbral_brightness(y, fs=sr)
    timbral_roughness = timbral_models.timbral_roughness(y, fs=sr)
    timbral_warmth = timbral_models.timbral_warmth(y, fs=sr)
    timbral_sharpness = timbral_models.timbral_sharpness(y, fs=sr)
    timbral_booming = timbral_models.timbral_booming(y, fs=sr)
    # wfm = Waveform(y, sr, 0.0)
    # spectrum = SpectrumByFFT(wfm, 4096)
    timbre = {
        "index": i,
        "freq": freq,
        "harm_ratio": ratio,
        "mod_index": index,
        "hardness": timbral_hardness,
        "depth": timbral_depth,
        "brightness": timbral_brightness,
        "roughness": timbral_roughness,
        "warmth": timbral_warmth,
        "sharpness": timbral_sharpness,
        "boominess": timbral_booming,
        # "spectral_centroid": spectrum.spectral_centroid,
        # "spectral_crest": spectrum.spectral_crest,
        # "spectral_decrease": spectrum.spectral_decrease,
        # "spectral_energy": spectrum.spectral_energy,
        # "spectral_flatness": spectrum.spectral_flatness,
        # "spectral_kurtosis": spectrum.spectral_kurtosis,
        # "spectral_roll_off": spectrum.spectral_roll_off,
        # "spectral_skewness": spectrum.spectral_skewness,
        # "spectral_slope": spectrum.spectral_slope,
        # "spectral_spread": spectrum.spectral_spread,
        # "inharmonicity": spectrum.inharmonicity
    }
    # print(timbre)
    # df_descriptors = pd.concat(
    #     [df_descriptors, pd.DataFrame(timbre, index=[i])], axis=0)
    timbre_list.append(timbre)

# %%
# create dataframe from list using the "index" key as index
df_perceptual = pd.DataFrame(timbre_list)
df_perceptual.set_index("index", inplace=True)
df_perceptual.head()

# df_descriptors.head()
# df_descriptors.to_csv("fm_synth_descriptors.csv", index=True)

# %%
# create the dataset
fm_synth_ds = FmSynthDataset("fm_synth_params.csv", sr=sr, dur=0.25)
print(len(fm_synth_ds))
test_fm = fm_synth_ds[0][0]
print(test_fm.shape)
test_embs = frechet.get_embeddings([test_fm], sr)
print(test_embs.shape)

# %%
# render all synths
all_y = np.zeros((len(fm_synth_ds), test_fm.shape[0]))
for i in tqdm(range(len(fm_synth_ds))):
    y, freq, ratio, index = fm_synth_ds[i]
    all_y[i] = y

# %%
# iterate all_y in chunks of batch size and render embeddings
all_embs = np.zeros((len(fm_synth_ds), test_embs.shape[0], test_embs.shape[1]))
batch_size = 64
for i in tqdm(range(0, len(fm_synth_ds), batch_size)):
    synths = all_y[i:i+batch_size]
    embs = frechet.get_embeddings(synths, sr)
    # reshape
    embs = embs.reshape((batch_size, -1, embs.shape[-1]))
    all_embs[i:i+batch_size] = embs

# %%
# iterate all_y one by one and render embeddings
all_embs = np.zeros((len(fm_synth_ds), test_embs.shape[0], test_embs.shape[1]))
for i in tqdm(range(len(fm_synth_ds))):
    synth = all_y[i]
    embs = frechet.get_embeddings([synth], sr)
    all_embs[i] = embs

# %%
# save all_embs to disk
print(all_embs.shape)
np.save("fm_synth_encodec_embeddings.npy", all_embs)
print("Saved fm_synth_encodec_embeddings.npy")

# %%
embs1 = frechet.get_embeddings(synths[:1], sr)
mu1, sigma1 = frechet.calculate_embd_statistics(embs1)

embs2 = frechet.get_embeddings(synths[1:2], sr)
mu2, sigma2 = frechet.calculate_embd_statistics(embs2)

fad = frechet.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
fad

# %%


# %%
np.mean(embs1, axis=1)
# %%
np.std(embs1, axis=1)
# %%
# %%
print(embs1.T.shape)
pca = PCA(n_components=30, whiten=True)
pca.fit(embs1.T)
print(pca.explained_variance_ratio_.sum())
# plot explained variance ratio
plt.plot(pca.explained_variance_ratio_)
plt.xlabel("Principal component")
plt.ylabel("Explained variance ratio")
plt.title("Explained variance ratio by principal component")
plt.show()
# %%
transformed = pca.transform(embs1.T)
print(transformed.shape)
print(transformed.mean(axis=0))
print(transformed.std(axis=0))

# %%
# read perceptual features from json
with open("fm_synth_perceptual_features.json", "r") as f:
    data = json.load(f)
df_perceptual = pd.DataFrame(data)
df_perceptual.set_index("index", inplace=True)
# order by index
df_perceptual.sort_index(inplace=True)
# save as csv
df_perceptual.to_csv("fm_synth_perceptual_features.csv", index=True)

# %%
# read spectral features from json
with open("fm_synth_spectral_features.json", "r") as f:
    data = json.load(f)
df_spectral = pd.DataFrame(data)
df_spectral.set_index("index", inplace=True)
# order by index
df_spectral.sort_index(inplace=True)
# save as csv
df_spectral.to_csv("fm_synth_spectral_features.csv", index=True)

# %%
# TODO:
# - save .npy files for each parameter, containing the step-to-step diff of features
# - save .npy files for each parameter, containing the step-to-step diff of embeddings (FAD)

# %%
# create a 2D scatter plot of the PCA-d synth parameters

# read dataset
df_params = pd.read_csv("fm_synth_params.csv", index_col=0)
# get the freq column
freq = df_params["freq"].values
# translate to midi
midi = frequency2midi(freq)
# add pitch column
df_params["pitch"] = midi
# extract pitch, harm_ratio, mod_index
df_params_3d = df_params[["pitch", "harm_ratio", "mod_index"]]

# scaling
scaler = MinMaxScaler()
# fit scaler
scaler.fit(df_params_3d)
# transform
df_params_3d = scaler.transform(df_params_3d)

# create PCA
pca = PCA(n_components=2, whiten=True, random_state=42)
# fit PCA
pca.fit(df_params_3d)
# transform
df_params_2d = pca.transform(df_params_3d)

# get scaled x y z for colors
x = df_params.x.values
y = df_params.y.values
z = df_params.z.values
# scale to between 0 and color_max
color_max = 0.9
x = (x - x.min()) / (x.max() - x.min()) * color_max
y = (y - y.min()) / (y.max() - y.min()) * color_max
z = (z - z.min()) / (z.max() - z.min()) * color_max
alpha = np.repeat(0.2, len(x))
colors = np.stack((x, y, z, alpha), axis=-1)

# create scatter plot with small dots and color by x, y, z as RGB
plt.scatter(df_params_2d[:, 0], df_params_2d[:, 1], s=1, c=colors)
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title(
    f"PCA of fm synth parameters, explained variance ratio: {np.round(pca.explained_variance_ratio_.sum(), 2)}")
plt.show()


# %%
# create a 2D scatter plot of the PCA-d perceptual features

# read dataset
df_perceptual = pd.read_csv("fm_synth_perceptual_features.csv", index_col=0)
# extract perceptual features
df_perceptual_7d = df_perceptual[[
    "hardness", "depth", "brightness", "roughness", "warmth", "sharpness", "boominess"]]

# replace inf values to the next highest in column
df_perceptual_7d_filtered = df_perceptual_7d.replace([np.inf, -np.inf], np.nan)
# get list of columns
cols = df_perceptual_7d_filtered.columns
# iterate through columns
for col in cols:
    # get the max value
    max_val = df_perceptual_7d_filtered[col].max()
    # replace nan values with max value
    df_perceptual_7d_filtered[col] = df_perceptual_7d_filtered[col].fillna(
        max_val)

# clip all columns to between nth and (1-n)th percentile
n = 0.05
for col in df_perceptual_7d_filtered.columns:
    # get 10th and 90th percentile
    low = df_perceptual_7d_filtered[col].quantile(n)
    high = df_perceptual_7d_filtered[col].quantile(1-n)
    # clip
    df_perceptual_7d_filtered.loc[:, col] = df_perceptual_7d_filtered[col].clip(
        low, high)

# scaling
scaler = MinMaxScaler()
# fit scaler
scaler.fit(df_perceptual_7d_filtered)
# transform
df_perceptual_7d_filtered_scaled = scaler.transform(df_perceptual_7d_filtered)
df_perceptual_7d_filtered_scaled = pd.DataFrame(
    df_perceptual_7d_filtered_scaled, columns=cols)
df_perceptual_7d_filtered_scaled.head()

# create PCA
pca = PCA(n_components=2, whiten=True, random_state=42)
# fit PCA
pca.fit(df_perceptual_7d_filtered_scaled)
# transform
df_perceptual_2d = pca.transform(df_perceptual_7d_filtered_scaled)
# create scatter plot with small dots
plt.scatter(df_perceptual_2d[:, 0], df_perceptual_2d[:, 1], s=1, c=colors)
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title(
    f"PCA of fm synth perceptual features, explained variance ratio: {np.round(pca.explained_variance_ratio_.sum(), 2)}")
plt.show()


# %%
# create a 2D scatter plot of the PCA-d spectral features

# read dataset
df_spectral = pd.read_csv("fm_synth_spectral_features.csv", index_col=0)

# extract spectral features
df_spectral_11d = df_spectral[[
    "spectral_centroid", "spectral_crest", "spectral_decrease", "spectral_energy", "spectral_flatness", "spectral_kurtosis", "spectral_roll_off", "spectral_skewness", "spectral_slope", "spectral_spread", "inharmonicity"]]

# translate spectral_roll_off, spectral_centroid, spectral_spread to midi (making a linear scale out of an exponential one)
for col in ["spectral_roll_off", "spectral_centroid", "spectral_spread"]:
    # get the column
    values = df_spectral_11d[col].values
    # translate to midi
    midi = frequency2midi(values)
    # replace values
    df_spectral_11d.loc[:, col] = midi

# clip all columns to between nth and (1-n)th percentile
n = 0.05
for col in df_spectral_11d.columns:
    # get 10th and 90th percentile
    low = df_spectral_11d[col].quantile(n)
    high = df_spectral_11d[col].quantile(1-n)
    # clip
    df_spectral_11d.loc[:, col] = df_spectral_11d[col].clip(low, high)

# scaling
scaler = MinMaxScaler()
# fit scaler
scaler.fit(df_spectral_11d)
# transform
df_spectral_11d_scaled = scaler.transform(df_spectral_11d)
df_spectral_11d_scaled = pd.DataFrame(
    df_spectral_11d_scaled, columns=df_spectral_11d.columns)
df_spectral_11d_scaled.head()

# create PCA
pca = PCA(n_components=2, whiten=True, random_state=42)
# fit PCA
pca.fit(df_spectral_11d_scaled)
# transform
df_spectral_2d = pca.transform(df_spectral_11d_scaled)
# create scatter plot with small dots
plt.scatter(df_spectral_2d[:, 0], df_spectral_2d[:, 1], s=1, c=colors)
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title(
    f"PCA of fm synth spectral features, explained variance ratio: {np.round(pca.explained_variance_ratio_.sum(), 2)}")
plt.show()


# %%
# combine df_spectral_11d_scaled and df_perceptual_7d_filtered_scaled
df_combined = pd.concat(
    [df_spectral_11d_scaled, df_perceptual_7d_filtered_scaled], axis=1)
df_combined.head()

# create PCA
pca = PCA(n_components=2, whiten=True, random_state=42)
# fit PCA
pca.fit(df_combined)
# transform
df_combined_2d = pca.transform(df_combined)
# create scatter plot with small dots
plt.scatter(df_combined_2d[:, 0], df_combined_2d[:, 1], s=1, c=colors)
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title(
    f"PCA of fm synth spectral and perceptual features, explained variance ratio: {np.round(pca.explained_variance_ratio_.sum(), 2)}")
plt.show()


# %%
# create pca plot for embeddings
# read embeddings
embeddings = np.load("fm_synth_encodec_embeddings.npy")
embeddings.shape

# %%
embeddings_2d = embeddings.reshape((embeddings.shape[0], -1))
embeddings_2d.shape

# %%
# create PCA
pca = PCA(n_components=2, whiten=True, random_state=42)
# fit PCA
pca.fit(embeddings_2d)
# transform
embeddings_2d_pca = pca.transform(embeddings_2d)
# create scatter plot with small dots
plt.scatter(embeddings_2d_pca[:, 0], embeddings_2d_pca[:, 1], s=1, c=colors)
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title(
    f"PCA of fm synth embeddings, explained variance ratio: {np.round(pca.explained_variance_ratio_.sum(), 2)}")
plt.show()


# %%
# save all of the pca plots and the colors array into fluid datasets

# save the colors array
colors_dict = array2fluid_dataset(colors)

# save the pca plots
pca_params = array2fluid_dataset(df_params_2d)
pca_perceptual = array2fluid_dataset(df_perceptual_2d)
pca_spectral = array2fluid_dataset(df_spectral_2d)
pca_combined = array2fluid_dataset(df_combined_2d)

# save them to json
with open("pca_params.json", "w") as f:
    json.dump(pca_params, f)
with open("pca_perceptual.json", "w") as f:
    json.dump(pca_perceptual, f)
with open("pca_spectral.json", "w") as f:
    json.dump(pca_spectral, f)
with open("pca_combined.json", "w") as f:
    json.dump(pca_combined, f)
with open("colors.json", "w") as f:
    json.dump(colors_dict, f)


# %%
# export fm params as fluid dataset
df_params_fm = df_params[["freq", "harm_ratio", "mod_index"]]
# convert to numpy array
df_params_fm = df_params_fm.values
# save as fluid dataset
df_params_fm_dict = array2fluid_dataset(df_params_fm)
# save to json
with open("fm_params.json", "w") as f:
    json.dump(df_params_fm_dict, f)

# %%
