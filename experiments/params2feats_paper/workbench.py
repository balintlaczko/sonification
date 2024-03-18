# %%
# imports
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
df.to_csv("fm_synth_params.csv", index=True)

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
        fm_synth = fm_synth_2(self.dur * self.sr, self.sr,
                              f_carrier, harm_ratio, mod_idx)
        return fm_synth, row.freq, row.harm_ratio, row.mod_index


# %%
# create the dataset
fm_synth_ds = FmSynthDataset("fm_synth_params.csv", sr=sr, dur=dur)
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
# iterate rows from the dataset, extract features, save them in a dataframe
df_descriptors = pd.DataFrame()
timbre_list = []
for i in tqdm(range(len(df))):
    y, freq, ratio, index = fm_synth_ds[i]
    # print(y.shape, freq, ratio, index)
    # timbre = timbral_models.timbral_extractor(y, fs=sr, verbose=False)
    # timbral_hardness = timbral_models.timbral_hardness(y, fs=sr)
    # timbral_depth = timbral_models.timbral_depth(y, fs=sr)
    # timbral_brightness = timbral_models.timbral_brightness(y, fs=sr)
    # timbral_roughness = timbral_models.timbral_roughness(y, fs=sr)
    # timbral_warmth = timbral_models.timbral_warmth(y, fs=sr)
    # timbral_sharpness = timbral_models.timbral_sharpness(y, fs=sr)
    # timbral_booming = timbral_models.timbral_booming(y, fs=sr)
    wfm = Waveform(y, sr, 0.0)
    spectrum = SpectrumByFFT(wfm, 4096)
    timbre = {
        "index": i,
        "freq": freq,
        "harm_ratio": ratio,
        "mod_index": index,
        # "hardness": timbral_hardness,
        # "depth": timbral_depth,
        # "brightness": timbral_brightness,
        # "roughness": timbral_roughness,
        # "warmth": timbral_warmth,
        # "sharpness": timbral_sharpness,
        # "boominess": timbral_booming,
        "spectral_centroid": spectrum.spectral_centroid,
        "spectral_crest": spectrum.spectral_crest,
        "spectral_decrease": spectrum.spectral_decrease,
        "spectral_energy": spectrum.spectral_energy,
        "spectral_flatness": spectrum.spectral_flatness,
        "spectral_kurtosis": spectrum.spectral_kurtosis,
        "spectral_roll_off": spectrum.spectral_roll_off,
        "spectral_skewness": spectrum.spectral_skewness,
        "spectral_slope": spectrum.spectral_slope,
        "spectral_spread": spectrum.spectral_spread,
        "inharmonicity": spectrum.inharmonicity
    }
    # print(timbre)
    # df_descriptors = pd.concat(
    #     [df_descriptors, pd.DataFrame(timbre, index=[i])], axis=0)
    timbre_list.append(timbre)

# df_descriptors.head()
# df_descriptors.to_csv("fm_synth_descriptors.csv", index=True)

# %%


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
