# %%
# imports
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
steps_per_axis = 10
pitches = np.linspace(38, 86, steps_per_axis)
freqs = midi2frequency(pitches)
ratios = np.linspace(0, 1, steps_per_axis) * 4
indices = np.linspace(0, 1, steps_per_axis) * 2
sr = 16000
dur = 2

# make into 3D mesh
freqs, ratios, indices = np.meshgrid(freqs, ratios, indices)

# %%
# create folder to save the files
folder = "fm_synth_sr16k_10steps_per_axis"
os.makedirs(folder, exist_ok=True)

# traverse through a line of indices while keeping the other two constant
for x in range(steps_per_axis):
    # create subfolder
    subfolder = f"{folder}/x_{x}"
    os.makedirs(subfolder, exist_ok=True)
    for y in range(steps_per_axis):
        # create subfolder
        subsubfolder = f"{subfolder}/y_{y}"
        os.makedirs(subsubfolder, exist_ok=True)
        for z in range(steps_per_axis):
            print(
                f"freq: {freqs[x, y, z]}, ratio: {ratios[x, y, z]}, index: {indices[x, y, z]}")
            # generate fm synth
            fm_synth = fm_synth_2(dur * sr, sr, np.array([freqs[x, y, z]]), np.array(
                [ratios[x, y, z]]), np.array([indices[x, y, z]]))
            # save to disk
            path = f"{subsubfolder}/fm_synth_f_{str(freqs[x, y, z]).zfill(3)}_r_{str(ratios[x, y, z]).zfill(3)}_i_{str(indices[x, y, z]).zfill(3)}.wav"
            wav.write(path, sr, fm_synth.astype(np.float32))

# %%
# same but create one folder per one change in the index
folder = "fm_synth_sr16k_10steps_per_axis_2"
os.makedirs(folder, exist_ok=True)

# traverse through a line of indices while keeping the other two constant
for z in range(steps_per_axis):
    # create subfolder
    subfolder = f"{folder}/z_{z}"
    os.makedirs(subfolder, exist_ok=True)
    for y in range(steps_per_axis):
        for x in range(steps_per_axis):
            print(
                f"freq: {freqs[x, y, z]}, ratio: {ratios[x, y, z]}, index: {indices[x, y, z]}")
            # generate fm synth
            fm_synth = fm_synth_2(dur * sr, sr, np.array([freqs[x, y, z]]), np.array(
                [ratios[x, y, z]]), np.array([indices[x, y, z]]))
            # save to disk
            path = f"{subfolder}/fm_synth_f_{str(freqs[x, y, z]).zfill(3)}_r_{str(ratios[x, y, z]).zfill(3)}_i_{str(indices[x, y, z]).zfill(3)}.wav"
            wav.write(path, sr, fm_synth.astype(np.float32))

# %%
# to use `vggish`
frechet = FrechetAudioDistance(
    ckpt_dir="./checkpoints/vggish",
    model_name="vggish",
    sample_rate=16000,
    use_pca=False,
    use_activation=False,
    verbose=False
)

# %%
# test fad score
baseline = "fm_synth_sr16k_10steps_per_axis_2/z_1/"
test = "fm_synth_sr16k_10steps_per_axis_2/z_2/"
# get files
baseline_files = [file for file in os.listdir(
    baseline) if not file.endswith(".DS_Store")]
test_files = [file for file in os.listdir(
    test) if not file.endswith(".DS_Store")]
print(len(baseline_files), len(test_files))

# %%
# cache embeddings paths
baseline_embeddings = "./embeddings/baseline_embeddings.npy"
test_embeddings = "./embeddings/test_embeddings.npy"

# get score
# score = frechet.score(baseline, test, baseline_embeddings, test_embeddings)
score = frechet.score(baseline, test)
print("FAD score:", score)

# %%
# evaluate FAD for all steps
root_folder = "fm_synth_sr16k_10steps_per_axis_2"

for i in range(steps_per_axis):
    print(f"Step {i}")
    baseline_folder_name = f"z_{i}"
    test_folder_name = f"z_{i+1}"
    print(baseline_folder_name, test_folder_name)
    baseline = os.path.join(root_folder, baseline_folder_name)
    test = os.path.join(root_folder, test_folder_name)
    baseline_embeddings = f"./embeddings/{root_folder}/{baseline_folder_name}_embeddings.npy"
    test_embeddings = f"./embeddings/{root_folder}/{test_folder_name}_embeddings.npy"
    score = frechet.score(baseline, test, baseline_embeddings, test_embeddings)
    print("FAD score:", score)

# %%
