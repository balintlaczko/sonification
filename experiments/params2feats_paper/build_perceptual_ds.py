from functools import partial
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
from pytimbre.waveform import Waveform
from pytimbre.spectral.spectra import SpectrumByFFT
import timbral_models
from torch.utils.data import Dataset
import pandas as pd
from sonification.datasets import FmSynthDataset
import json

# create the dataset
sr = 48000
dur = 1
csv_path = "./experiments/params2feats_paper/fm_synth_params.csv"
fm_synth_ds = FmSynthDataset(csv_path, sr=sr, dur=dur)
# print(len(fm_synth_ds))

# create progress bar
# pbar = tqdm(total=len(fm_synth_ds))

# extract features


def extract_features(i, synths, sr):
    y, freq, ratio, index = synths[i]
    # timbre = timbral_models.timbral_extractor(y, fs=sr, verbose=False)
    timbral_hardness = timbral_models.timbral_hardness(y, fs=sr)
    timbral_depth = timbral_models.timbral_depth(y, fs=sr)
    timbral_brightness = timbral_models.timbral_brightness(y, fs=sr)
    timbral_roughness = timbral_models.timbral_roughness(y, fs=sr)
    timbral_warmth = timbral_models.timbral_warmth(y, fs=sr)
    timbral_sharpness = timbral_models.timbral_sharpness(y, fs=sr)
    timbral_booming = timbral_models.timbral_booming(y, fs=sr)
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
    }
    # print(i)
    return timbre


if __name__ == '__main__':
    executor = ProcessPoolExecutor()
    jobs = [executor.submit(extract_features, idx, fm_synth_ds, sr)
            for idx in range(len(fm_synth_ds))]
    results = []
    for job in tqdm(as_completed(jobs), total=len(jobs)):
        results.append(job.result())

    # print(results)
    print("Finished extracting features")
    # save to json
    json_object = json.dumps(results, indent=4)

    outfile_path = "./experiments/params2feats_paper/fm_synth_perceptual_features.json"
    with open(outfile_path, "w") as outfile:
        outfile.write(json_object)

    print("Features saved to json")
