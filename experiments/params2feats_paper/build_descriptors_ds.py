from functools import partial
from multiprocessing import Pool, cpu_count
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
    # print(i)
    return timbre


if __name__ == '__main__':
    num_workers = cpu_count() - 2
    with Pool(num_workers) as p:
        func = partial(extract_features, synths=fm_synth_ds,
                       sr=sr)
        results = p.map(func, range(len(fm_synth_ds)))

    # print(results)
    print("Finished extracting features")
    # save to json
    json_object = json.dumps(results, indent=4)

    outfile_path = "./experiments/params2feats_paper/fm_synth_spectral_features.json"
    with open(outfile_path, "w") as outfile:
        outfile.write(json_object)

    print("Features saved to json")
