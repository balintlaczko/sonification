import numpy as np
import pandas as pd
import os
import argparse
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_name", type=str,
                        default="./experiments/lsm_paper/sinewave.csv")
    parser.add_argument('--min_pitch', type=int, default=38)
    parser.add_argument('--max_pitch', type=int, default=86)
    parser.add_argument('--min_db', type=int, default=-24)
    parser.add_argument('--max_db', type=int, default=0)
    parser.add_argument('--n_samples', type=int, default=10000)
    parser.add_argument('--val_split', type=float, default=0.2)
    args = parser.parse_args()

    # parse inputs
    out_name = args.out_name
    min_pitch = args.min_pitch
    max_pitch = args.max_pitch
    min_db = args.min_db
    max_db = args.max_db
    n_samples = args.n_samples
    n_steps_per_axis = int(np.sqrt(n_samples))
    val_split = args.val_split

    # create the dataframe
    df = pd.DataFrame(columns=["pitch", "loudness", "dataset"])

    # generate all possible x and y coordinates
    pitches = np.linspace(min_pitch, max_pitch, n_steps_per_axis)
    print(pitches)
    loudnesses = np.linspace(min_db, max_db, n_steps_per_axis)
    print(loudnesses)
    # create the grid
    xx, yy = np.meshgrid(pitches, loudnesses)
    # generate all possible combinations
    all_combinations = np.vstack([xx.ravel(), yy.ravel()]).T

    # create train and validation sets
    train, val = train_test_split(all_combinations, test_size=val_split)
    print(f"train: {train.shape}, val: {val.shape}")

    # add the train set to the dataframe
    df_train = pd.DataFrame(train, columns=["pitch", "loudness"])
    df_train["dataset"] = "train"
    df = df._append(df_train)

    # add the val set to the dataframe
    df_val = pd.DataFrame(val, columns=["pitch", "loudness"])
    df_val["dataset"] = "val"
    df = df._append(df_val)

    # save the dataframe
    df.to_csv(out_name, index=False)
    print(f"saved to {out_name}")


if __name__ == "__main__":
    main()
