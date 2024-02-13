import numpy as np
import pandas as pd
import os
import argparse
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_name", type=str,
                        default="./experiments/lsm_paper/white_squares_xy_16_2.csv")
    parser.add_argument("--img_size", type=int, default=16)
    parser.add_argument("--square_size", type=int, default=2)
    parser.add_argument("--val_split", type=float, default=0.2)
    args = parser.parse_args()

    # parse inputs
    out_name = args.out_name
    img_size = args.img_size
    square_size = args.square_size
    val_split = args.val_split

    # create the dataframe
    df = pd.DataFrame(columns=["x", "y", "dataset"])

    # generate all possible x and y coordinates
    x = np.arange(0, img_size - square_size)
    y = np.arange(0, img_size - square_size)
    # create the grid
    xx, yy = np.meshgrid(x, y)
    # generate all possible combinations
    all_combinations = np.vstack([xx.ravel(), yy.ravel()]).T

    # create train and validation sets
    train, val = train_test_split(all_combinations, test_size=val_split)
    print(f"train: {train.shape}, val: {val.shape}")

    # add the train set to the dataframe
    df_train = pd.DataFrame(train, columns=["x", "y"])
    df_train["dataset"] = "train"
    df = df._append(df_train)

    # add the val set to the dataframe
    df_val = pd.DataFrame(val, columns=["x", "y"])
    df_val["dataset"] = "val"
    df = df._append(df_val)

    # save the dataframe
    df.to_csv(out_name, index=False)
    print(f"saved to {out_name}")


if __name__ == "__main__":
    main()
