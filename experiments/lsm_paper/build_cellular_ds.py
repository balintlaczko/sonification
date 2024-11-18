import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split


# process an image

def process_image(root_path, image_paths_rg, kernel_size=128, stride=64):
    img_path_r, img_path_g = image_paths_rg
    # load images as 16-bit grayscale
    img_r = cv2.imread(os.path.join(root_path, img_path_r),
                       cv2.IMREAD_UNCHANGED)
    img_g = cv2.imread(os.path.join(root_path, img_path_g),
                       cv2.IMREAD_UNCHANGED)
    row = {
        "path_r": img_path_r,
        "path_g": img_path_g,
        "r_min": img_r.min(),
        "r_max": img_r.max(),
        "g_min": img_g.min(),
        "g_max": img_g.max()
    }
    return row


if __name__ == '__main__':
    root_path = '/home/balint/cellular/images'
    # get all image paths
    image_paths = [img for img in os.listdir(
        root_path) if img.endswith(".TIF")]
    # sort the images
    image_paths.sort()
    # create the image pairs
    img_path_r = [img for img in image_paths if "_w2" in img]
    img_path_g = [img for img in image_paths if "_w1" in img]
    # create list of pairs
    image_paths_rg = list(zip(img_path_r, img_path_g))
    # process the images
    executor = ProcessPoolExecutor()
    jobs = [executor.submit(process_image, root_path, img_paths)
            for img_paths in image_paths_rg]
    results = []
    for job in tqdm(as_completed(jobs), total=len(jobs)):
        results.append(job.result())
    # train/test split
    train, val = train_test_split(results, test_size=0.2)
    # create the dataframes
    df_train = pd.DataFrame(train)
    df_train["dataset"] = "train"
    df_val = pd.DataFrame(val)
    df_val["dataset"] = "val"
    # concatenate the dataframes
    df = pd.concat([df_train, df_val])
    # save the dataframe
    df.to_csv("cellular.csv", index=False)
    print("Saved to cellular.csv")
