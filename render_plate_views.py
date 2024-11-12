# %%
# imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
from tqdm import tqdm
from sonification.utils.video import video_from_images

# %%

root_path = '/Volumes/T7RITMO/Sonification/Amani_230117/merged images'

# crawl the root path and collect all the .jpg files
image_paths = []
for root, dirs, files in os.walk(root_path):
    for file in files:
        if file.endswith(".jpg"):
            image_paths.append(os.path.join(root, file))

print(f"Found {len(image_paths)} images")

# %%
# create a dataframe from the filenames
rows = []
for image_path in tqdm(image_paths):
    filename = os.path.basename(image_path)
    filename = filename.replace("__MERGE_ADJ.jpg", "")
    filename = filename.replace("Timepoint_", "")
    filename = filename.replace("230117-", "")
    filename = filename.split("_")
    row = {
        "timepoint": int(filename[0]),
        "stage": filename[1],
        "well": filename[2],
        "site": int(filename[3][1:]),
        "image_path": image_path
    }
    rows.append(row)

df = pd.DataFrame(rows)
df.head()

# %%
# function to encode well id as a 2d coordinate
def well_to_xy(well, first_well="C03"):
    row = ord(well[0]) - ord(first_well[0])
    col = int(well[1:]) - int(first_well[1:])
    return row, col

# test the function
row, col = well_to_xy("C12")
print(row, col)

# %%
# get plate dimensions by encoding all well ids
rows = []
for well in df["well"].unique():
    row, col = well_to_xy(well)
    rows.append((row, col))
rows = np.array(rows)
plate_dims = rows.max(axis=0) + 1
print(plate_dims)

# %%
# create a 2d grid of images
def render_plate_img(df, plate_dims, timepoint, stage, site, target_path):
    fig, axs = plt.subplots(plate_dims[0], plate_dims[1], figsize=(plate_dims[1], plate_dims[0]), dpi=300)
    df_filtered = df[(df["timepoint"] == timepoint) & (df["stage"] == stage) & (df["site"] == site)]
    for i, row in df_filtered.iterrows():
        row_id, col_id = well_to_xy(row["well"])
        img = cv2.imread(row["image_path"])
        # convert from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axs[row_id, col_id].imshow(img)
        axs[row_id, col_id].axis("off")
    # remove whitespace between subplots
    plt.subplots_adjust(wspace=0, hspace=0)
    # make background transparent
    fig.patch.set_alpha(0)
    plt.tight_layout()
    plt.savefig(target_path, dpi=300)
    plt.close()

timepoint = 1
stage = "BST"
site = 1
render_folder = "rendered_plate_views"
os.makedirs(render_folder, exist_ok=True)
target_path = f"timepoint_{timepoint}_{stage}_s{site}.png"
# render_plate_img(df, plate_dims, timepoint, stage, site, os.path.join(render_folder, target_path))


# %%
# render all plate views
for stage in df["stage"].unique():
    df_stage = df[df["stage"] == stage]
    for timepoint in df_stage["timepoint"].unique():
        df_timepoint = df_stage[df_stage["timepoint"] == timepoint]
        for site in df_timepoint["site"].unique():
            target_path = f"timepoint_{timepoint}_{stage}_s{site}.png"
            # check if the file already exists
            if os.path.exists(os.path.join(render_folder, target_path)):
                print(f"Skipping {target_path}")
                continue
            render_plate_img(df, plate_dims, timepoint, stage, site, os.path.join(render_folder, target_path))
            print(f"Rendered {target_path}")

# %%
# render videos from the images

# create a list of all the images
images = os.listdir(render_folder)
images = [img for img in images if img.endswith(".png")]
# images = [os.path.join(render_folder, img) for img in images]

# stages_in_order = ["BST", "ST", "RF"]
stages_in_order = ["ST", "RF"]
# steps_per_stage = [1, 3, 7]
steps_per_stage = [3, 7]
n_sites = 3

# create a video for each site
for site in range(1, n_sites + 1):
    print(f"Rendering video for site s{site}")
    images_site = [img for img in images if f"s{site}" in img]
    images_site_ordered = []
    for idx, stage in enumerate(stages_in_order):
        print(f"    Adding stage {stage}")
        stage = f"_{stage}"
        images_stage = [img for img in images_site if stage in img]
        images_stage.sort()
        images_site_ordered.extend(images_stage)
    # target_video = os.path.join(render_folder, f"site_s{site}.avi")
    target_video = os.path.join(render_folder, f"site_s{site}_nobst.avi")
    video_from_images(images_site_ordered, render_folder, target_name=target_video, fps=1, overwrite=True, print_cmd=True)


# %%
