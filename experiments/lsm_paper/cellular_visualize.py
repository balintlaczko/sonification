# %%
# imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
from tqdm import tqdm
from sonification.utils.matrix import view
# from sonification.utils.video import video_from_images

# %%

images_folder = r'C:\Users\Balint Laczko\Desktop\work\Sonification\CELLULAR\images'

# crawl the root path and collect all the .TIF files
image_paths = []
for root, dirs, files in os.walk(images_folder):
    for file in files:
        if file.endswith(".TIF"):
            image_paths.append(os.path.join(root, file))

print(f"Found {len(image_paths)} images")

# %%
# create a dataframe from the filenames
rows = []
for image_path in tqdm(image_paths):
    filename = os.path.basename(image_path)
    filename = filename.replace("Timepoint_", "")
    filename = filename.replace(".TIF", "")
    filename = filename.split("_")
    row = {
        "timepoint": int(filename[0]),
        "date": filename[1].split("-")[0],
        "stage": filename[1].split("-")[1],
        "well": filename[2],
        "site": int(filename[3][1:]),
        # 0: green, 1: red, 2: brightfield
        "channel": int(filename[4][1:]) - 1,
        "image_path": image_path
    }
    rows.append(row)

df = pd.DataFrame(rows)
df.head()

# %%
# get the masks and metadata

masks_folder = r'C:\Users\Balint Laczko\Desktop\work\Sonification\CELLULAR\CELLULAR tracking\tracking test cell masks'

# crawl the root path and collect all the .png files
mask_paths = []
for root, dirs, files in os.walk(masks_folder):
    for file in files:
        if file.endswith(".png"):
            mask_paths.append(os.path.join(root, file))

print(f"Found {len(mask_paths)} masks")

# %%
# create a dataframe from the filenames
rows = []
for mask_path in tqdm(mask_paths):
    filename = os.path.basename(mask_path)
    filename = filename.replace("_w3_cp_mask.png", "")
    filename = filename.replace("Timepoint_", "")
    filename = filename.split("_")
    row = {
        "timepoint": int(filename[0]),
        "date": filename[1].split("-")[0],
        "stage": filename[1].split("-")[1],
        "well": filename[2],
        "site": int(filename[3][1:]),
        "mask_path": mask_path
    }
    rows.append(row)

df_masks = pd.DataFrame(rows)
df_masks.head()

# %%
# read mask tracking data
tracking_csv = r'C:\Users\Balint Laczko\Desktop\work\Sonification\CELLULAR\CELLULAR tracking\CELLULAR_tracking_220518Cells.csv'
df_tracking = pd.read_csv(tracking_csv)
df_tracking.head()

# %%
# extract the size of the largest bounding box
widths = df_tracking["AreaShape_BoundingBoxMaximum_X"] - \
    df_tracking["AreaShape_BoundingBoxMinimum_X"]
heights = df_tracking["AreaShape_BoundingBoxMaximum_Y"] - \
    df_tracking["AreaShape_BoundingBoxMinimum_Y"]
max_width = widths.max()
max_height = heights.max()
max_width, max_height

# %%
# filter tracked masks by id and visualize tracked mask over time

# filter by id
# idx = 2
# target_site = 1

# discard the lowest 5% of cell areas
area_threshold = df_tracking["AreaShape_Area"].quantile(0.5)
print(f"Discarding cells with area less than {area_threshold}")
df_tracking = df_tracking[df_tracking["AreaShape_Area"] > area_threshold]

all_tracked_ids = df_tracking["TrackObjects_Label_100"].unique()
print(f"Found {len(all_tracked_ids)} tracked objects")

# %%
# for each tracked object, visualize the tracking over time
for idx in all_tracked_ids:
    print(f"Visualizing tracked object {idx}")

    df_tracking_id = df_tracking[df_tracking["TrackObjects_Label_100"] == idx]

    # for each row fetch the corresponding images and masks

    for i, row in df_tracking_id.iterrows():
        timepoint = row['Metadata_Timepoint']
        well = row['Metadata_Well']
        date = str(row['Metadata_Date'])
        stage = row['Metadata_Condition']
        # get the site from the file name
        filename = row['Metadata_FileLocation']
        filename = os.path.basename(filename)
        filename = filename.replace("_w3_cp_masks.png", "")
        site = int(filename.split("_")[-1][1:])
        # if site != target_site:
        #     continue
        print(idx, timepoint, well, date, stage, site)
        # get the corresponding images
        images_df = df[
            (df["timepoint"] == timepoint) &
            (df["well"] == well) &
            (df["site"] == site) &
            (df["date"] == date) &
            (df["stage"] == stage)
        ]
        assert len(images_df) == 3  # green, red, brightfield
        # get the corresponding masks
        masks_df = df_masks[
            (df_masks["timepoint"] == timepoint) &
            (df_masks["well"] == well) &
            (df_masks["site"] == site) &
            (df_masks["date"] == date) &
            (df_masks["stage"] == stage)
        ]
        assert len(masks_df) == 1  # mask
        # read the images
        image_r_path = images_df[images_df["channel"]
                                 == 1]["image_path"].values[0]
        image_g_path = images_df[images_df["channel"]
                                 == 0]["image_path"].values[0]
        image_b_path = images_df[images_df["channel"]
                                 == 2]["image_path"].values[0]
        image_r = cv2.imread(image_r_path, cv2.IMREAD_UNCHANGED)
        image_g = cv2.imread(image_g_path, cv2.IMREAD_UNCHANGED)
        image_b = cv2.imread(image_b_path, cv2.IMREAD_UNCHANGED)
        # read the mask
        mask_path = masks_df["mask_path"].values[0]
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        # get object id for tracked label idx
        object_id = row["ObjectNumber"]
        # extract the mask for the object
        mask_object = mask.copy()
        mask_object[mask_object != object_id] = 0
        mask_object[mask_object == object_id] = 1
        # merge images into a 3-plane image
        image_b = np.zeros_like(image_r)
        image = np.stack([image_r, image_g, image_b], axis=-1)  # channels last
        # apply the mask
        image = image * mask_object[:, :, np.newaxis]
        # normalize the image
        image_min = image.min()
        image_max = image.max()
        image = (image - image_min) / (image_max - image_min)
        image *= 255
        image = image.astype(np.uint8)
        # rgb to bgr
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # visualize
        view(image, scale=0.25)


# %%
df_tracking_id = df_tracking[df_tracking["TrackObjects_Label_100"] == 11]

df_tracking_id = df_tracking_id[
    (df_tracking_id["Metadata_Timepoint"] == 5) &
    (df_tracking_id["Metadata_Well"] == "C06") &
    (df_tracking_id["Metadata_Date"] == 220518) &
    (df_tracking_id["Metadata_Condition"] == "ST")
]

df_tracking_id
# %%
