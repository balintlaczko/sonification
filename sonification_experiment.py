# %%
# imports
import numpy as np
from matplotlib import pyplot as plt
from utils import *
import musicalgestures

# %%
# create dataset
# path to folder
images_folder = r"C:\Users\Balint Laczko\Desktop\work\Sonification\210226_ RRI"
# parse folder into a dataset (pandas.DataFrame)
ds = folder2dataset(images_folder)
seq = ImageSequence(ds)  # class to interact with the dataset
# see the first bit of the dataset that is available at ImageSequence.sequence
seq.sequence.head()

# %%
# save dataset to csv
# save it next to our input folder
out_csv = os.path.join(os.path.dirname(images_folder), "new_images.csv")
seq.save_dataset(out_csv)

# %%
# load dataset
ds = pd.read_csv(
    r"C:\Users\Balint Laczko\Desktop\work\Sonification\new_images.csv", index_col=0)
ds.head()

# %%
# create esf sets
esf = ds.loc[ds["group"] == "esf"]
esf_rfp = esf.loc[esf["color_space"] == "rfp"]
esf_gfp = esf.loc[esf["color_space"] == "gfp"]
esf_rfp.head()

# %%
# create pbsga sets
pbsga = ds.loc[ds["group"] == "pbsga"]
pbsga_rfp = pbsga.loc[pbsga["color_space"] == "rfp"]
pbsga_gfp = pbsga.loc[pbsga["color_space"] == "gfp"]
pbsga_rfp.head()

# %%
# create pbsgb sets
pbsgb = ds.loc[ds["group"] == "pbsgb"]
pbsgb_rfp = pbsgb.loc[pbsgb["color_space"] == "rfp"]
pbsgb_gfp = pbsgb.loc[pbsgb["color_space"] == "gfp"]
pbsgb_rfp.head()

# %%
# functions to write starvation and refed videos


def write_starvation_videos(dataset, target_folder, prefix, fps, width, height, min_pix_val, max_pix_val):
    for vp in dataset["visiting_point"].unique():
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        target_filename = f'{prefix}_st_{vp}.avi'
        target_name = os.path.join(target_folder, target_filename)
        out = cv2.VideoWriter(target_name, fourcc, fps, (width, height), 0)
        bst_file = dataset.loc[(dataset["stage"] == "bst") & (
            dataset["visiting_point"] == vp)]["path"].values[0]
        bst_matrix = cv2.imread(bst_file, -1)
        bst_matrix_8bit = stretch_contrast(
            bst_matrix, in_min=min_pix_val, in_max=max_pix_val, out_max=255).astype(np.uint8)
        out.write(bst_matrix_8bit)
        st = dataset.loc[(dataset["stage"] == "st") &
                         (dataset["visiting_point"] == vp)]
        steps = st["step"].unique().tolist()
        for step in steps:
            row = st.loc[st["step"] == step]
            path = row["path"].values[0]
            st_matrix = cv2.imread(path, -1)
            st_matrix_8bit = stretch_contrast(
                st_matrix, in_min=min_pix_val, in_max=max_pix_val, out_max=255).astype(np.uint8)
            out.write(st_matrix_8bit)
    out.release()


def write_refed_videos(dataset, target_folder, prefix, fps, width, height, min_pix_val, max_pix_val):
    for vp in dataset["visiting_point"].unique():
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        target_filename = f'{prefix}_rf_{vp}.avi'
        target_name = os.path.join(target_folder, target_filename)
        out = cv2.VideoWriter(target_name, fourcc, fps, (width, height), 0)
        st = dataset.loc[(dataset["stage"] == "st") &
                         (dataset["visiting_point"] == vp)]
        steps = st["step"].unique().tolist()
        last_step = max(steps)
        st_file = st.loc[st["step"] == last_step]["path"].values[0]
        st_matrix = cv2.imread(st_file, -1)
        st_matrix_8bit = stretch_contrast(
            st_matrix, in_min=min_pix_val, in_max=max_pix_val, out_max=255).astype(np.uint8)
        out.write(st_matrix_8bit)
        rf = dataset.loc[(dataset["stage"] == "rf") &
                         (dataset["visiting_point"] == vp)]
        steps = rf["step"].unique().tolist()
        for step in steps:
            row = rf.loc[rf["step"] == step]
            path = row["path"].values[0]
            rf_matrix = cv2.imread(path, -1)
            rf_matrix_8bit = stretch_contrast(
                rf_matrix, in_min=min_pix_val, in_max=max_pix_val, out_max=255).astype(np.uint8)
            out.write(rf_matrix_8bit)
    out.release()


# %%
# find min and max pixel values in the different colorspaces
# rfp
all_rfp = ds.loc[ds["color_space"] == "rfp"]
min_rfp = min(all_rfp["pixel_min"].tolist())
max_rfp = max(all_rfp["pixel_max"].tolist())
# gfp
all_gfp = ds.loc[ds["color_space"] == "gfp"]
min_gfp = min(all_gfp["pixel_min"].tolist())
max_gfp = max(all_gfp["pixel_max"].tolist())

# %%
# render settings
width, height = ds["width"].tolist()[0], ds["height"].tolist()[0]
target_folder = r"C:\Users\Balint Laczko\Desktop\cell_videos"
fps = 10
datasets = [esf_rfp, esf_gfp, pbsga_rfp, pbsga_gfp, pbsgb_rfp, pbsgb_gfp]
dataset_names = ["esf_rfp", "esf_gfp", "pbsga_rfp",
                 "pbsga_gfp", "pbsgb_rfp", "pbsgb_gfp"]
min_pix_vals = [min_rfp, min_gfp, min_rfp, min_gfp, min_rfp, min_gfp]
max_pix_vals = [max_rfp, max_gfp, max_rfp, max_gfp, max_rfp, max_gfp]

# %%
# render all starvation videos
for i, dset in enumerate(datasets):
    write_starvation_videos(
        dset, target_folder, dataset_names[i], fps, width, height, min_pix_vals[i], max_pix_vals[i])

# %%
# render all refed videos
for i, dset in enumerate(datasets):
    write_refed_videos(
        dset, target_folder, dataset_names[i], fps, width, height, min_pix_vals[i], max_pix_vals[i])

# %%
# make motiongrams for each video
videos = [video for video in sorted(
    os.listdir(target_folder)) if video.endswith(".avi")]
for video in videos:
    musicalgestures.MgVideo(os.path.join(target_folder, video)).motiongrams()

# %%
# or make videograms for each video
videos = [video for video in sorted(
    os.listdir(target_folder)) if video.endswith(".avi")]
for video in videos:
    musicalgestures.MgVideo(os.path.join(target_folder, video)).videograms()

# %%
# sonify each motiongram/videogram
st_steps = 9
rf_steps = 16

images = [image for image in sorted(
    os.listdir(target_folder)) if image.endswith("png")]

for image in images:
    fname_parts = os.path.splitext(image)[0].split("_")
    timedim = "height" if fname_parts[-1] == "vgx" else "width"
    stage = fname_parts[2]
    outname = os.path.splitext(image)[0] + '.wav'
    if stage == "rf":
        image2sines(os.path.join(target_folder, image), os.path.join(target_folder, outname),
                    rf_steps * 6, num_sines=16, lowest_freq=110, harmonic=True, normalize=False, time_dim=timedim)
    elif stage == "st":
        image2sines(os.path.join(target_folder, image), os.path.join(target_folder, outname),
                    st_steps * 3, num_sines=16, lowest_freq=110, harmonic=True, normalize=False, time_dim=timedim)

# %%
videos = [video for video in sorted(
    os.listdir(target_folder)) if video.endswith(".avi")]
num_tiles_w = 8
num_tiles_h = 8

for video in videos:
    cap = cv2.VideoCapture(os.path.join(target_folder, video))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    w_increment = int(width / num_tiles_w)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    h_increment = int(height / num_tiles_h)
    tile_sums = np.zeros((num_tiles_w * num_tiles_h, num_frames))
    frame_id = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame[..., 0]  # only first channel
        tile_id = 0
        for i in range(num_tiles_h):
            for j in range(num_tiles_w):
                tile = frame[j*w_increment:(j+1)*w_increment,
                             i*h_increment:(i+1)*h_increment]
                tile_sums[tile_id, frame_id] = np.sum(tile)
                tile_id += 1
        frame_id += 1
    target_name = os.path.join(
        target_folder, f'{os.path.splitext(video)[0]}.npy')
    np.save(target_name, tile_sums)

# %%
# test
testy = np.load(r"C:\Users\Balint Laczko\Desktop\cell_videos\esf_gfp_rf_1.npy")

# %%
tile_files = [file for file in sorted(
    os.listdir(target_folder)) if file.endswith(".npy")]

for tile_file in tile_files[:1]:
    tiles = np.load(os.path.join(target_folder, tile_file))
    tiles_8bit = scale_array_auto(tiles, 0, 255).astype(np.uint8)
    target_name = os.path.join(
        target_folder, f'{os.path.splitext(tile_file)[0]}_img.png')
    cv2.imwrite(target_name, tiles_8bit)

# %%
# sonify each tile_img
st_steps = 9
rf_steps = 16

images = [image for image in sorted(os.listdir(
    target_folder)) if image.endswith("_img.png")]

for image in images:
    fname_parts = os.path.splitext(image)[0].split("_")[:-1]
    timedim = "height" if fname_parts[-1] == "vgx" else "width"
    stage = fname_parts[2]
    outname = os.path.splitext(image)[0] + '.wav'
    if stage == "rf":
        image2sines(os.path.join(target_folder, image), os.path.join(target_folder, outname),
                    rf_steps * 6, num_sines=64, lowest_freq=110, harmonic=True, normalize=True, time_dim=timedim)
    elif stage == "st":
        image2sines(os.path.join(target_folder, image), os.path.join(target_folder, outname),
                    st_steps * 3, num_sines=64, lowest_freq=110, harmonic=True, normalize=True, time_dim=timedim)


# %%
#########################################################################################
#################################### New proto stuff ####################################


# %%
# attempt to work with RGB images by removing the legend overlay
# (that prevents the stretch contrast algorithm to work)

rgb_img_path = "/Volumes/T7 Touch/Sonification/210223_tiff images/Fed/20210223_BST_MIP_ESF1_GFP.tif"
img_matrix = cv2.imread(rgb_img_path, -1)

# %%
# view it

view(img_matrix)

# %%
# histogram of cell values


plt.hist(img_matrix.flatten(), bins=np.arange(256))
plt.title("histogram")
plt.show()

# %%
# try stretch contrast
test_stretched = stretch_contrast(img_matrix, out_max=255, in_percentile=99.9)
view(test_stretched, swap_rb=False, scale=0.5, text="autophagy")


# %%
# erase fully white cells

white_cell_indices = np.where(np.all(img_matrix == (255, 255, 255), axis=-1))
print(np.transpose(white_cell_indices))

# %%
# get bounding box of affected area and view it

y, x = white_cell_indices
min_y, max_y = np.min(y), np.max(y)
min_x, max_x = np.min(x), np.max(x)
print(min_y, max_y, min_x, max_x)
affected_area = img_matrix[min_y:max_y+1, min_x:max_x+1]
view(affected_area)

# %%
# removing legend overlay

img_matrix_nolegend = img_matrix.copy()
img_matrix_nolegend[y, x] = 0

# %%
# now we can stretch contrast

img_matrix_stretched = stretch_contrast(
    img_matrix_nolegend, out_max=255, in_percentile=99.9)
view(img_matrix_stretched, swap_rb=False, scale=0.5, text="autophagy")

# %%
