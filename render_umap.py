# %%
# imports
import os
import json
import numpy as np
import platform
from sonification.utils.array import array2fluid_dataset
import umap


# %%
# dataset parameters

dataset_folder = "/Volumes/T7/synth_dataset_2"
# if on Windows, use this path
if platform.system() == "Windows":
    dataset_folder = "D:/synth_dataset_2"

# %%

# load the dataset from the npy file

melspec = np.load(os.path.join(dataset_folder, "melspec_2_mean_std.npy"))

# %%

# fit umap to the dataset

reducer = umap.UMAP(n_neighbors=500, min_dist=0.1,
                    n_components=2, metric="euclidean", random_state=42)
embedding = reducer.fit_transform(melspec[..., 0])

# %%

# save the embedding to a npy file
np.save(os.path.join(dataset_folder, "melspec_2_umap_500.npy"), embedding)

# %%


# create json dataset with umap embedding
umap_ds = array2fluid_dataset(embedding)

# save the umap_ds dataset to a json file
with open(os.path.join(dataset_folder, "melspec_umap_500.json"), "w") as f:
    json.dump(umap_ds, f)

# %%
print("Done!")