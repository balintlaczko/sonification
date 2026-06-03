# %%
# imports

import os
import torch
import numpy as np
from torchvision import transforms
from sonification.models.models import PlMapper
from sonification.datasets import MNISTPairDataset
from sonification.utils.tensor import scale
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
from pythonosc import udp_client, dispatcher, osc_server
import math

# %%
# find the latest mapper checkpoint
ckpt_path = '../../ckpt/mapper_exp2'
model_version = '1.9'
ckpt_name = 'v' + model_version
ckpt_path = os.path.join(ckpt_path, ckpt_name)
# list files, find the one that has "last" in it
ckpt_files = [f for f in os.listdir(ckpt_path) if 'last' in f]
if len(ckpt_files) == 0:
    raise ValueError(f"No checkpoint file found in {ckpt_path} with 'last' in the name")
ckpt_file = ckpt_files[0]
ckpt_path = os.path.join(ckpt_path, ckpt_file)
print(f"Checkpoint file found: {ckpt_path}")

# %%
# load checkpoint and extract saved args
mapper_ckpt = torch.load(ckpt_path, map_location='cpu')
args = mapper_ckpt["hyper_parameters"]['args']

# %%
# # create mapper model
model = PlMapper(args)
model.load_state_dict(mapper_ckpt['state_dict'])
# model.in_model.requires_grad_(False)
# model.out_model.requires_grad_(False)
model.eval()

def cb_synthesize_predicted_params(norm_predicted_params):
    # scale the predicted params
    predicted_params = model.out_model.scale_predicted_params(norm_predicted_params)
    # now repeat on the samples dimension
    predicted_freqs = predicted_params[:, 0].unsqueeze(1).repeat(1, model.out_model.sr)
    predicted_ratios = predicted_params[:, 1].unsqueeze(1).repeat(1, model.out_model.sr)
    predicted_indices = predicted_params[:, 2].unsqueeze(1).repeat(1, model.out_model.sr)
    # generate the output
    y = model.out_model.output_synth(predicted_freqs, predicted_ratios, predicted_indices)
    # select a random slice of model.n_samples
    start_idx = torch.randint(0, model.out_model.sr - model.out_model.n_samples, (1,))
    y = y[:, start_idx:start_idx + model.out_model.n_samples]
    y = y.unsqueeze(1)
    # get mel spectrogram
    in_spec = model.out_model.mel_spectrogram(y.detach())
    # normalize it
    in_spec = scale(in_spec, in_spec.min(), in_spec.max(), 0, 1)
    return in_spec
model.callback_post_decoder_hook = cb_synthesize_predicted_params
print("Mapper model created")

# %%
# create image dataset
transform = transforms.Compose([
    transforms.Resize((model.in_model.args.img_size, model.in_model.args.img_size)),
    transforms.ToTensor()
])
dataset_train = MNISTPairDataset(root='./data', train=True, download=True, transform=transform)
dataset_val = MNISTPairDataset(root='./data', train=False, download=True, transform=transform)
dataset = ConcatDataset([dataset_train, dataset_val])
print("Image dataset created")
# create image dataloader
batch_size = 128
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)#, num_workers=4, persistent_workers=True)
print("Image dataloader created")

# %%
# get the cuda/mps device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
# move model to device
model = model.to(device)
model.eval()
# in_model = in_model.to(device)
# in_model.eval()
model.in_model.eval()
# out_model = out_model.to(device)
# out_model.eval()
model.out_model.eval()
print(f"Using device: {device}")

# %%
# determine the latent size of the in and out models

# if they have an "active_indices" attribute, use the length of that
if hasattr(model.in_model.model, "active_indices") and model.in_model.model.active_indices is not None:
    in_latent_size = len(model.in_model.model.active_indices)
else:
    in_latent_size = model.in_model.args.latent_size
if hasattr(model.out_model.model, "active_indices") and model.out_model.model.active_indices is not None:
    out_latent_size = len(model.out_model.model.active_indices)
else:
    out_latent_size = model.out_model.args.latent_size
print(f"In model latent size: {in_latent_size}, out model latent size: {out_latent_size}")

# %%
# iterate through the dataloader and encode, map, decode, re-encode
z_1_all = torch.zeros(len(dataset), in_latent_size).to(device) # img encoded
z_2_all = torch.zeros(len(dataset), out_latent_size).to(device) # mapped
z_3_all = torch.zeros(len(dataset), out_latent_size).to(device) # audio re-encoded
with torch.no_grad():
    for batch_idx, data in tqdm(enumerate(dataloader)):
        x, _ = data
        mu, logvar = model.in_model.model.encode(x.to(device))
        z_1 = model.in_model.model.reparameterize(mu, logvar)
        # store
        z_1_all[batch_idx * batch_size: batch_idx * batch_size + batch_size, :] = z_1
        # map to audio latent space
        z_2 = model(z_1)
        z_2_all[batch_idx * batch_size: batch_idx * batch_size + batch_size, :] = z_2
        # decode to audio
        out_params = model.out_model.model.decoder(z_2)
        out_spec = model.callback_post_decoder_hook(out_params)
        # re-encode audio
        mu, logvar = model.out_model.model.encode(out_spec)
        z_3 = model.out_model.model.reparameterize(mu, logvar)
        z_3_all[batch_idx * batch_size: batch_idx * batch_size + batch_size, :] = z_3


# %%
# create a scatter plot of the mapped latent space
z_all = z_2_all.cpu().numpy()
num_dims = z_all.shape[1]
dim_pairs = list(itertools.combinations(range(num_dims), 2))
num_pairs = len(dim_pairs)

if num_pairs > 0:
    cols = math.ceil(math.sqrt(num_pairs))
    rows = math.ceil(num_pairs / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

    # Flatten the axes array to make it easy to iterate over 1D index
    axes = np.atleast_1d(axes).flatten()

    # Set font properties for the plot
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 12
    })

    # --- Calculate global min and max for consistent axis scaling ---
    global_min = z_all.min()
    global_max = z_all.max()
    # Add a 5% margin so points don't sit exactly on the plot boundaries
    margin = (global_max - global_min) * 0.05
    axis_min = global_min - margin
    axis_max = global_max + margin

    for i, (dim1, dim2) in enumerate(dim_pairs):
        ax = axes[i]

        # Consider lowering size (s) and adding alpha for better visibility if data is dense
        ax.scatter(z_all[:, dim1], z_all[:, dim2], s=1, alpha=0.5)
        ax.set_xlabel(f"Dim {dim1}")
        ax.set_ylabel(f"Dim {dim2}")
        ax.set_aspect('equal', adjustable='datalim') 

        # --- Apply the global axis limits ---
        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)
        
        # Optional: Disable axis ticks if it gets too cluttered
        # ax.set_xticks([])
        # ax.set_yticks([])

    # Hide any extra subplots in the grid that aren't used
    for i in range(num_pairs, len(axes)):
        axes[i].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.subplots_adjust(wspace=0.3) 
    plt.show()
    # plt.savefig("mnist_model_latent_space.png", dpi=300)

    # Reset rcParams to default to not affect other plots
    plt.rcdefaults()

# %%
# create a scatter plot color coded by distance between points in z_1 and corresponding points in z_2
# only if the latent spaces are the same size, otherwise we can't directly compare them
if z_1_all.shape[1] != z_2_all.shape[1]:
    print("Latent spaces have different sizes, cannot directly compare. Skipping scatter plot colored by mapping displacement.")
else:
    z_1_all_cpu = z_1_all.cpu().numpy()
    z_2_all_cpu = z_2_all.cpu().numpy()
    z_dist = np.linalg.norm(z_2_all_cpu - z_1_all_cpu, axis=1)
    z_diff_mean = np.mean(np.abs(z_2_all_cpu - z_1_all_cpu), axis=1)
    if num_pairs > 0:
        fig, axes = plt.subplots(num_pairs, 1, figsize=(10, 5 * num_pairs), squeeze=False)

        epoch_idx = ckpt_file.split('_')[-1].split('.')[0].split("=")[-1]
        fig.suptitle(f"Model v{model_version} - Epoch {epoch_idx} | Mapping Displacement")

        for i, (dim1, dim2) in enumerate(dim_pairs):
            ax = axes[i, 0]

            # Scatter plot colored by Mapping Displacement
            sc = ax.scatter(z_all[:, dim1], z_all[:, dim2], c=z_diff_mean, cmap='viridis', s=3)
            fig.colorbar(sc, ax=ax, label='Mapping Displacement', shrink=0.8)
            ax.set_title(f"Latent Dims {dim1} vs {dim2} by Mapping Displacement")
            ax.set_xlabel(f"Latent Dim {dim1}")
            ax.set_ylabel(f"Latent Dim {dim2}")
            ax.set_aspect('equal', adjustable='box')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

# %%
# create a scatter plot color coded by the diff between the relative positions of points in z_1 and corresponding points in z_2
num_points = 20000
z_1_dist = torch.cdist(z_1_all[:num_points], z_1_all[:num_points], p=2).cpu().numpy()
z_2_dist = torch.cdist(z_2_all[:num_points], z_2_all[:num_points], p=2).cpu().numpy()
# normalize the distance matrices to [0, 1]
z_1_dist = z_1_dist / np.mean(z_1_dist)
z_2_dist = z_2_dist / np.mean(z_2_dist)
# measure vector norm of the difference between the distance matrices
z_rel_diff = np.linalg.norm(z_2_dist - z_1_dist, axis=1)
if num_pairs > 0:
    fig, axes = plt.subplots(num_pairs, 1, figsize=(10, 5 * num_pairs), squeeze=False)

    epoch_idx = ckpt_file.split('_')[-1].split('.')[0].split("=")[-1]
    fig.suptitle(f"Model v{model_version} - Epoch {epoch_idx} | Locality Loss", fontsize=16)

    for i, (dim1, dim2) in enumerate(dim_pairs):
        ax = axes[i, 0]

        # Scatter plot colored by Locality Loss
        sc = ax.scatter(z_all[:num_points, dim1], z_all[:num_points, dim2], c=z_rel_diff, cmap='viridis', s=3)
        fig.colorbar(sc, ax=ax, label='Locality Loss', shrink=0.8)
        ax.set_title(f"Latent Dims {dim1} vs {dim2} by Locality Loss")
        ax.set_xlabel(f"Latent Dim {dim1}")
        ax.set_ylabel(f"Latent Dim {dim2}")
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# %%
# create a scatter plot color coded by the cycle consistency loss between z_2 and z_3
z_2_all_cpu = z_2_all.cpu().numpy()
z_3_all_cpu = z_3_all.cpu().numpy()
z_dist = np.linalg.norm(z_3_all_cpu - z_2_all_cpu, axis=1)
if num_pairs > 0:
    fig, axes = plt.subplots(num_pairs, 1, figsize=(10, 5 * num_pairs), squeeze=False)

    epoch_idx = ckpt_file.split('_')[-1].split('.')[0].split("=")[-1]
    fig.suptitle(f"Model v{model_version} - Epoch {epoch_idx} | Cycle Consistency Loss")

    for i, (dim1, dim2) in enumerate(dim_pairs):
        ax = axes[i, 0]

        # Scatter plot colored by Cycle Consistency Loss
        sc = ax.scatter(z_all[:, dim1], z_all[:, dim2], c=z_dist, cmap='viridis', s=3)
        fig.colorbar(sc, ax=ax, label='Cycle Consistency Loss', shrink=0.8)
        ax.set_title(f"Latent Dims {dim1} vs {dim2} by Cycle Consistency Loss")
        ax.set_xlabel(f"Latent Dim {dim1}")
        ax.set_ylabel(f"Latent Dim {dim2}")
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# %%
# measure the total correlation between the dimensions of z_2
z_2_all_cpu = z_2_all.cpu()
z_2_var = torch.var(z_2_all_cpu, dim=0)
z_2_std = torch.std(z_2_all_cpu, dim=0)
z_2_corr = torch.corrcoef(z_2_all_cpu.T)
z_2_tc = -0.5 * torch.log(torch.det(z_2_corr))
print(f"Total correlation of z_2: {z_2_tc.item()}, mean variance: {torch.mean(z_2_var).item()}, mean std: {torch.mean(z_2_std).item()}")


# %%
# per-dim scaling to 10-90th percentiles
percentile_low = 5
percentile_high = 95
num_dims = z_1_all.shape[1]
z_1_all_mins, z_1_all_maxs = torch.zeros(num_dims), torch.zeros(num_dims)
for dim in range(num_dims):
    dim_min = np.percentile(z_1_all[:, dim].cpu().numpy(), percentile_low)
    dim_max = np.percentile(z_1_all[:, dim].cpu().numpy(), percentile_high)
    z_1_all_mins[dim] = float(dim_min)
    z_1_all_maxs[dim] = float(dim_max)
z_1_all_mins, z_1_all_maxs

# %%
# set the FM synth model's maximum harmonicity ratio and modulation index to what is specified in the training params, not the starting one (because curriculum learning was used)
# check if they are buffers, then they must be fine (new versions) if not, then overwrite from args (old versions)
model.out_model.max_harm_ratio = torch.tensor(model.out_model.args.max_harm_ratio).to(device)
model.out_model.max_mod_idx = torch.tensor(model.out_model.args.max_mod_idx).to(device)


# %%
# create osc client
ip = "127.0.0.1"
port = 12345
client = udp_client.SimpleUDPClient(ip, port)

# create a function to generate an input image based on latent codes received from Max
# then encode, map, decode, and send back the predicted parameters to Max

def handle_img_latent_code(unused_addr, *args):
    # args is a tuple of the latent code
    latent_code = torch.tensor(args, dtype=torch.float32)
    # decode the latent code
    with torch.no_grad():
        # latent_code_scaled = scale(latent_code, -1, 1, z_all_min, z_all_max)
        latent_code_scaled = scale(latent_code, -1, 1, z_1_all_mins, z_1_all_maxs)
        predicted_images = model.in_model.model.decoder(latent_code_scaled.unsqueeze(0).to(device))

    # Flatten the image to a 1D list of floats 
    flat_image = predicted_images.squeeze().cpu().numpy().flatten().tolist()

    # send back the decoded images to Max
    client.send_message("/decoded_images", flat_image)

    # map the scaled latent code to the audio latent space and decode to predicted parameters
    with torch.no_grad():
        mapped_code = model(latent_code_scaled.unsqueeze(0).to(device))
        norm_predicted_params = model.out_model.model.decoder(mapped_code)
        scaled_predicted_params = model.out_model.scale_predicted_params(norm_predicted_params)

    # send back the predicted parameters to Max
    client.send_message("/fm_params", scaled_predicted_params.squeeze().cpu().numpy().tolist())

def handle_input_img(unused_addr, *args):
    # args is a tuple of the input image
    input_image = torch.tensor(args, dtype=torch.float32).reshape(1, 1, model.in_model.args.img_size, model.in_model.args.img_size)
    with torch.no_grad():
        # encode the input image to the latent space
        mean, logvar = model.in_model.model.encode(input_image.to(device))
        z = model.in_model.model.reparameterize(mean, logvar)
    # scale the latent code to -1 to 1 using the global min and max
    z_scaled = scale(z.squeeze().cpu(), z_1_all_mins, z_1_all_maxs, -1, 1)
    # send back the latent code to Max
    client.send_message("/img_latent_code", z_scaled.numpy().tolist())

# create an OSC receiver and start it
# create a dispatcher
d = dispatcher.Dispatcher()
d.map("/img_latent_code", handle_img_latent_code)
d.map("/input_img", handle_input_img)
# create a server
ip = "127.0.0.1"
port = 12346
server = osc_server.ThreadingOSCUDPServer(
    (ip, port), d)
print("Serving on {}".format(server.server_address))

try:
    server.serve_forever()
except KeyboardInterrupt:
    server.shutdown()
    server.server_close()
    print("Server stopped.")

# %%
