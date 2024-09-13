# imports
from torch.utils.data import DataLoader
from sonification.utils.array import fluid_dataset2array, array2fluid_dataset
from tqdm import tqdm
import torch
import pandas as pd
import json
from sys import stderr
from workbench import Args, FmMel2PCADataset, PlMelEncoder

n_mels = 400
args = Args()
args.sr = 48000
args.conv_in_channels = 1
args.conv_layers_channels = [64, 128, 256]
args.conv_in_size = n_mels
args.conv_out_features = 256
args.out_features = 2
args.proj_hidden_layers_features = [256, 256, 256]
args.lr = 1e-5
args.lr_decay = 0.999
args.ema_decay = 0
args.teacher_loss_weight = 0.0
args.triplet_loss_weight = 0.0
args.mel_triplet_loss_weight = 4
args.mel_triplet_miner_radius = 0.2
args.pitch_loss_weight = 0.0
args.plot_interval = 1
args.mode = "contrastive"
args.ckpt_path = "ckpt"
args.ckpt_name = "pca_finetuning_only_contrastive_11"
args.last_epoch = 506
args.resume_ckpt_path = f"{args.ckpt_path}/{args.ckpt_name}/{args.ckpt_name}_last_epoch={str(args.last_epoch).zfill(2)}.ckpt"
args.logdir = "logs"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps")
model = PlMelEncoder(args).to(device)
model.create_shadow_model()
resume_ckpt = torch.load(args.resume_ckpt_path, map_location=model.device)
model.load_state_dict(resume_ckpt['state_dict'])
model.eval()

csv_abspath = "./experiments/pca_finetuning/fm_synth_params.csv"
pca_abspath = "./experiments/pca_finetuning/pca_mels_mean.json"
train_split_path = "./experiments/pca_finetuning/train_split.pt"
train_split = torch.load(train_split_path)
df_fm = pd.read_csv(csv_abspath, index_col=0)
df_fm_train = df_fm.loc[list(train_split)]
pca_array = fluid_dataset2array(json.load(open(pca_abspath, "r")))
pca_train = pca_array[list(train_split)]
dataset_train = FmMel2PCADataset(
    df_fm_train, pca_train, sr=48000, dur=1, n_mels=n_mels)
dataset_train.fit_scalers()
dataset_full = FmMel2PCADataset(
    df_fm, pca_array, sr=48000, dur=1, n_mels=n_mels)
dataset_full.mel_scaler = dataset_train.mel_scaler
dataset_full.pca_scaler = dataset_train.pca_scaler
batch_size = 2048
dataset_full_sampler = torch.utils.data.BatchSampler(
    range(len(dataset_full)), batch_size=batch_size, drop_last=False)
dataset_full_loader = DataLoader(
    dataset_full, batch_size=None, sampler=dataset_full_sampler)

predictions = torch.zeros(len(dataset_full), 2).to(device)

for batch_idx, batch in enumerate(tqdm(dataset_full_loader)):
    mel, _ = batch
    B = mel.shape[0]
    with torch.no_grad():
        pred = model(mel.to(device))
    predictions[batch_idx*batch_size:batch_idx*batch_size + B] = pred
predictions = predictions.cpu().numpy()
predictions_fluid_dataset = array2fluid_dataset(predictions)
json_path = "./experiments/pca_finetuning/predictions_only_contrastive_11.json"
with open(json_path, "w") as f:
    json.dump(predictions_fluid_dataset, f)
print(f"Predictions saved to {json_path}", file=stderr)
