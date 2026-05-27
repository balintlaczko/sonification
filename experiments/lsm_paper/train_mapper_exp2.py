import argparse
import os
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
from torch.utils.data import DataLoader
import random
import numpy as np
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from sonification.models.models import PlImgFactorVAE, PlFMFactorVAE, PlMapper, CompactLatentWrapper
from sonification.datasets import MNISTPairDataset
from torchvision import transforms
from sonification.utils.tensor import scale
import wandb


def main():
    parser = argparse.ArgumentParser()

    # model params
    parser.add_argument('--in_features', type=int, default=2, help='input size')
    parser.add_argument('--out_features', type=int, default=2, help='output size')
    parser.add_argument('--determine_inout_features', type=int, default=1, help='if non-zero, will determine the in and out features via CompactLatentWrapper')
    parser.add_argument('--hidden_layers_features', type=int, default=128, help='the size of the hidden layers')
    parser.add_argument("--n_res_block", type=int, default=16)
    parser.add_argument("--n_res_features", type=int, default=64)
    parser.add_argument("--d_hidden_size", type=int, default=32)
    parser.add_argument("--d_num_layers", type=int, default=5)

    # training params
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument('--train_epochs', type=int, default=10000000, help='number of training epochs')
    # locality loss params
    parser.add_argument('--locality_loss_type', type=str, default='mse', help='locality loss type: l1 or mse')
    parser.add_argument('--locality_weight', type=float, default=50)
    # mmd loss params
    parser.add_argument('--mmd_weight', type=float, default=50)
    # cycle consistency loss params
    parser.add_argument('--cycle_consistency_loss_type', type=str, default='mse', help='cycle consistency loss type: l1 or mse')
    parser.add_argument('--cycle_consistency_weight_start', type=float, default=10)
    parser.add_argument('--cycle_consistency_weight_end', type=float, default=10)
    parser.add_argument('--cycle_consistency_ramp_start_epoch', type=int, default=0, help='cycle consistency start epoch')
    parser.add_argument('--cycle_consistency_ramp_end_epoch', type=int, default=1)
    # # tc loss params
    parser.add_argument('--tc_weight_max', type=float, default=2, help='tc weight at the end of the warmup')
    parser.add_argument('--tc_weight_min', type=float, default=2, help='tc weight at the start of the warmup')
    parser.add_argument('--tc_start_epoch', type=int, default=0, help='the epoch at which to start the tc warmup from tc_weight_min to tc_weight_max')
    parser.add_argument('--tc_warmup_epochs', type=int, default=1, help='the number of epochs to warmup the tc weight')

    # optimizer params
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--lr_decay", type=float, default=0.85)
    parser.add_argument("--lr_d", type=float, default=0.000005)
    parser.add_argument("--lr_decay_d", type=float, default=0.85)

    # image model
    parser.add_argument('--img_model_ckpt_path', type=str, default='./ckpt/mnist_vae2/v3.5/v3.5_last_epoch=4215.ckpt', help='image model checkpoint path')

    # audio model
    parser.add_argument('--audio_model_ckpt_path', type=str, default='./ckpt/fm_vae/imv_v5.10/imv_v5.10_last_epoch=7950.ckpt', help='sound model checkpoint path')

    # checkpoint & logging
    parser.add_argument('--ckpt_path', type=str, default='./ckpt/mapper_exp2', help='checkpoint path')
    parser.add_argument('--ckpt_name', type=str, default='mapper_exp2_v0_test', help='checkpoint name')
    parser.add_argument('--logdir', type=str, default='./logs/mapper_exp2', help='log directory')

    # quick comment
    parser.add_argument('--comment', type=str, default='', help='add a comment if needed')

    args = parser.parse_args()

    # change relative path to absolute path
    logdir = os.path.abspath(args.logdir)
    logdir = os.path.join(logdir, args.ckpt_name)
    os.makedirs(logdir, exist_ok=True)
    print(f"Logging to {logdir}")
    args.logdir = logdir

    # load in model checkpoint and extract saved args
    ckpt = torch.load(args.img_model_ckpt_path, map_location='cpu')
    in_model_args = ckpt["hyper_parameters"]['args']

    # create in model with args and load state dict
    in_model = PlImgFactorVAE(in_model_args)
    in_model.load_state_dict(ckpt['state_dict'])
    in_model.eval()
    print("In model loaded")

    # load out model checkpoint and extract saved args
    ckpt = torch.load(args.audio_model_ckpt_path, map_location='cpu')
    out_model_args = ckpt["hyper_parameters"]['args']

    # create out model with args and load state dict
    out_model = PlFMFactorVAE(out_model_args)
    out_model.load_state_dict(ckpt['state_dict'])
    out_model.eval()
    print("Out model loaded")

    # create image dataset
    transform = transforms.Compose([
        transforms.Resize((in_model_args.img_size, in_model_args.img_size)),
        transforms.ToTensor()
    ])
    dataset = MNISTPairDataset(root='./data', train=False, download=True, transform=transform)
    # create image dataloader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4, persistent_workers=True)
    print("Image dataloader created")

    def get_fm_synth_batch(model, batch_size):
        # encode a batch of random fm synth waves with the out model
        norm_params, freqs, ratios, indices = model.sample_fm_params(batch_size)
        x = model.input_synth(freqs, ratios, indices).detach()
        # select a random slice of model.n_samples
        start_idx = torch.randint(0, model.sr - model.n_samples, (1,))
        x = x[:, start_idx:start_idx + model.n_samples]
        # add random phase flip
        phase_flip = torch.rand(batch_size, 1, device=model.device)
        phase_flip = torch.where(phase_flip > 0.5, 1, -1)
        x = x * phase_flip
        in_wf_slice = x.unsqueeze(1)
        # add random noise
        noise = torch.randn_like(x, device=model.device)
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        noise = noise * 2 - 1
        noise_coeff = torch.rand(batch_size, 1, device=model.device) * 0.001
        noise = noise * noise_coeff
        x = x + noise
        # forward pass
        # get mel spectrogram
        in_spec = model.mel_spectrogram(in_wf_slice.detach())
        # normalize it
        in_spec = scale(in_spec, in_spec.min(), in_spec.max(), 0, 1)
        return in_spec

    # determine in and out features if needed
    if args.determine_inout_features > 0:
        # do it for the image model
        print("Determining in features for the image model...")
        in_model.model = CompactLatentWrapper(in_model.model, in_model_args.latent_size)
        # analyze KLD on a batch of data
        data_batch = next(iter(dataloader))
        kld_per_dim, active_indices = in_model.model.analyze_kld(data_batch[0], threshold=0.1)
        print(f"KLD per dimension: {kld_per_dim}")
        print(f"Active latent dimensions: {active_indices}")
        args.in_features = len(active_indices)

        # do it for the audio model
        print("Determining out features for the audio model...")
        out_model.model = CompactLatentWrapper(out_model.model, out_model_args.latent_size)
        # generate a batch of input samples
        in_spec = get_fm_synth_batch(out_model, args.batch_size)
        kld_per_dim, active_indices = out_model.model.analyze_kld(in_spec, threshold=0.2)
        print(f"KLD per dimension: {kld_per_dim}")
        print(f"Active latent dimensions: {active_indices}")
        args.out_features = len(active_indices)

        print(f"Determined in features: {args.in_features}, out features: {args.out_features}")

    # create mapper model where args also references the in and out models
    args.in_model = in_model
    args.out_model = out_model
    print("Creating mapper model...")
    model = PlMapper(args)
    model.in_model.requires_grad_(False)
    model.out_model.requires_grad_(False)

    def cb_get_target_domain_latents():
        # encode a batch of random fm synth waves with the out model
        in_spec = get_fm_synth_batch(model.out_model, model.batch_size).to(model.device)
        # encode
        mu, logvar = model.out_model.model.encode(in_spec)
        # reparameterize
        z_real = model.out_model.model.reparameterize(mu, logvar)
        return z_real
    # set the callback for getting target domain latents in the mapper model
    model.callback_get_target_domain_latents = cb_get_target_domain_latents

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

    # checkpoint callbacks
    checkpoint_path = os.path.join(args.ckpt_path, args.ckpt_name)
    best_checkpoint_callback = ModelCheckpoint(
        monitor="loss",
        dirpath=checkpoint_path,
        filename=args.ckpt_name + "_best_{epoch:02d}-{loss:.4f}",
        save_top_k=1,
        mode="min",
    )
    last_checkpoint_callback = ModelCheckpoint(
        monitor="epoch",
        dirpath=checkpoint_path,
        filename=args.ckpt_name + "_last_{epoch:02d}",
        save_top_k=1,
        mode="max",
    )
    callbacks = [best_checkpoint_callback, last_checkpoint_callback]

    # create logger
    logger = WandbLogger(
        name=args.ckpt_name,
        project="mapper_exp1",
        save_dir=logdir,
        offline=False,
        settings=wandb.Settings(_disable_stats=True),
        )

    # create trainer
    trainer = Trainer(
        max_epochs=args.train_epochs,
        enable_checkpointing=True,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
    )

    hyperparams = vars(args).copy()
    trainer.logger.log_hyperparams(hyperparams)

    # train model
    try:
        resume_path = os.listdir(checkpoint_path)
    except FileNotFoundError:
        resume_path = None
    if resume_path:
        resume_path = sorted(resume_path)
        resume_path = os.path.join(checkpoint_path, resume_path[-1])
        print(f"Resuming from {resume_path}")
        trainer.fit(model, train_dataloaders=dataloader, ckpt_path=resume_path)
    else:
        trainer.fit(model, train_dataloaders=dataloader)


if __name__ == "__main__":
    fix_seed = 2025
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    torch.set_float32_matmul_precision('high')
    main()
