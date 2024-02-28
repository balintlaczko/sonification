import argparse
import torch
from torch.utils.data import DataLoader
from sonification.utils.matrix import square_over_bg
from sonification.models.models import PlFactorVAE, PlFactorVAE1D, PlMapper
from sonification.datasets import Sinewave_dataset
from sklearn.neighbors import KDTree
from pythonosc import udp_client, dispatcher, osc_server


def main():
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--in_features', type=int,
                        default=2, help='input size')
    parser.add_argument('--out_features', type=int,
                        default=2, help='output size')
    parser.add_argument('--hidden_layers_features', type=int, nargs='*', default=[64, 128, 256, 512, 256, 128, 64],
                        help='the size of the hidden layers')

    # training
    parser.add_argument('--train_epochs', type=int,
                        default=50001, help='number of training epochs')
    # batch size is determined by the image dataset size, since it's small
    # parser.add_argument('--batch_size', type=int,
    #                     default=144, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.999)
    parser.add_argument('--locality_weight', type=float, default=10)
    parser.add_argument('--bounding_box_weight', type=float, default=100)
    parser.add_argument('--bounding_box_weight_decay',
                        type=float, default=0.98)
    parser.add_argument('--target_matching_weight', type=float, default=0)
    parser.add_argument('--target_matching_start', type=int,
                        default=300, help='target matching start epoch')
    parser.add_argument('--target_matching_warmup_epochs',
                        type=int, default=1000)
    parser.add_argument('--cycle_consistency_weight', type=float, default=10)
    parser.add_argument('--cycle_consistency_start', type=int,
                        default=300, help='cycle consistency start epoch')
    parser.add_argument('--cycle_consistency_warmup_epochs',
                        type=int, default=1000)
    # new
    parser.add_argument('--preserve_source_dist_weight',
                        type=float, default=50)
    parser.add_argument('--preserve_target_dist_weight',
                        type=float, default=10)
    parser.add_argument('--cycle_consistency_l1_weight', type=float, default=1)
    parser.add_argument('--centering_weight', type=float, default=0.001)

    # image model
    parser.add_argument('--img_model_ckpt_path', type=str,
                        default='./ckpt/white_squares_fvae_opt/factorvae-opt-v3/factorvae-opt-v3_last_epoch=258706.ckpt', help='image model checkpoint path')

    # audio model
    parser.add_argument('--audio_model_ckpt_path', type=str,
                        default='./ckpt/sinewave_fvae-opt/opt-v27/opt-v27_last_epoch=10405.ckpt', help='sound model checkpoint path')

    # checkpoint & logging
    parser.add_argument('--ckpt_path', type=str,
                        default='./ckpt/mapper', help='checkpoint path')
    parser.add_argument('--ckpt_name', type=str,
                        default='mapper-avstyle', help='checkpoint name')
    parser.add_argument('--resume_ckpt_path', type=str,
                        default="ckpt/mapper/mapper-clean-v1/mapper-clean-v1_last_epoch=37203.ckpt",)
    parser.add_argument('--logdir', type=str,
                        default='./logs/mapper', help='log directory')
    parser.add_argument('--plot_interval', type=int, default=10)

    # quick comment
    parser.add_argument('--comment', type=str, default='',
                        help='add a comment if needed')

    mapper_args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class Args:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    # load image model
    img_model_args = Args(
        # root path
        root_path='./experiments/lsm_paper/',
        # dataset
        csv_path='white_squares_xy_16_4.csv',
        img_size=16,
        square_size=4,
        # model
        in_channels=1,
        latent_size=2,
        layers_channels=[512, 256, 1024],
        d_hidden_size=512,
        d_num_layers=5,
        # training
        dataset_size=144,
        mmd_prior_distribution="gaussian",
        kld_weight=0.02,
        mmd_weight=0.02,
        tc_weight=2,
        l1_weight=0,
        onpix_weight=1,
        lr_vae=0.03,
        lr_decay_vae=0.99,
        lr_d=1e-4,
        lr_decay_d=0.99,
        plot_interval=1000,
    )

    ckpt_path = mapper_args.img_model_ckpt_path
    ckpt = torch.load(ckpt_path, map_location=device)
    img_model = PlFactorVAE(img_model_args).to(device)
    img_model.load_state_dict(ckpt['state_dict'])
    img_model.eval()

    # load audio model
    audio_model_args = Args(
        # root path
        root_path='./experiments/lsm_paper/',
        # dataset
        csv_path='sinewave.csv',
        img_size=64,
        # model
        in_channels=1,
        latent_size=2,
        layers_channels=[64, 128, 256, 512],
        d_hidden_size=512,
        d_num_layers=5,
        # training
        recon_weight=200,
        kld_weight=0.1,
        kld_start=0,
        kld_warmup_epochs=1,
        tc_weight=4,
        tc_start=0,
        tc_warmup_epochs=1,
        l1_weight=0.0,
        lr_d=0.01,
        lr_decay_d=0.999,
        lr_decay_vae=0.999,
        lr_vae=0.01,
        # checkpoint & logging
        ckpt_path='./ckpt/sinewave_fvae-opt',
        ckpt_name='opt-v33',
        logdir='./logs/sinewave_fvae-opt',
        plot_interval=1,
    )

    # create train and val datasets and loaders
    sinewave_ds_train = Sinewave_dataset(
        root_path=audio_model_args.root_path, csv_path=audio_model_args.csv_path, flag="train")
    sinewave_ds_val = Sinewave_dataset(
        root_path=audio_model_args.root_path, csv_path=audio_model_args.csv_path, flag="val", scaler=sinewave_ds_train.scaler)

    # load model
    ckpt_path = mapper_args.audio_model_ckpt_path
    ckpt = torch.load(ckpt_path, map_location=device)
    audio_model_args.train_scaler = sinewave_ds_train.scaler
    audio_model = PlFactorVAE1D(audio_model_args).to(device)
    audio_model.load_state_dict(ckpt['state_dict'])
    audio_model.eval()

    # pre-render the audio model latent space (mapper model needs it as a reference)
    batch_size = 256
    dataset = sinewave_ds_val
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, drop_last=False)
    audio_model_latent_space = torch.zeros(
        len(dataset), audio_model.args.latent_size).to(audio_model.device)
    for batch_idx, data in enumerate(loader):
        x, y = data
        x_recon, mean, logvar, z = audio_model.VAE(x.to(audio_model.device))
        z = z.detach()
        audio_model_latent_space[batch_idx *
                                 batch_size: batch_idx*batch_size + batch_size] = z

    # add missing args
    mapper_args.batch_size = img_model_args.dataset_size
    mapper_args.in_model = img_model
    mapper_args.out_model = audio_model
    mapper_args.out_latent_space = audio_model_latent_space

    # load trained model
    ckpt_path = mapper_args.resume_ckpt_path
    ckpt = torch.load(ckpt_path, map_location=device)
    model = PlMapper(mapper_args).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    # fit a KD tree to the scaled sinewave dataset
    sinewave_ds_all = Sinewave_dataset(root_path=audio_model_args.root_path, csv_path=audio_model_args.csv_path,
                                       flag="all", scaler=sinewave_ds_train.scaler)

    melbands = sinewave_ds_all.all_tensors.squeeze(-1).cpu().numpy()
    tree = KDTree(melbands)

    # create osc client
    ip = "127.0.0.1"
    port = 12347
    client = udp_client.SimpleUDPClient(ip, port)

    # create a function to generate an input image based on xy coordinates

    def handle_pictslider(unused_addr, x, y):
        # create the image
        img = square_over_bg(x, y, img_model_args.img_size,
                             img_model_args.square_size)
        # add a channel dimension
        img = img.unsqueeze(0).unsqueeze(0)
        # encode the image
        z_1 = img_model.encode(img.to(device))
        # project to audio latent space
        z_2 = model(z_1.to(device))
        # decode the audio
        mels_norm = audio_model.decode(z_2)
        # convert to numpy
        mels_norm = mels_norm.squeeze(1).detach().cpu().numpy()
        # query the KD tree
        _, idx = tree.query(mels_norm, k=1)
        idx = idx[0][0]
        # look up pitch and loudness from dataset
        row = sinewave_ds_all.df.iloc[idx]
        pitch = row["pitch"]
        loudness = row["loudness"]
        # send pitch and loudness to Max
        client.send_message("/sineparams", [pitch, loudness])

    # create an OSC receiver and start it
    # create a dispatcher
    d = dispatcher.Dispatcher()
    d.map("/pictslider", handle_pictslider)
    # create a server
    ip = "127.0.0.1"
    port = 12346
    server = osc_server.ThreadingOSCUDPServer(
        (ip, port), d)
    print("Serving on {}".format(server.server_address))
    server.serve_forever()


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    main()
