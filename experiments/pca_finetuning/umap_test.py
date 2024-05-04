# %%
# imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import mse_loss
import torchvision
from torchvision.transforms import transforms
from lightning.pytorch import LightningModule, LightningDataModule, Trainer
import umap
from umap.umap_ import fuzzy_simplicial_set, find_ab_params
from pynndescent import NNDescent
import json
from sklearn.utils import check_random_state
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KDTree
from sonification.utils.array import array2fluid_dataset
from sonification.models.layers import ConvEncoder1D

# %%
mel_path = "../params2feats_paper/fm_synth_mel_spectrograms_mean.npy"
mels = np.load(mel_path)
mels.shape

# %%
scaler = MinMaxScaler()
mels_scaled = scaler.fit_transform(mels)

# %%
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=400,
    min_dist=0.1)
embedding = reducer.fit_transform(mels_scaled)

# %%
embedding_outname = "fm_synth_mel_spectrograms_mean_umap.npy"
np.save(embedding_outname, embedding)

# %%
# save as fluid dataset to json
fluid_dataset = array2fluid_dataset(embedding)
fluid_dataset_outname = "fm_synth_mel_spectrograms_mean_umap.json"
with open(fluid_dataset_outname, "w") as f:
    json.dump(fluid_dataset, f)

# %%
# plot embedding
plt.figure(figsize=(10, 10))
plt.scatter(embedding[:, 0], embedding[:, 1], s=1)
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of FM synth mel spectrograms')
plt.show()

# %%


def convert_distance_to_probability(distances, a=1.0, b=1.0):
    # return 1.0 / (1.0 + a * distances ** (2 * b))
    return -torch.log1p(a * distances ** (2 * b))


def compute_cross_entropy(
    probabilities_graph, probabilities_distance, EPS=1e-4, repulsion_strength=1.0
):
    # cross entropy
    attraction_term = -probabilities_graph * torch.nn.functional.logsigmoid(
        probabilities_distance
    )
    repellant_term = (
        -(1.0 - probabilities_graph)
        * (torch.nn.functional.logsigmoid(probabilities_distance)-probabilities_distance)
        * repulsion_strength
    )

    # balance the expected losses between atrraction and repel
    CE = attraction_term + repellant_term
    return attraction_term, repellant_term, CE


def umap_loss(embedding_to, embedding_from, _a, _b, batch_size, negative_sample_rate=5, device=None):
    # get negative samples by randomly shuffling the batch
    embedding_neg_to = embedding_to.repeat(negative_sample_rate, 1)
    repeat_neg = embedding_from.repeat(negative_sample_rate, 1)
    embedding_neg_from = repeat_neg[torch.randperm(repeat_neg.shape[0])]
    distance_embedding = torch.cat((
        (embedding_to - embedding_from).norm(dim=1),
        (embedding_neg_to - embedding_neg_from).norm(dim=1)
    ), dim=0)

    # convert probabilities to distances
    probabilities_distance = convert_distance_to_probability(
        distance_embedding, _a, _b
    )
    # set true probabilities based on negative sampling
    probabilities_graph = torch.cat(
        (torch.ones(batch_size), torch.zeros(batch_size * negative_sample_rate)), dim=0,
    )

    # compute cross entropy
    (attraction_loss, repellant_loss, ce_loss) = compute_cross_entropy(
        probabilities_graph.to(device),
        probabilities_distance.to(device),
    )
    loss = torch.mean(ce_loss)
    return loss


def get_umap_graph(X, n_neighbors=10, metric="cosine", random_state=None):
    random_state = check_random_state(
        None) if random_state is None else random_state
    # number of trees in random projection forest
    n_trees = 5 + int(round((X.shape[0]) ** 0.5 / 20.0))
    # max number of nearest neighbor iters to perform
    n_iters = max(5, int(round(np.log2(X.shape[0]))))
    # distance metric

    # get nearest neighbors
    nnd = NNDescent(
        X.reshape((len(X), np.product(np.shape(X)[1:]))),
        n_neighbors=n_neighbors,
        metric=metric,
        n_trees=n_trees,
        n_iters=n_iters,
        max_candidates=60,
        verbose=True
    )

    # get indices and distances
    knn_indices, knn_dists = nnd.neighbor_graph
    print("knn_indices.shape", knn_indices.shape)
    print("knn_dists.shape", knn_dists.shape)
    # build fuzzy_simplicial_set
    umap_graph, sigmas, rhos = fuzzy_simplicial_set(
        X=X,
        n_neighbors=n_neighbors,
        metric=metric,
        random_state=random_state,
        knn_indices=knn_indices,
        knn_dists=knn_dists,
    )

    return umap_graph


# %%


def get_graph_elements(graph_, n_epochs):

    graph = graph_.tocoo()
    # eliminate duplicate entries by summing them together
    graph.sum_duplicates()
    # number of vertices in dataset
    n_vertices = graph.shape[1]
    # get the number of epochs based on the size of the dataset
    if n_epochs is None:
        # For smaller datasets we can use more epochs
        if graph.shape[0] <= 10000:
            n_epochs = 500
        else:
            n_epochs = 200
    # remove elements with very low probability
    graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
    graph.eliminate_zeros()
    # get epochs per sample based upon edge probability
    epochs_per_sample = n_epochs * graph.data

    head = graph.row
    tail = graph.col
    weight = graph.data

    return graph, epochs_per_sample, head, tail, weight, n_vertices


class UMAPDataset(Dataset):
    def __init__(self, data, graph_, n_epochs=200):
        graph, epochs_per_sample, head, tail, weight, n_vertices = get_graph_elements(
            graph_, n_epochs)

        self.edges_to_exp, self.edges_from_exp = (
            np.repeat(head, epochs_per_sample.astype("int")),
            np.repeat(tail, epochs_per_sample.astype("int")),
        )
        shuffle_mask = np.random.permutation(np.arange(len(self.edges_to_exp)))
        self.edges_to_exp = self.edges_to_exp[shuffle_mask].astype(np.int64)
        self.edges_from_exp = self.edges_from_exp[shuffle_mask].astype(
            np.int64)
        self.data = torch.Tensor(data)

    def __len__(self):
        return int(self.data.shape[0])

    def __getitem__(self, index):
        edges_to_exp = self.data[self.edges_to_exp[index]]
        edges_from_exp = self.data[self.edges_from_exp[index]]
        return (edges_to_exp, edges_from_exp)


# %%


class conv(nn.Module):
    def __init__(self, n_components=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1,
            ),
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1,
            ),
            nn.Flatten(),
            nn.Linear(6272, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, n_components)
        )

    def forward(self, X):
        return self.encoder(X)


# %%
device = torch.device("cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu")
model = conv(2).to(device)
print(model.parameters)
print(model(torch.randn((12, 1, 28, 28)).to(device)).shape)

# %%

""" Model """


class Model(LightningModule):
    def __init__(
        self,
        lr: float,
        encoder: nn.Module,
        decoder=None,
        beta=1.0,
        min_dist=0.1,
    ):
        super().__init__()
        self.lr = lr
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta  # weight for reconstruction loss
        self._a, self._b = find_ab_params(1.0, min_dist)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        (edges_to_exp, edges_from_exp) = batch
        embedding_to, embedding_from = self.encoder(
            edges_to_exp), self.encoder(edges_from_exp)
        encoder_loss = umap_loss(embedding_to, embedding_from, self._a,
                                 self._b, edges_to_exp.shape[0], negative_sample_rate=5, device=self.device)
        self.log("umap_loss", encoder_loss)

        if self.decoder is not None:
            recon = self.decoder(embedding_to)
            recon_loss = mse_loss(recon, edges_to_exp)
            self.log("recon_loss", recon_loss)
            return encoder_loss + self.beta * recon_loss
        else:
            return encoder_loss


""" Datamodule """


class Datamodule(LightningDataModule):
    def __init__(
        self,
        dataset,
        batch_size,
        num_workers,
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )


class PUMAP():
    def __init__(
        self,
        encoder,
        decoder=None,
        n_neighbors=10,
        min_dist=0.1,
        metric="euclidean",
        lr=1e-3,
        epochs=30,
        batch_size=64,
        num_workers=1,
        random_state=None,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_state = random_state

    def fit(self, X):
        trainer = Trainer(accelerator='gpu', devices=1, max_epochs=self.epochs)
        self.model = Model(self.lr, self.encoder, min_dist=self.min_dist)
        graph = get_umap_graph(X, n_neighbors=self.n_neighbors,
                               metric=self.metric, random_state=self.random_state)
        trainer.fit(
            model=self.model,
            datamodule=Datamodule(UMAPDataset(X, graph),
                                  self.batch_size, self.num_workers)
        )

    @torch.no_grad()
    def transform(self, X):
        self.embedding_ = self.model.encoder(X).detach().cpu().numpy()
        return self.embedding_

    @torch.no_grad()
    def inverse_transform(self, Z):
        return self.model.decoder(Z).detach().cpu().numpy()

# %%


train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, transform=transforms.ToTensor(), download=True)
train_tensor = torch.stack([example[0] for example in train_dataset])[
    :, 0][:, None, ...]
labels = [str(example[1]) for example in train_dataset]
X = train_tensor

# %%
pumap = PUMAP(conv(2), lr=1e-3, epochs=10, num_workers=0)
pumap.fit(X)

# %%
embedding = pumap.transform(X)

# %%

sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=labels, s=0.4)

# %%
X.shape

# %%
mel_path = "../params2feats_paper/fm_synth_mel_spectrograms_mean.npy"
mels = np.load(mel_path)
mels.shape

# %%
scaler = MinMaxScaler()
mels_scaled = scaler.fit_transform(mels)
mels_scaled = mels_scaled.reshape(
    (mels_scaled.shape[0], 1, mels_scaled.shape[-1]))

# %%
mels_scaled.shape

# %%
encoder = ConvEncoder1D(
    in_channels=1,
    output_size=2,
    layers_channels=[128, 256, 512],
    input_size=mels_scaled.shape[-1]
)
pumap = PUMAP(
    encoder,
    n_neighbors=100,
    lr=1e-4,
    batch_size=256,
    epochs=20,
    num_workers=0)
pumap.fit(mels_scaled)

# %%
pumap.model.to(device)
embedding = pumap.transform(torch.Tensor(mels_scaled).to(device))
embedding.shape

# %%
sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1])

# %%
tree = KDTree(
    mels_scaled[:, 0, :]
)

# %%
idx = np.random.randint(0, mels_scaled.shape[0])
print(idx)
query_point = mels_scaled[idx, 0, :]
query_point = query_point.reshape((1, -1))
dist, ind = tree.query(
    query_point,
    k=10,
    return_distance=True,
)
print(ind.shape)
print(dist.shape)
print(dist)
print(ind)

# %%
idx = np.random.randint(0, mels_scaled.shape[0])
print(idx)
query_point = mels_scaled[idx, 0, :]
query_point = query_point.reshape((1, -1))
ind, dist = tree.query_radius(
    query_point,
    r=0.8,
    return_distance=True,
    count_only=False
)
print(ind.shape)
print(dist.shape)
print(dist)
print(ind)

# %%
a = torch.rand(4, 200)
b = torch.rand(4, 200)
dist = torch.cdist(a, b)
print(dist.shape)
print(dist)


# %%
def mine_triplets_based_on_mels(fake_mels, fake_embeddings, r=0.1):
    mels = fake_mels  # (B, 1, n_mels)
    embeddings = fake_embeddings  # (B, 2)
    # get the distance matrix of the embeddings
    dist = torch.cdist(embeddings, embeddings)
    B = embeddings.shape[0]
    anchor_ids = []
    positive_ids = []
    negative_ids = []
    for i in range(B):
        p0_dist = dist[i]
        # print(p0_dist)
        mask = p0_dist < r
        # print(mask)
        # get the indices of the embeddings that are within the radius
        indices = torch.nonzero(mask, as_tuple=True)[0]
        if len(indices) < 3:
            continue
        # print(i)
        # print(indices)
        mels_within_radius = mels[indices]
        # print(mels_within_radius.shape)
        mels_within_radius = mels_within_radius.squeeze(1)
        dist_mels = torch.cdist(mels[i], mels_within_radius)
        # print(dist_mels)
        sort_by_dist = torch.argsort(dist_mels.squeeze(0))
        # print(sort_by_dist)
        id_of_closest, id_of_farthest = sort_by_dist[1], sort_by_dist[-1]
        # print(id_of_closest, id_of_farthest)
        id_of_closest, id_of_farthest = indices[id_of_closest], indices[id_of_farthest]
        # print(id_of_closest, id_of_farthest)
        # print(
        #     f"For point {i} as anchor, positive sample is {id_of_closest} and negative sample is {id_of_farthest}")
        # print()
        anchor_ids.append(i)
        positive_ids.append(id_of_closest.item())
        negative_ids.append(id_of_farthest.item())
    return torch.tensor(anchor_ids), torch.tensor(positive_ids), torch.tensor(negative_ids)


# %%
fake_mels = torch.rand(512, 1, 200)
fake_embeddings = torch.rand(512, 2)
anchor_ids, pos_ids, neg_ids = mine_triplets_based_on_mels(
    fake_mels, fake_embeddings, r=0.02)
print(len(anchor_ids))
if len(anchor_ids) > 0:
    anchor, positive, negative = fake_embeddings[anchor_ids], fake_embeddings[pos_ids], fake_embeddings[neg_ids]
    d_ap = torch.norm(anchor - positive, dim=1)
    d_an = torch.norm(anchor - negative, dim=1)
    margin = 0.05
    triplet_loss = torch.nn.functional.relu(d_ap - d_an + margin).mean()
    print(triplet_loss)

# %%


def test_kd_tree_loss(fake_mels, fake_embeddings, r=0.1):
    mels = fake_mels  # (B, 1, n_mels)
    embeddings = fake_embeddings  # (B, 2)
    # get the distance matrix of the embeddings
    dist = torch.cdist(embeddings, embeddings)
    p0_dist = dist[0]
    print(p0_dist)
    mask = p0_dist < r
    # print(mask)
    # get the indices of the embeddings that are within the radius
    indices = torch.nonzero(mask, as_tuple=True)[0]
    print(indices)
    if len(indices) < 3:
        return
    mels_within_radius = mels[indices]
    print(mels_within_radius.shape)
    mels_within_radius = mels_within_radius.squeeze(1)
    dist_mels = torch.cdist(
        mels_within_radius[0].unsqueeze(0), mels_within_radius)
    print(dist_mels)
    sort_by_dist = torch.argsort(dist_mels.squeeze(0))
    print(sort_by_dist)
    id_of_closest, id_of_farthest = sort_by_dist[1], sort_by_dist[-1]
    print(id_of_closest, id_of_farthest)
    id_of_closest, id_of_farthest = indices[id_of_closest], indices[id_of_farthest]
    print(id_of_closest, id_of_farthest)
    print(
        f"For point 0 as anchor, positive sample is {id_of_closest} and negative sample is {id_of_farthest}")


fake_mels = torch.rand(20, 1, 200)
fake_embeddings = torch.rand(20, 2)
test_kd_tree_loss(fake_mels, fake_embeddings, r=0.1)

# %%


def test_kd_tree_loss_2d(fake_mels, fake_embeddings, r=0.1):
    mels = fake_mels  # (B, 1, n_mels)
    embeddings = fake_embeddings  # (B, 2)
    # get the distance matrix of the embeddings
    dist = torch.cdist(embeddings, embeddings)  # (B, B)
    mask = dist < r
    # print(mask)
    # get the indices of the embeddings that are within the radius
    indices = torch.nonzero(mask, as_tuple=True)
    print(indices)
    y, x = indices
    embeddings_within_radius = embeddings[indices]
    mels_within_radius = mels[indices]
    return
    print(embeddings_within_radius.shape)
    print(mels_within_radius.shape)
    mels_within_radius = mels_within_radius.squeeze(1)
    dist_mels = torch.cdist(
        mels_within_radius[0].unsqueeze(0), mels_within_radius)
    print(dist_mels.shape)
    print(dist_mels)
    sort_by_dist = torch.argsort(dist_mels.squeeze(0))
    print(sort_by_dist)
    id_of_closest, id_of_farthest = sort_by_dist[1], sort_by_dist[-1]
    print(id_of_closest, id_of_farthest)
    print(
        f"For point 0 as anchor, positive sample is {id_of_closest} and negative sample is {id_of_farthest}")


fake_mels = torch.rand(10, 1, 200)
fake_embeddings = torch.rand(10, 2)
test_kd_tree_loss_2d(fake_mels, fake_embeddings, r=0.4)
# %%
