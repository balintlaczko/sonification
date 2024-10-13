import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from ..utils.misc import quickSort

# taken from https://github.com/1Konny/FactorVAE/blob/master/ops.py


def recon_loss(x, x_recon):
    n = x.size(0)
    loss = F.binary_cross_entropy_with_logits(
        x_recon, x, reduction="sum").div(n)
    return loss


def kld_loss(mu: Tensor, logvar: Tensor) -> Tensor:
    """
    Compute the Kullback-Leibler divergence loss.
    :param mu: (Tensor) Mean of the latent Gaussian
    :param logvar: (Tensor) Standard deviation of the latent Gaussian
    :return: (Tensor) KLD loss
    """
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kld.mean()


def gap_loss(latent_vectors):
    # normalize the latent vectors between 0 and 1
    latent_vectors_norm = (latent_vectors - latent_vectors.min()) / \
        (latent_vectors.max() - latent_vectors.min())
    num_dims = latent_vectors.shape[1]
    # loop over the dimensions
    loss = 0
    for i in range(num_dims):
        # sort the latent vectors
        sorted_latent, _ = torch.sort(latent_vectors_norm[:, i])
        # sorted_latent = latent_vectors_norm[:, i]
        # quickSort(sorted_latent, 0, len(sorted_latent)-1)
        # calculate the difference between the sorted vectors
        diff = sorted_latent[1:] - sorted_latent[:-1]
        # take the standard deviation of the differences
        loss += diff.std().abs()
    return loss


# based on: https://github.com/ChunjinSong/audioviewer_code
def preserve_distance_loss(source_latent, target_latent, e=1e-8, weight=1):
    batch_size = source_latent.shape[0]
    N = batch_size // 2

    source_z1 = source_latent[:N, :]
    source_z2 = source_latent[N: 2*N, :]

    target_z1 = target_latent[:N, :]
    target_z2 = target_latent[N: 2*N, :]

    dist_source = torch.sqrt(torch.sum((source_z1 - source_z2)
                                       ** 2, -1, keepdim=True)) * weight
    dist_target = torch.sqrt(
        torch.sum((target_z1 - target_z2) ** 2, -1, keepdim=True))

    loss = F.mse_loss(torch.log(torch.abs(dist_target + e)),
                      torch.log(torch.abs(dist_source)))
    loss /= N

    return loss


def preserve_axiswise_distance_loss(source_latent, target_latent, e=1e-8, weight=1):
    batch_size = source_latent.shape[0]
    N = batch_size // 2
    num_dims = source_latent.shape[1]

    source_z1 = source_latent[:N, :]
    source_z2 = source_latent[N: 2*N, :]

    target_z1 = target_latent[:N, :]
    target_z2 = target_latent[N: 2*N, :]

    loss = 0
    for i in range(num_dims):
        dist_source = torch.sqrt(
            (source_z1[:, i] - source_z2[:, i]) ** 2) * weight
        dist_target = torch.sqrt((target_z1[:, i] - target_z2[:, i]) ** 2)

        loss += F.mse_loss(torch.log(torch.abs(dist_target + e)),
                           torch.log(torch.abs(dist_source))) / N

    return loss


def preserve_directions_and_distance_loss(source_latent, target_latent, e=1e-8):
    batch_size = source_latent.shape[0]
    N = batch_size // 2

    # split latents into 2 groups
    # for source latents
    source_z1 = source_latent[:N, :]
    source_z2 = source_latent[N: 2*N, :]
    # for target latents
    target_z1 = target_latent[:N, :]
    target_z2 = target_latent[N: 2*N, :]

    # create vectors between the two groups
    v_source = source_z2 - source_z1
    v_target = target_z2 - target_z1

    # calculate the distance loss
    dist_source = torch.linalg.vector_norm(v_source, dim=1, keepdim=True)
    dist_target = torch.linalg.vector_norm(v_target, dim=1, keepdim=True)
    loss = F.mse_loss(torch.log(dist_source + e),
                      torch.log(dist_target + e)) / N

    # calculate the direction loss
    source_dir = F.normalize(source_z2 - source_z1)
    target_dir = F.normalize(target_z2 - target_z1)
    loss += F.mse_loss(source_dir, target_dir) / N

    return loss


def latent_consistency_loss(encoder, decoder, x, shift_vector, lambda_consistency=1.0, lambda_cross_consistency=1.0):
    """
    Implements the consistency loss where a shift in latent space leads to consistent changes in pixel space
    across the batch.
    
    Args:
        encoder: The encoder network that maps x to latent space.
        decoder: The decoder network that maps latent embeddings back to pixel space.
        x: Batch of input x (batch_size, channels, height, width).
        shift_vector: The constant vector to apply as a shift to all latent embeddings (latent_dim,).
        lambda_consistency: Weight for the proportionality consistency loss term.
        lambda_cross_consistency: Weight for the cross-sample consistency term.
    
    Returns:
        Combined consistency loss.
    """
    # Get the latent embeddings of the original x
    latent_embeddings = encoder(x)  # Shape: (batch_size, latent_dim)
    
    # Apply the same shift vector to all embeddings
    shifted_latent_embeddings = latent_embeddings + shift_vector
    
    # Decode both the original and shifted embeddings
    decoded_x = decoder(latent_embeddings)  # Shape: (batch_size, channels, height, width)
    decoded_shifted_x = decoder(shifted_latent_embeddings)  # Shape: (batch_size, channels, height, width)
    
    # Measure the pixel space difference between decoded x
    delta_pixel = decoded_shifted_x - decoded_x  # The change in pixel space
    
    # Part 1: Proportionality consistency
    latent_norm = torch.norm(shift_vector, p=2)
    consistency_loss = F.mse_loss(delta_pixel, latent_norm * torch.ones_like(delta_pixel))
    
    # Part 2: Cross-sample consistency
    
    # Reshape delta_pixel to (batch_size, -1) to flatten the pixel space dimensions
    delta_pixel_flat = delta_pixel.view(delta_pixel.size(0), -1)  # Shape: (batch_size, num_pixels)
    
    # Normalize the flattened delta_pixel to get unit vectors (directional vectors)
    delta_pixel_flat_norm = F.normalize(delta_pixel_flat, p=2, dim=1)  # Shape: (batch_size, num_pixels)
    
    # Compute pairwise cosine similarity between deltas across the batch
    cosine_similarity_matrix = torch.matmul(delta_pixel_flat_norm, delta_pixel_flat_norm.T)  # Shape: (batch_size, batch_size)
    
    # We want the cosine similarity between all pairs to be close to 1 (indicating same direction)
    # So we compute a loss that penalizes deviation from 1
    cosine_consistency_loss = F.mse_loss(cosine_similarity_matrix, torch.ones_like(cosine_similarity_matrix))
    
    # Exclude self-comparisons by zeroing the diagonal
    cosine_consistency_loss = cosine_consistency_loss - F.mse_loss(torch.diag(cosine_similarity_matrix), torch.ones_like(torch.diag(cosine_similarity_matrix)))
    
    
    # Combine the two losses
    total_loss = lambda_consistency * consistency_loss + lambda_cross_consistency * cosine_consistency_loss
    
    return total_loss


# taken form: https://github.com/AntixK/PyTorch-VAE/blob/master/models/info_vae.py
# then modified it into its own class
class MMDloss(nn.Module):
    def __init__(self,
                 kernel_type: str = 'imq',
                 latent_var: float = 2.,
                 ):
        super(MMDloss, self).__init__()
        self.kernel_type = kernel_type
        self.z_var = latent_var

    def compute_kernel(self,
                       x1: Tensor,
                       x2: Tensor) -> Tensor:
        # Convert the tensors into row and column vectors
        D = x1.size(1)
        N = x1.size(0)

        x1 = x1.unsqueeze(-2)  # Make it into a column tensor
        x2 = x2.unsqueeze(-3)  # Make it into a row tensor

        """
        Usually the below lines are not required, especially in our case,
        but this is useful when x1 and x2 have different sizes
        along the 0th dimension.
        """
        x1 = x1.expand(N, N, D)
        x2 = x2.expand(N, N, D)

        if self.kernel_type == 'rbf':
            result = self.compute_rbf(x1, x2)
        elif self.kernel_type == 'imq':
            result = self.compute_inv_mult_quad(x1, x2)
        else:
            raise ValueError('Undefined kernel type.')

        return result

    def compute_rbf(self,
                    x1: Tensor,
                    x2: Tensor,
                    eps: float = 1e-7) -> Tensor:
        """
        Computes the RBF Kernel between x1 and x2.
        :param x1: (Tensor)
        :param x2: (Tensor)
        :param eps: (Float)
        :return:
        """
        z_dim = x2.size(-1)
        sigma = 2. * z_dim * self.z_var

        result = torch.exp(-((x1 - x2).pow(2).mean(-1) / sigma))
        return result

    def compute_inv_mult_quad(self,
                              x1: Tensor,
                              x2: Tensor,
                              eps: float = 1e-7) -> Tensor:
        """
        Computes the Inverse Multi-Quadratics Kernel between x1 and x2,
        given by

                k(x_1, x_2) = \sum \frac{C}{C + \|x_1 - x_2 \|^2}
        :param x1: (Tensor)
        :param x2: (Tensor)
        :param eps: (Float)
        :return:
        """
        z_dim = x2.size(-1)
        C = 2 * z_dim * self.z_var
        kernel = C / (eps + C + (x1 - x2).pow(2).sum(dim=-1))

        # Exclude diagonal elements
        result = kernel.sum() - kernel.diag().sum()

        return result

    def compute_mmd(self, z: Tensor, prior_distribution: str = "gaussian", custom_prior=None) -> Tensor:
        assert prior_distribution in [
            "gaussian", "uniform", "custom"], "prior distribution must be either 'gaussian' or 'uniform'"
        if prior_distribution == "gaussian":
            # Sample from prior (Gaussian) distribution
            prior_z = torch.randn_like(z)
        elif prior_distribution == "uniform":
            # Sample from prior (Uniform) distribution
            prior_z = torch.rand_like(z)
        elif prior_distribution == "custom":
            # Sample from custom prior distribution
            prior_z = custom_prior

        prior_z__kernel = self.compute_kernel(prior_z, prior_z)
        z__kernel = self.compute_kernel(z, z)
        priorz_z__kernel = self.compute_kernel(prior_z, z)

        mmd = prior_z__kernel.mean() + \
            z__kernel.mean() - \
            2 * priorz_z__kernel.mean()
        return mmd
