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

# by chatgpt
def kld_loss_uniform(mu: Tensor, logvar: Tensor, a: float=-4, b: float=4) -> Tensor:
    """
    Compute the Kullback-Leibler divergence loss between a Gaussian distribution 
    with mean `mu` and log-variance `logvar`, and a uniform distribution over [a, b].
    
    :param mu: (Tensor) Mean of the latent Gaussian
    :param logvar: (Tensor) Log-variance of the latent Gaussian
    :param a: (float) Lower bound of the uniform distribution
    :param b: (float) Upper bound of the uniform distribution
    :return: (Tensor) KLD loss
    """
    # Convert logvar to variance
    var = logvar.exp()
    
    # Uniform distribution log-probability over [a, b]
    uniform_log_prob = -torch.log(torch.tensor(b - a))

    # Gaussian log-probability term for KL divergence
    # D_KL(P || U) = 0.5 * (log(var) + (mu^2 + var) - 1) - log_prob(U)
    kld = 0.5 * (logvar + (mu.pow(2) + var) - 1) - uniform_log_prob

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
