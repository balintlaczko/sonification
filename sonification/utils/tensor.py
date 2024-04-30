import torch


def fmod(
        x: torch.Tensor,
        y: torch.Tensor,
) -> torch.Tensor:
    return x - x.div(y, rounding_mode="floor") * y


def wrap(
        x: torch.Tensor,
        min: float,
        max: float,
) -> torch.Tensor:
    """
    Wrap a tensor to the range [min, max].

    Args:
        x: tensor to wrap
        min: minimum value
        max: maximum value

    Returns:
        wrapped tensor
    """
    return min + fmod(x - min, max - min)


def scale(x, in_low, in_high, out_low, out_high, exp=1):
    if in_low == in_high:
        return torch.ones_like(x) * out_high
    return torch.where(
        (x-in_low)/(in_high-in_low) == 0,
        out_low,
        torch.where(
            (x-in_low)/(in_high-in_low) > 0,
            out_low + (out_high-out_low) *
            ((x-in_low)/(in_high-in_low))**exp,
            out_low + (out_high-out_low) * -
            ((((-x+in_low)/(in_high-in_low)))**(exp))
        )
    )


def permute_dims(x: torch.Tensor) -> torch.Tensor:
    # x is (batch, z)
    b, z_dim = x.shape
    y = torch.zeros_like(x)
    # permutate dimensions independently
    for i in range(z_dim):
        y[:, i] = x[:, i][torch.randperm(b)]
    return y


def db2amp(db: torch.Tensor) -> torch.Tensor:
    return 10**(db / 20)


def amp2db(amp: torch.Tensor) -> torch.Tensor:
    return 20 * torch.log10(amp)


def frequency2midi(
        frequency: torch.Tensor,
        base_frequency: float = 440.0,
) -> torch.Tensor:
    """
    Converts a frequency in Hz to a MIDI note number.

    Args:
        frequency: Frequency in Hz. Can be a scalar or an array as a torch.tensor.
        base_frequency: Frequency of MIDI note 69. Defaults to 440.0.

    Returns:
        np.ndarray: MIDI note number.
    """

    return 69 + 12 * torch.log2(frequency / base_frequency)
