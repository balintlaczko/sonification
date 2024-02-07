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


# function to scale with exponent
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