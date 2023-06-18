import torch
from torch import nn


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


class Phasor(nn.Module):
    def __init__(
        self,
        sr: int
    ):
        super().__init__()

        self.sr = sr

    def forward(self, freq: torch.Tensor):  # input shape: (batch_size, n_samples)
        batch_size = freq.shape[0]
        increment = freq[:, :-1] / self.sr
        phase = torch.cumsum(increment, dim=-1)
        phase = torch.cat([torch.zeros(batch_size, 1).to(phase.device), phase], dim=-1)
        phasor = wrap(phase, 0.0, 1.0)
        return phasor


class Sinewave(nn.Module):
    def __init__(
        self,
        sr: int
    ):
        super().__init__()

        self.sr = sr
        self.phasor = Phasor(self.sr)

    def forward(self, freq: torch.Tensor):
        phasor = self.phasor(freq)
        sine = torch.sin(2 * torch.pi * phasor)
        return sine


class FMSynth(nn.Module):
    def __init__(
        self,
        sr: int,
    ):
        super().__init__()

        self.sr = sr
        self.modulator_sine = Sinewave(self.sr)
        self.carrier_sine = Sinewave(self.sr)

    def forward(
            self,
            carrier_frequency: torch.Tensor,
            harmonicity_ratio: torch.Tensor,
            modulation_index: torch.Tensor,
    ):
        modulator_frequency = carrier_frequency * harmonicity_ratio
        modulator_buf = self.modulator_sine(modulator_frequency)
        modulation_amplitude = modulator_frequency * modulation_index
        return self.carrier_sine(carrier_frequency + modulator_buf * modulation_amplitude)
