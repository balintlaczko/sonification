import torch
from sonification.models.ddsp import FMSynth

# Create inputs
carrier_frequency = torch.rand(1, 48000, requires_grad=True)
harmonicity_ratio = torch.rand(1, 48000, requires_grad=True)
modulation_index = torch.rand(1, 48000, requires_grad=True)

# Instantiate synth
synth = FMSynth(sr=48000)

# Forward pass
output = synth(carrier_frequency, harmonicity_ratio, modulation_index)

# Check if gradients flow
loss = output.mean()
loss.backward()

# If no errors and all gradients are computed, it's differentiable
print(carrier_frequency.grad is not None)  # Should be True