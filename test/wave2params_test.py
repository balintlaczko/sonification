# %%
# imports
import torch
from models.ddsp import FMSynth, Wave2Params, MelbandsEncoder, MFCCEncoder, PitchEncoder


# %%
# create synth & models

synth = FMSynth(sr=48000)
model = Wave2Params()
melbands_encoder = MelbandsEncoder()
mfcc_encoder = MFCCEncoder()
pitch_encoder = PitchEncoder(num_frames=376)

# %%
# create input wave
buffer_length = 48000 * 4
carrier_frequency = torch.tensor([440.0]).repeat(buffer_length).unsqueeze(0)
harmonicity_ratio = torch.tensor([4.0]).repeat(buffer_length).unsqueeze(0)
modulation_index = torch.tensor([2.0]).repeat(buffer_length).unsqueeze(0)
y = synth(carrier_frequency, harmonicity_ratio, modulation_index)
y.shape


# %%
# function to save tensor to wav file

def save_buffer_to_wav(buffer: torch.Tensor, filename: str, sr: int = 48000):
    import soundfile as sf
    assert len(buffer.shape) == 1
    buffer_np = buffer.detach().cpu().numpy()
    sf.write(filename, buffer_np, sr)


# %%

# save y to wav file
save_buffer_to_wav(y, '/Users/balintl/Desktop/test_my_fmsynth.wav')


# %%
# create batches for testing

y_batch = y.repeat(2, 1)
y_batch.shape

# %%
# model forward pass

y_pred, params_pred = model(y_batch)
y_pred.shape, params_pred.shape


# %%

melbands = melbands_encoder(y_batch)
melbands.shape

# %%
melbands = MelbandsEncoder(use_gru=False)(y_batch)
melbands.shape

# %%

mfcc = mfcc_encoder(y_batch)
mfcc.shape

# %%

mfcc = MFCCEncoder(use_gru=False)(y_batch)
mfcc.shape

# %%

pitch = pitch_encoder(y_batch)
pitch.shape

# %%
