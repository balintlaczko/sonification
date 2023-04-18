# %%
# imports
import torch
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.losses import SelfSupervisedLoss, TripletMarginLoss
from flucoma.utils import cleanup, get_buffer
from flucoma import fluid
from pathlib import Path
import numpy as np
from scipy.io import wavfile as wav
from utils import *

# %%
# TODO:
# - integrate RGB processing into existing pipeline, be able to generate videos en mass


# %%

# test resize_interp


a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(a)
print(np.round(resize_interp(a, 6)).astype(int))
a = np.arange(10)
print(a)
print(np.round(resize_interp(a, 6)).astype(int))
a = np.arange(15)
print(a)
print(resize_interp(a, 20))


# %%

# test phasor


sr = 48000
frequency = np.array([1, 10, 2])
samples = 48000 * 10
bufphasor = phasor(samples, sr, frequency)
target_name = "/Users/balintl/Desktop/test_phasor.wav"
wav.write(target_name, sr, bufphasor.astype(np.float32))


# %%

# test sinewave


sr = 48000
frequency = np.array([1, 5, 1])
samples = 48000 * 20
testy = sinewave(samples, sr, frequency)
target_name = "/Users/balintl/Desktop/test_sine.wav"
wav.write(target_name, sr, testy.astype(np.float32))

# %%

# time phasor vs sinewave vs generate_sine

%timeit phasor(48000 * 60, 48000, np.array([1]))

# %%
%timeit sinewave(48000 * 60, 48000, np.array([1]))

# %%
%timeit generate_sine(48000, 60, 1, 0, 4096)

# %%

# test fm_synth

sr = 48000
carrfreq = np.array([440])
modfreq = np.array([50])
modamp = np.array([100])
samples = sr * 10
testy = fm_synth(samples, sr, carrfreq, modfreq, modamp)
target_name = "/Users/balintl/Desktop/test_fm.wav"
wav.write(target_name, sr, testy.astype(np.float32))

# %%

# time fm_synth

%timeit fm_synth(48000 * 60, 48000, np.array([440]), np.array([1]), np.array([10]))

# %%

# test array2broadcastable

a = np.arange(10, dtype=np.float64)
print(a)
print(array2broadcastable(a, 10))
b = np.array([42], dtype=np.float64)
print(b)
print(array2broadcastable(b, 10))


# %%

# test fm_synth_2

sr = 48000
carrfreq = np.array([440])
harmratio = np.array([42.42])
modindex = np.array([10])
samples = sr * 10
testy = fm_synth_2(samples, sr, carrfreq, harmratio, modindex)
target_name = "/Users/balintl/Desktop/test_fm_2.wav"
wav.write(target_name, sr, testy.astype(np.float32))

# %%

# time fm_synth_2

%timeit fm_synth_2(48000 * 60, 48000, np.array([440]), np.array([1]), np.array([10]))


# %%

# test am_synth

sr = 48000
carrfreq = np.array([440])
modfreq = np.array([50])
modamp = np.array([0.5])
samples = sr * 10
testy = am_synth(samples, sr, carrfreq, modfreq, modamp)
target_name = "/Users/balintl/Desktop/test_am.wav"
wav.write(target_name, sr, testy.astype(np.float32))

# %%

# recreate assignment41 patch as a test


sr = 48000
samples = sr * 60

carr_sine = sinewave(samples, sr, np.array([440]))
carr_sine *= am_module(samples, sr, np.array([100]), np.array([1]))
carr_sine *= am_module(samples, sr, np.array([1]), np.array([0.3]))
carr_sine *= am_module(samples, sr, np.array([8]), np.array([0.2]))

target_name = "/Users/balintl/Desktop/test_am_assignment41.wav"
wav.write(target_name, sr, carr_sine.astype(np.float32))

# %%

# recreate assignment42 patch as a test


sr = 48000
samples = sr * 60

carrier = fm_synth(samples, sr, np.array(
    [300]), np.array([400]), np.array([1300]))
carrier *= am_module(samples, sr, np.array([250]), np.array([0.5]))
carrier *= am_module(samples, sr, np.array([1]), np.array([0.3]))
carrier *= am_module(samples, sr, np.array([8]), np.array([0.15]))

target_name = "/Users/balintl/Desktop/test_fm_assignment42.wav"
wav.write(target_name, sr, carrier.astype(np.float32))

# %%

# test history function

test_signal = np.array([1, 2, 3, 4, 5])
test_history = history(test_signal)
assert np.array_equal(test_history, np.array([0, 1, 2, 3, 4]))

# %%

# test ramp2trigger function

sr = 48000
test_ramp = phasor(sr*10, sr, np.array([2]))
test_trigger = ramp2trigger(test_ramp)
assert np.sum(test_trigger) == 20

# %%

# test ramp2slope

ramp = phasor(48000 * 10, 48000, np.array([1]))
slope = ramp2slope(ramp)
slope

# %%

# test scale_array_exp

test_array = np.arange(0, 1.1, 0.1)
print(test_array)
print(scale_array_exp(test_array, 0, 1, 0, 1))
print(scale_array_exp(test_array, 0, 1, 0, 1, 2))
print(scale_array_exp(test_array, 0, 1, 0, 1, 0.5))


# %%
# flucoma test stuff
# You will need to replace the path here with your own sound file
source = Path("/Users/balintl/Desktop/test_fm_assignment42.wav")

# Take the MFCCs of the source
mfcc = fluid.mfcc(source, numcoeffs=5, numbands=20)

# fluid.processes return a python dataclass that contains two things:
# 1. The file_path which is a string pointing to the file which exists on disk.
# 2. The data stored in that file.
# If you do not pass a specific path for an output it will put the output in a temporary file
# The default location for all output files is temporary location is ~/.python-flucoma

# From the dataclass we can extract

# Wrap a fluid.process() in a get_buffer to get a fairly native output
stats = fluid.stats(mfcc, numderivs=0)

# Let's see the data
for i, band in enumerate(stats):
    print(type(band))
    printout = f"Stats for MFCC band number {i}: {band} \n"
    print(printout)
print(
    "You should see 10 values for each band's statistics, because we have 5 coefficients and the input is stereo!"
)


# We didn't set any specific output so we can cleanup the temporary file if we want
# cleanup()

# %%
get_buffer(fluid.stats(fluid.mfcc(source, numcoeffs=5,
           numbands=20), numderivs=1), output="numpy").shape

# %%
print(get_buffer(fluid.stats(fluid.spectralshape(
    source), numderivs=0), output="numpy").shape)
cleanup()

# %%
# pytorch-metric-learning test stuff

# %%
loss_func = SelfSupervisedLoss(TripletMarginLoss())

# %%
a = np.random.rand(10, 1)
b = np.random.rand(10, 1)
a_t = torch.from_numpy(a)
b_t = torch.from_numpy(b)
cos_sim = CosineSimilarity()
distances = cos_sim.pairwise_distance(a_t, b_t)
distances, distances.shape


# %%

# your training for-loop
# for i, data in enumerate(dataloader):
#     optimizer.zero_grad()
#     embeddings = your_model(data)
#     augmented = your_model(your_augmentation(data))
#     loss = loss_func(embeddings, augmented)
#     loss.backward()
#     optimizer.step()
