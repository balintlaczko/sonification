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
from sonification.utils.array import array2broadcastable
from sonification.utils.dsp import phasor, sinewave, fm_synth, fm_synth_2, am_synth, am_module, ramp2slope

# %%
# TODO:
# - integrate RGB processing into existing pipeline, be able to generate videos en mass


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

# %timeit phasor(48000 * 60, 48000, np.array([1]))

# %%
# %timeit sinewave(48000 * 60, 48000, np.array([1]))

# %%
# %timeit generate_sine(48000, 60, 1, 0, 4096)

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

# %timeit fm_synth(48000 * 60, 48000, np.array([440]), np.array([1]), np.array([10]))

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

# %timeit fm_synth_2(48000 * 60, 48000, np.array([440]), np.array([1]), np.array([10]))


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

# test ramp2slope

ramp = phasor(48000 * 10, 48000, np.array([1]))
slope = ramp2slope(ramp)
slope


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


# %%
from sonification.utils.video import videos2planes

video_gfp = "/Users/balintl/Desktop/Sonification Pilot Video/Materials/processing/pbsgb_gfp_st_1.avi"
video_rfp = "/Users/balintl/Desktop/Sonification Pilot Video/Materials/processing/pbsgb_rfp_st_1.avi"
target_name = "/Users/balintl/Desktop/Sonification Pilot Video/Materials/processing/rfp_red.avi"

# %%
target_name = "/Users/balintl/Desktop/Sonification Pilot Video/Materials/processing/rfp_red.avi"
videos2planes(video2red=video_rfp, target_name=target_name)


# %%
import cv2
import numpy as np
from sonification.utils.matrix import view

# %%
mask_path = "/Volumes/T7RITMO/Sonification/Amani_230117/W3onlywith masks/Timepoint_001_230117-BST_C03_s1_w3_cp_masks.png"
image_w1_path = "/Volumes/T7RITMO/Sonification/Amani_230117/W1W2/Timepoint_001_230117-BST_C03_s1_w1.TIF"
image_w2_path = "/Volumes/T7RITMO/Sonification/Amani_230117/W1W2/Timepoint_001_230117-BST_C03_s1_w2.TIF"
mask = cv2.imread(mask_path, cv2.IMREAD_ANYDEPTH)
w1 = cv2.imread(image_w1_path, cv2.IMREAD_ANYDEPTH)
w2 = cv2.imread(image_w2_path, cv2.IMREAD_ANYDEPTH)
np.unique(mask)

# %%
label_1 = np.where(mask == 1, 1, 0)
view(label_1.astype(np.uint8) * 255, 0.3)

# %%
# view normalized rgb
all_rgb = np.dstack((w1, w2, np.zeros_like(w1)))
all_rgb_8bit = (all_rgb / np.max(all_rgb) * 255).astype(np.uint8)
view(all_rgb_8bit, 0.3)

# %%
# mask original imige at a given label
label = 42
w1_masked = np.where(mask == label, w1, 0)
w2_masked = np.where(mask == label, w2, 0)
# make rgb image
rgb = np.dstack((w1_masked, w2_masked, np.zeros_like(w1_masked)))
rgb_8bit = (rgb / np.max(rgb) * 255).astype(np.uint8)
view(rgb_8bit, 0.3)

# %%

