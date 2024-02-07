import torch
from utils import tensor
from models import ddsp
import utils
import numpy as np


def test_wrap():
    # test for parity with the numpy version
    for i in range(100):
        n_batches = 128
        n_samps = 100
        input_np = np.random.uniform(-2.0, 2.0, size=(n_batches, n_samps))
        input_t = torch.tensor(input_np)
        wrap_np = utils.wrap(input_np, -1.0, 1.0)
        wrap_t = tensor.wrap(input_t, -1.0, 1.0)
        assert np.allclose(wrap_np, wrap_t.numpy())


test_wrap()


def test_phasor():
    # test for parity with the numpy version
    for i in range(100):
        n_batches = 128
        n_samps = 100
        input_freq = np.random.uniform(0.1, 200.0, size=(n_batches, n_samps))
        input_freq_t = torch.tensor(input_freq)
        phasor_np = np.zeros((n_batches, n_samps))
        for j in range(n_batches):
            phasor_np[j, :] = utils.phasor(n_samps, n_samps, input_freq[j, :])
        phasor_t = ddsp.Phasor(n_samps)(input_freq_t)
        assert np.allclose(phasor_np, phasor_t.numpy())


test_phasor()


def test_sine():
    # test for parity with the numpy version
    for i in range(100):
        n_batches = 128
        n_samps = 100
        input_freq = np.random.uniform(0.1, 200.0, size=(n_batches, n_samps))
        input_freq_t = torch.tensor(input_freq)
        sine_np = np.zeros((n_batches, n_samps))
        for j in range(n_batches):
            sine_np[j, :] = utils.sinewave(n_samps, n_samps, input_freq[j, :])
        sine_t = ddsp.Sinewave(n_samps)(input_freq_t)
        assert np.allclose(sine_np, sine_t.numpy())


test_sine()


def test_fm_synth():
    # test for parity with the numpy version (fm_synth_2, because fm_synth is not proportional)
    for i in range(100):
        n_batches = 128
        n_samps = 100
        carr_freq = np.random.uniform(0.1, 200.0, size=(n_batches, n_samps))
        harm_ratio = np.random.uniform(0.1, 10.0, size=(n_batches, n_samps))
        mod_index = np.random.uniform(0.1, 10.0, size=(n_batches, n_samps))
        carr_freq_t = torch.tensor(carr_freq)
        harm_ratio_t = torch.tensor(harm_ratio)
        mod_index_t = torch.tensor(mod_index)
        fm_np = np.zeros((n_batches, n_samps))
        for j in range(n_batches):
            fm_np[j, :] = utils.fm_synth_2(
                n_samps, n_samps, carr_freq[j, :], harm_ratio[j, :], mod_index[j, :])
        fm_t = ddsp.FMSynth(n_samps)(carr_freq_t, harm_ratio_t, mod_index_t)
        assert np.allclose(fm_np, fm_t.numpy())


test_fm_synth()
