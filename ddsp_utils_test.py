from ddsp_utils import *
import utils
import numpy as np


def test_wrap():
    # test for parity with the numpy version
    for i in range(100):
        input_np = np.random.uniform(-2.0, 2.0, size=(100,))
        input_t = torch.tensor(input_np)
        wrap_np = utils.wrap(input_np, -1.0, 1.0)
        wrap_t = wrap(input_t, -1.0, 1.0)
        assert np.allclose(wrap_np, wrap_t.numpy())


test_wrap()


def test_phasor():
    # test for parity with the numpy version
    for i in range(100):
        n_samps = 100
        input_freq = np.random.uniform(0.1, 200.0, size=(n_samps,))
        input_freq_t = torch.tensor(input_freq)
        phasor_np = utils.phasor(n_samps, n_samps, input_freq)
        phasor_t = Phasor(n_samps)(input_freq_t)
        assert np.allclose(phasor_np, phasor_t.numpy())


test_phasor()


def test_sine():
    # test for parity with the numpy version
    for i in range(100):
        n_samps = 100
        input_freq = np.random.uniform(0.1, 200.0, size=(n_samps,))
        input_freq_t = torch.tensor(input_freq)
        sine_np = utils.sinewave(n_samps, n_samps, input_freq)
        sine_t = Sinewave(n_samps)(input_freq_t)
        assert np.allclose(sine_np, sine_t.numpy())


test_sine()


def test_fm_synth():
    # test for parity with the numpy version (fm_synth_2, because fm_synth is not proportional)
    for i in range(100):
        n_samps = 100
        carr_freq = np.random.uniform(0.1, 200.0, size=(n_samps,))
        harm_ratio = np.random.uniform(0.1, 10.0, size=(n_samps,))
        mod_index = np.random.uniform(0.1, 10.0, size=(n_samps,))
        carr_freq_t = torch.tensor(carr_freq)
        harm_ratio_t = torch.tensor(harm_ratio)
        mod_index_t = torch.tensor(mod_index)
        fm_np = utils.fm_synth_2(
            n_samps, n_samps, carr_freq, harm_ratio, mod_index)
        fm_t = FMSynth(n_samps)(carr_freq_t, harm_ratio_t, mod_index_t)
        assert np.allclose(fm_np, fm_t.numpy())


test_fm_synth()
