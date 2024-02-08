import numpy as np
from sonification.utils import dsp

def test_history():
    # test that the shift is correct
    for i in range(5):
        in_array_length = np.random.randint(10, 1000)
        random_div = np.random.randint(1, 10)
        in_array = np.arange(in_array_length) / random_div
        out_array = dsp.history(in_array)
        # test that length is correct
        assert len(out_array) == in_array_length
        # test that the shift is correct
        assert np.allclose(out_array[1:], in_array[:-1])

test_history()


def test_ramp2trigger():
    # test that the trigger is correct
    sr_list = [48000, 44100, 96000]
    for sr in sr_list:
        for i in range(5):
            random_length_s = np.random.randint(1, 10)
            random_freq_hz = np.random.randint(1, 10)
            test_ramp = dsp.phasor(sr * random_length_s, sr, np.array([random_freq_hz]))
            test_trigger = dsp.ramp2trigger(test_ramp)
            # test that length is correct
            assert len(test_trigger) == random_length_s * sr
            # test that the trigger is correct
            assert np.sum(test_trigger) == random_length_s * random_freq_hz

test_ramp2trigger()


