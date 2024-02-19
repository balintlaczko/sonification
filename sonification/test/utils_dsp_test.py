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


def test_switch():
    for i in range(5):
        in_array_length = np.random.randint(10, 1000)
        switch_sig = np.random.rand(in_array_length)
        switch_sig = np.where(switch_sig > 0.5, 1, 0)
        true_sig = np.random.rand(in_array_length)
        false_sig = np.random.rand(in_array_length)
        out_switch = dsp.switch(switch_sig, true_sig, false_sig)
        out_where = np.where(switch_sig, true_sig, false_sig)
        # test that length is correct
        assert len(out_switch) == in_array_length
        # test that the switch is correct
        assert np.allclose(out_switch, out_where)


def test_accum():
    reset_signal = np.array([0, 1, 0, 0, 1, 1, 1, 0, 0, 0])
    in_array_length = len(reset_signal)
    accum_signal = np.ones(in_array_length)
    out_accum_post = dsp.accum(accum_signal, reset_signal, "post")
    out_accum_pre = dsp.accum(accum_signal, reset_signal, "pre")
    solution_post = np.array([1, 2, 1, 2, 3, 1, 1, 1, 2, 3])
    solution_pre = np.array([1, 1, 2, 3, 1, 1, 1, 2, 3, 4])
    # test that length is correct
    assert len(out_accum_post) == in_array_length
    assert len(out_accum_pre) == in_array_length
    # test that the accum is correct
    assert np.allclose(out_accum_post, solution_post)
    assert np.allclose(out_accum_pre, solution_pre)


def test_mix():
    # test that it works with a single mix value
    in_array_length = 100
    in_a = np.zeros(in_array_length)
    in_b = np.ones(in_array_length)
    mix_value = 0.5
    out_mix = dsp.mix(in_a, in_b, mix_value)
    # test that length is correct
    assert len(out_mix) == in_array_length
    # test that the mix is correct
    assert np.allclose(out_mix, in_b * mix_value)
    # test that it works with a mix array
    mix_signal = np.random.rand(in_array_length)
    out_mix = dsp.mix(in_a, in_b, mix_signal)
    # test that length is correct
    assert len(out_mix) == in_array_length
    # test that the mix is correct
    assert np.allclose(out_mix, in_b * mix_signal)


def test_ramp2trigger():
    # test that the trigger is correct
    sr_list = [48000, 44100, 96000]
    for sr in sr_list:
        for i in range(5):
            random_length_s = np.random.randint(1, 10)
            random_freq_hz = np.random.randint(1, 10)
            test_ramp = dsp.phasor(sr * random_length_s,
                                   sr, np.array([random_freq_hz]))
            test_trigger = dsp.ramp2trigger(test_ramp)
            # test that length is correct
            assert len(test_trigger) == random_length_s * sr
            # test that the trigger is correct
            assert np.sum(test_trigger) == random_length_s * random_freq_hz


###### RUN TESTS ######
test_history()
test_switch()
test_accum()
test_mix()
test_ramp2trigger()
