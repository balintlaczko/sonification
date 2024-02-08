import numpy as np
from ..utils import array

def test_resize_interp():
    # test results length, and matching first and last elements, and average distance between elements
    for i in range(5):
        in_array_length = np.random.randint(10, 1000)
        out_array_length = np.random.randint(10, 1000)
        in_array = np.arange(in_array_length)
        out_array = array.resize_interp(in_array, out_array_length)
        # test that length is correct
        assert len(out_array) == out_array_length
        # test that first and last elements match
        assert np.allclose(out_array[0], in_array[0])
        assert np.allclose(out_array[-1], in_array[-1])
        # test that elements are evenly spaced
        assert np.std(np.diff(out_array)) < 1e-6

test_resize_interp()


def test_scale_array_exp():
    in_array = np.arange(0, 1.1, 0.1) # include the last element
    # test if no scaling creates no change
    out_array = array.scale_array_exp(in_array, 0, 1, 0, 1)
    assert np.allclose(in_array, out_array)
    # when only changing exponent, test that the first and last elements match
    out_array = array.scale_array_exp(in_array, 0, 1, 0, 1, 2)
    assert np.allclose(in_array[0], out_array[0])
    assert np.allclose(in_array[-1], out_array[-1])
    out_array = array.scale_array_exp(in_array, 0, 1, 0, 1, 0.5)
    assert np.allclose(in_array[0], out_array[0])
    assert np.allclose(in_array[-1], out_array[-1])

test_scale_array_exp()