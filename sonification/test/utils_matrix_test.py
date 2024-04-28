from sonification.utils import matrix
import numpy as np


def test_matrix_crop(test_shape: tuple, test_crop_left: int, test_crop_right: int, test_crop_top: int, test_crop_bottom: int) -> None:
    test_input = np.zeros(test_shape)
    test_result = matrix.matrix_crop(
        test_input, test_crop_left, test_crop_right, test_crop_top, test_crop_bottom)
    assert test_result.shape[0] == test_shape[0] - \
        test_crop_top - test_crop_bottom
    assert test_result.shape[1] == test_shape[1] - \
        test_crop_left - test_crop_right
    assert len(test_shape) == len(test_result.shape)


test_matrix_crop((100, 200, 3), 10, 20, 30, 40)
test_matrix_crop((100, 200, 3), 1, 2, 3, 4)
test_matrix_crop((100, 200), 1, 2, 3, 4)
test_matrix_crop((100, 200, 3, 4), 1, 2, 3, 4)


def test_matrix_crop_at(test_shape: tuple, test_crop_left: int, test_crop_right: int, test_crop_top: int, test_crop_bottom: int) -> None:
    test_input = np.zeros(test_shape)
    test_result = matrix.matrix_crop_at(
        test_input, test_crop_left, test_crop_right, test_crop_top, test_crop_bottom)
    assert test_result.shape[0] == test_crop_bottom - test_crop_top
    assert test_result.shape[1] == test_crop_right - test_crop_left
    assert len(test_shape) == len(test_result.shape)


test_matrix_crop_at((100, 200, 3), 10, 20, 30, 40)
test_matrix_crop_at((100, 200, 3), 1, 2, 3, 4)
test_matrix_crop_at((100, 200), 1, 2, 3, 4)
test_matrix_crop_at((100, 200, 3, 4), 1, 2, 3, 4)


def test_matrix_crop_around(test_shape: tuple, test_crop: int) -> None:
    test_input = np.zeros(test_shape)
    test_result = matrix.matrix_crop_around(test_input, test_crop)
    assert test_result.shape[0] == test_shape[0] - (2*test_crop)
    assert test_result.shape[1] == test_shape[1] - (2*test_crop)
    assert len(test_shape) == len(test_result.shape)


test_matrix_crop_around((100, 200, 3), 20)
test_matrix_crop_around((100, 200, 3), 5)
test_matrix_crop_around((100, 200), 10)
test_matrix_crop_around((100, 200, 3, 4), 10)


def test_matrix_crop_empty(test_width: int, test_height: int, side_pad_min: int, side_pad_max: int) -> None:
    test_shape = (test_height, test_width)
    # random_paddig = np.random.randint(side_pad_min, side_pad_max+1, size=4)
    padding_left, padding_right, padding_top, padding_bottom = np.random.randint(
        side_pad_min, side_pad_max+1, size=4)
    # print(padding_left, padding_right, padding_top, padding_bottom)
    padded_width = test_width + padding_left + padding_right
    padded_height = test_height + padding_top + padding_bottom
    # print(padded_height, padded_width)
    padded_matrix = np.zeros((padded_height, padded_width))
    test_content = np.ones(test_shape)
    padded_matrix[padding_top:test_height+padding_top,
                  padding_left:test_width+padding_left] = test_content
    assert np.array_equal(
        test_content, matrix.matrix_crop_empty(padded_matrix))


test_matrix_crop_empty(400, 400, 1, 50)
test_matrix_crop_empty(400, 400, 0, 0)
test_matrix_crop_empty(100, 20, 0, 1)
test_matrix_crop_empty(100, 20, 1, 2)
