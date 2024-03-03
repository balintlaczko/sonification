import numpy as np
import torch
import cv2
import platform
from .tensor import scale


def view(matrix: np.ndarray, scale: float = 1.0, text: str = None, swap_rb: bool = True) -> None:
    """
    Quickly view a matrix.

    Args:
        matrix (np.ndarray): The matrix to view.
        scale (float, optional): Scale the matrix (image) axes. 0.5 corresponds to halving the width and height, 2 corresponds to a 2x zoom. Defaults to 1.
        text (str, optional): The text to display at the bottom left corner as an overlay.
        swap_rb (bool, optional): Whether to swap red and blue channels before displaying the matrix. Defaults to True.
    """
    # opencv needs bgr order, so swap it if input matrix is colored
    to_show = matrix.copy()
    if swap_rb and len(to_show.shape) > 2:
        to_show[:, :, [0, -1]] = to_show[:, :, [-1, 0]]
    if scale != 1:
        h, w = to_show.shape[:2]
        h_scaled, w_scaled = [int(np.ceil(ax * scale)) for ax in [h, w]]
        to_show = cv2.resize(to_show, (h_scaled, w_scaled))
    if text != None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            to_show, text, (12, to_show.shape[1] - 12), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    # avoid hanging windows on Mac OS
    if platform.system() == "Darwin":
        cv2.startWindowThread()
    cv2.imshow("view", to_show.astype(np.uint8))
    # avoid unmovable windows on Mac OS
    if platform.system() == "Darwin":
        cv2.moveWindow("view", 50, 50)
    cv2.waitKey(0)
    # avoid hanging windows on Mac OS
    if platform.system() == "Darwin":
        cv2.destroyAllWindows()
        cv2.waitKey(1)


def stretch_contrast(
    matrix: np.ndarray,
    in_min: int = None,
    in_max: int = None,
    in_percentile: float = None,
    out_min: int = None,
    out_max: int = None,
    clip: bool = True,
) -> np.ndarray:
    """
    Implements the stretch contrast algorithm, a wide-spread method of normalization. The input and output ranges
    of the mapping can optionally be specified using `in_min` and `in_max` for the input and `out_min` and `out_max`
    for the output. If not specified, the inputs will map to the detected minimum and maximum cell values, while the
    output will map from 0 to the maximum of what the bit-rate allows: 2^bitdepth-1, which is 255 in case of 8-bit
    images, and 65535 for 16-bit images. Optionally clipping can be used for the mapping, which clips the input
    matrix between `in_min` and `in_max` before scaling. If `in_percentile` is defined, `in_max` will become the
    cell value from the matrix distribution at the percentile `in_percentile` (refer to np.percentile for details).

    Args:
        matrix (np.ndarray): The input matrix to be normalized.
        in_min (int, optional): The minimum for defining the input range. If not specified, it will take the minimum detected cell value in the input matrix. Defaults to None.
        in_max (int, optional): The maximum for defining the input range. If not specified, it will take the maximum detected cell value in the input matrix. Defaults to None.
        in_percentile (float, optional): If specified, `in_max` will become the cell value from the matrix distribution at the percentile `in_percentile` (refer to np.percentile for details). Defaults to None.
        out_min (int, optional): The minimum for defining the output range. If not specified, it will be 0. Defaults to None.
        out_max (int, optional): The minimum for defining the output range. If not specified, it will be the maximum cell value of the detected bitdepth (2^bitdepth-1). Defaults to None.
        clip (bool, optional): Whether to clip the input matrix between `in_min` and `in_max` before scaling. Defaults to True.

    Returns:
        np.ndarray: The normalized matrix.
    """
    # if in_min is not defined, take the minimum cell value from the input matrix
    if in_min == None:
        in_min = np.min(matrix)
    # if in_max is not defined, take the maximum cell value from the input matrix
    if in_max == None:
        in_max = np.max(matrix)
    # if in_percentile is specified, take the cell value from the distribution according to it
    if in_percentile != None:
        in_max = np.percentile(matrix, in_percentile)
    # if out_min is not specified let it be 0
    if out_min == None:
        out_min = 0
    # if out_max is not specified, let it be the maximum of the detected bitdepth
    if out_max == None:
        bitdepth = 0
        if matrix.dtype == np.uint16:
            bitdepth = 16
        else:
            bitdepth = 8
        out_max = (2 ^ bitdepth) - 1
    # if clip is True, clip the unput matrix between in_min and in_max before scaling
    if clip:
        return (np.clip(matrix, in_min, in_max) - in_min)*(((out_max-out_min)/(in_max-in_min))+out_min)
    # or just do the scaling (can result in lower or higher values than out_min and out_max)
    else:
        return (matrix - in_min)*(((out_max-out_min)/(in_max-in_min))+out_min)


def square_over_bg(
    x: int,
    y: int,
    img_size: int = 512,
    square_size: int = 50,
) -> torch.Tensor:
    """
    Create a binary image of a square over a black background.

    Args:
        x (int): The x coordinate of the top-left corner of the square.
        y (int): The y coordinate of the top-left corner of the square.
        img_size (int, optional): The size of each side of the image. Defaults to 512.
        square_size (int, optional): The size of each side of the square. Defaults to 50.

    Returns:
        torch.Tensor: _description_
    """
    # create a black image
    img = torch.zeros((img_size, img_size))
    # set the square to white
    img[y:y + square_size, x:x + square_size] = 1

    return img


def square_over_bg_falloff(
    x: int,
    y: int,
    img_size: int = 64,
    square_size: int = 2,
    falloff_mult: int = 0.1
) -> torch.Tensor:
    """
    Create a binary image of a square over a black background with a falloff.

    Args:
        x (int): The x coordinate of the top-left corner of the square.
        y (int): The y coordinate of the top-left corner of the square.
        img_size (int, optional): The size of each side of the image. Defaults to 512.
        square_size (int, optional): The size of each side of the square. Defaults to 50.
        falloff_mult (int, optional): The falloff multiplier. Defaults to 0.5.

    Returns:
        torch.Tensor: _description_
    """
    # create a black image
    img = torch.zeros((img_size, img_size))
    # set the square to white
    img[y:y + square_size, x:x + square_size] = 1
    # create falloff
    falloff = torch.zeros((img_size, img_size))
    _x, _y = x + square_size / 2, y + square_size / 2
    i, j = torch.meshgrid(torch.arange(img_size), torch.arange(img_size))
    v_to_square = torch.stack(
        [i, j]) - torch.tensor([_y, _x], dtype=torch.float32).view(2, 1, 1)
    v_length = torch.norm(v_to_square, dim=0)
    falloff = 1 - torch.clip(scale(v_length, 0, img_size,
                             0, img_size, exp=falloff_mult) / img_size, 0, 1)

    return torch.clip(img + falloff, 0, 1)
