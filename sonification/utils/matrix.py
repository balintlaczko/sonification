import numpy as np
import torch
import cv2
import platform
from .tensor import scale
import colorsys
from sklearn.neighbors import KDTree


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
    falloff_mult: int = 0.1,
    color: list = None,
) -> torch.Tensor:
    """
    Create an image of a square over a black background with a falloff.

    Args:
        x (int): The x coordinate of the top-left corner of the square.
        y (int): The y coordinate of the top-left corner of the square.
        img_size (int, optional): The size of each side of the image. Defaults to 512.
        square_size (int, optional): The size of each side of the square. Defaults to 50.
        falloff_mult (int, optional): The falloff multiplier. Defaults to 0.5.
        color (list, optional): A color vector to multiply the image with. If None, single-channel images are generated. Defaults to None.

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
    img = torch.clip(img + falloff, 0, 1)
    if color != None:
        img = img.unsqueeze(-1) * torch.tensor(color)

    return img


def matrix2binary(matrix: np.ndarray) -> np.ndarray:
    """ 
    Returns a binary matrix where each non-zero cell of the input is denoted with 1.

    Args:
        matrix (np.ndarray): The input matrix.

    Returns:
        np.ndarray: The (1-channel) binary matrix. 
    """
    matrix_bin = np.where(matrix != 0, 1, 0)
    if len(matrix.shape) > 2:
        return np.clip(np.sum(matrix_bin, axis=2), 0, 1)
    return matrix_bin


def matrix_crop(matrix: np.ndarray, crop_left: int, crop_right: int, crop_top: int, crop_bottom: int) -> np.ndarray:
    """
    Crops a matrix by the amount of pixels specified for each side.
    Returns the cropped matrix.

    Args:
        matrix (np.ndarray): The matrix to crop.
        crop_left (int): Size of the crop on the left, measured in pixels.
        crop_right (int): Size of the crop on the left, measured in pixels.
        crop_top (int): Size of the crop on the left, measured in pixels.
        crop_bottom (int): Size of the crop on the left, measured in pixels.

    Returns:
        np.ndarray: The cropped matrix.
    """
    height, width = matrix.shape[:2]
    return matrix[crop_top:height-crop_bottom, crop_left:width-crop_right, ...]


def matrix_crop_at(matrix: np.ndarray, crop_left: int, crop_right: int, crop_top: int, crop_bottom: int) -> np.ndarray:
    """
    Crops a matrix at the specified cells. Returns the cropped matrix.

    Args:
        matrix (np.ndarray): The matrix to crop.
        crop_left (int): The horizontal coordinate of the top left point of the crop.
        crop_right (int): The horizontal coordinate of the bottom right point of the crop.
        crop_top (int): The vertical coordinate of the top left point of the crop.
        crop_bottom (int): The vertical coordinate of the bottom right point of the crop.

    Returns:
        np.ndarray: The cropped matrix.
    """
    return matrix[crop_top:crop_bottom, crop_left:crop_right, ...]


def matrix_crop_around(matrix: np.ndarray, crop_side: int) -> np.ndarray:
    """
    Crops a matrix from all sides by the amount of pixels specified.
    Returns the cropped matrix.

    Args:
        matrix (np.ndarray): The matrix to crop around.
        crop_side (int): The size of the crop on each side, measured in pixels. The width/height of the cropped matrix will be 2*crops_side shorter than the input.

    Returns:
        np.ndarray: The cropped matrix.
    """
    return matrix_crop(matrix, crop_side, crop_side, crop_side, crop_side)


def matrix_crop_empty(matrix: np.ndarray) -> np.ndarray:
    """
    Automatically remove the empty padding around a matrix (image). The algorithm extracts the middle row
    and the middle column from the matrix, then iteratively walk from the edges inwards until it
    finds a non-zero cell value on each four sides. Returns the cropped matrix

    Args:
        matrix (np.ndarray): The matrix to be cropped.

    Returns:
        np.ndarray: The cropped matrix
    """

    matrix_gray = matrix
    if len(matrix.shape) > 2:
        matrix_gray = cv2.cvtColor(matrix, cv2.COLOR_BGR2GRAY)
    matrix_bin = np.where(matrix_gray > 0, 1, 0)
    height, width = matrix.shape[:2]
    mid_row = matrix_bin[int(height/2), :]
    mid_col = matrix_bin[:, int(width/2)]

    autocrop_left = 0
    for i in range(width):
        if mid_row[i] == 0:
            continue
        else:
            autocrop_left = i
            break

    autocrop_right = 0
    for i in range(width):
        if mid_row[-1 * (i+1)] == 0:
            continue
        else:
            autocrop_right = i
            break

    autocrop_top = 0
    for i in range(height):
        if mid_col[i] == 0:
            continue
        else:
            autocrop_top = i
            break

    autocrop_bottom = 0
    for i in range(height):
        if mid_col[-1 * (i+1)] == 0:
            continue
        else:
            autocrop_bottom = i
            break

    return matrix_crop(matrix, autocrop_left, autocrop_right, autocrop_top, autocrop_bottom)


def matrix_to_3plane(matrix: np.ndarray) -> np.ndarray:
    """
    Create a 3-plane ("rgb") matrix from a single channel one
    by copying the matrix on each channel.

    Args:
        matrix (np.ndarray): The single channel matrix to copy to the planes.

    Returns:
        np.ndarray: The 3-plane matrix.
    """
    out_matrix = np.zeros(matrix.shape[:2] + (3,))
    for i in range(3):
        if len(matrix.shape) > 2:
            out_matrix[:, :, i] = matrix[..., 0]
        else:
            out_matrix[:, :, i] = matrix
    return out_matrix


def extract_unique_masks_bbox(contours_matrix: np.ndarray, discard_bg: bool = False, check_active_only: bool = False, visualize: bool = False) -> list:
    """
    Extracts all unique masks from an input contours matrix. It acquires the masks by
    marching all rows in the bounding box of the matrix, generating a mask with
    cv2.floodFill, and saving all unique masks to the output list. Optionally it also
    discards the background mask that is assumed to be the largest area mask.

    Args:
        contours_matrix (np.ndarray): A matrix containing the annotation contours.
        discard_bg (bool, optional): Whether to discard the background mask (the largest area mask). Defaults to False.
        check_active_only (bool, optional): Whether to only check floodfill masks on active pixels. Defaults to False.
        visualize (bool, optional): Whether to visualize the bounding box. Defaults to False.

    Returns:
        list: A list of np.ndarray matrices with the unique masks extracted from the input.
    """
    contours_bin = matrix2binary(contours_matrix)
    h, w = contours_bin.shape[:2]
    _unique_masks = []
    x, y, width, height = cv2.boundingRect(contours_bin.astype(np.uint8))
    if visualize:
        contours_bin_3plane = matrix_to_3plane(
            contours_bin*255).astype(np.uint8)
        contours_w_rect = cv2.rectangle(
            contours_bin_3plane, (x, y), (x+width, y+height), (0, 255, 0), 1)
        view(contours_w_rect, text="Finding unique submasks...")
    # adding a 1-pix padding around to catch the bg
    x -= 1
    y -= 1
    width += 1
    height += 1
    # clamp x and y to 0
    x = max(x, 0)
    y = max(y, 0)
    # loop through rows in bbox
    for k in range(height):
        # march through row in bbox
        for i in range(width):
            if check_active_only and contours_bin[k+y, i+x] == 0:
                continue
            # seed_point = (k+y, i+x)
            seed_point = (i+x, k+y)
            mask_buf = contours_bin.astype(np.uint8)
            mask = np.zeros((h+2, w+2), np.uint8)
            cv2.floodFill(mask_buf, mask, seed_point, (255, 255, 255))
            mask_buf = np.where(mask_buf == 255, 255, 0)
            # if this is the first, just append it
            if len(_unique_masks) == 0:
                _unique_masks.append(mask_buf)
            # otherwise compare it to all unique masks collected so far
            # if it is different from all of them, then append to output
            else:
                num_current_masks = len(_unique_masks)
                num_different_masks = 0
                for i in range(num_current_masks):
                    unique_mask = _unique_masks[i]
                    if not np.array_equal(mask_buf, unique_mask):
                        num_different_masks += 1
                if num_different_masks == num_current_masks:
                    _unique_masks.append(mask_buf)
    if discard_bg and len(_unique_masks) <= 1:
        if visualize:
            view(matrix2binary(
                _unique_masks[0])*255, text="Error: only this submask exists")
        raise Exception(
            "Only one unique mask found. That suggests that the bbox area is empty of full.")
    # optionally discard background (the one with the largest area)
    if discard_bg:
        _sums = np.zeros(len(_unique_masks))
        # measure the area of each mask
        for i, found_mask in enumerate(_unique_masks):
            the_sum = np.sum(found_mask/255)
            _sums[i] = the_sum
        # discard the largest
        _unique_masks = [_unique_masks[i] for i in range(
            len(_unique_masks)) if i != np.argmax(_sums)]
    return _unique_masks


def sort_matrices_by_size(matrices: list) -> tuple:
    """
    Sorts a list of matrices based on the amount of non—zero cells in them.

    Args:
        matrices (list): A list of matrices to sort.

    Returns:
        tuple: Where the first element is the sorted list of matrices, the second is the sorted indices.
    """
    matrix_sums = np.zeros(len(matrices))
    outlist = list(matrix_sums)
    for i, matrix in enumerate(matrices):
        matrix_sums[i] = np.sum(matrix2binary(matrix))
    order = np.argsort(matrix_sums)
    for i, j in enumerate(order):
        outlist[i] = matrices[j]
    return outlist, order


def close_matrix(matrix: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Convenience wrapper around `cv2.morphologyEx` to perform a morphological close operation on an incoming matrix. It is necessary to convert the matrix to
    8-bit for this.

    Args:
        matrix (np.ndarray): The matrix to close.
        kernel_size (int, optional): The size of the kernel's sides. Defaults to 5.

    Returns:
        np.ndarray: The close matrix as 8-bit.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    bin_matrix = np.where(matrix > 0, 255, 0).astype(np.uint8)
    return cv2.morphologyEx(bin_matrix, cv2.MORPH_CLOSE, kernel)


def mask_is_filled(mask: np.ndarray) -> bool:
    """
    Checks whether the input binary mask is filled or not.

    Args:
        mask (np.ndarray): A matrix to examine. All non-zero cells will be considered active.

    Returns:
        bool: Whether the mask is filled or not.
    """
    mask_bin = matrix2binary(mask).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        image=mask_bin, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    contours_mask = np.zeros_like(mask_bin, dtype=np.uint8)
    cv2.drawContours(image=contours_mask, contours=contours, contourIdx=-1,
                     color=(255, 255, 255), thickness=1, lineType=cv2.LINE_4)
    contours_mask_closed = close_matrix(contours_mask, 3)
    masks_in_contours = extract_unique_masks_bbox(
        contours_mask_closed, discard_bg=True)
    if len(masks_in_contours) != 2:
        return False
    # if there are two
    sorted_by_size, _ = sort_matrices_by_size(masks_in_contours)
    smaller, larger = sorted_by_size
    sum_mask = np.sum(matrix2binary(mask))
    sum_smaller = np.sum(matrix2binary(smaller))
    sum_larger = np.sum(matrix2binary(larger))
    if np.abs(sum_mask - sum_larger) < np.abs(sum_mask - sum_smaller):
        return True
    return False


def generate_equidistant_colors_rgb(num):
    """ 
    Generates an arbitrary number of equidistant colors (in hue space).

    Args:
        num (int): The number of colors to generate.

    Returns:
        list: The list of generated 8-bit rgb colors as tuples. 
    """
    return [tuple(i*255 for i in colorsys.hsv_to_rgb(x*1.0/num, 1, 1)) for x in range(num)]


def get_label_centroid(labels_matrix, label):
    label_matrix = np.where(labels_matrix == label, 255, 0)
    M = cv2.moments(label_matrix, binaryImage=True)
    # calculate x,y coordinate of center
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return np.array([cx, cy])


def sort_masks_horizontally(masks: list) -> tuple:
    """
    Sorts a list of masks (binary matrices) based on the horizontal order of their centroids.

    Args:
        masks (list): The list of masks to sort.

    Returns:
        tuple: Where the first elemenet is the sorted list of masks, and the second is the sorted indices.
    """
    x_coords = np.zeros(len(masks))
    outlist = list(x_coords)
    for i, mask in enumerate(masks):
        mask_bin = matrix2binary(mask)
        x, y = get_label_centroid(mask_bin, 1)
        x_coords[i] = x
    order = np.argsort(x_coords)
    for i, j in enumerate(order):
        outlist[i] = masks[j]
    return outlist, order


def kmeans_color_quantization(image: np.ndarray, clusters: int, rounds: int = 1) -> np.ndarray:
    """
    Quantizes colors in an image via k-means clustering. Returns the quantized image.

    Args:
        image (np.ndarray): The image to quantize the colors in.
        clusters (int): The number of desired colors in the resulting image (including background).
        rounds (int, optional): Number of clustering attempts. Defaults to 1.

    Returns:
        np.ndarray: The clustered image.
    """
    h, w = image.shape[:2]
    samples = np.zeros([h*w, 3], dtype=np.float32)
    count = 0
    for x in range(h):
        for y in range(w):
            samples[count] = image[x][y]
            count += 1

    compactness, labels, centers = cv2.kmeans(samples,
                                              clusters,
                                              None,
                                              (cv2.TERM_CRITERIA_EPS +
                                               cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001),
                                              rounds,
                                              cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((image.shape))


def colors2labels(color_matrix: np.ndarray) -> np.ndarray:
    """
    Return a labels matrix from a colors matrix, where each unique color
    will get a unique label. Returns the labels matrix.

    Args:
        color_matrix (np.ndarray): A 3-channel colored image matrix to label.

    Returns:
        np.ndarray: The 1-channel labels matrix.
    """
    unique_colors = np.unique(
        color_matrix.reshape(-1, color_matrix.shape[2]), axis=0).astype(np.uint8)
    unique_colors_wo_bg = np.array(
        [color for color in unique_colors if not np.array_equal(color, np.array([0, 0, 0]))])
    labels_matrix = color_matrix.copy()
    for i, color in enumerate(unique_colors_wo_bg):
        labels_matrix = np.where(color_matrix == color, i+1, labels_matrix)
    return labels_matrix[:, :, 0]


def get_largest_submask(matrix: np.ndarray) -> np.ndarray:
    """
    Extracts the largest area floodfill mask in the input matrix. It does so
    by chaining extract_unique_masks_bbox(discard_bg=True) into
    sort_matrices_by_size to get the one with the largest area.

    Args:
        matrix (np.ndarray): The matrix to extract the largest submask from. All non-zero cells will be considered active.

    Returns:
        np.ndarray: The largest unique mask.
    """
    matrix_bin = matrix2binary(matrix)
    unique_submasks = extract_unique_masks_bbox(matrix_bin, discard_bg=True)
    ordered_submasks, _ = sort_matrices_by_size(unique_submasks)
    return ordered_submasks[-1]


def get_most_fitting_submask(matrix: np.ndarray, discard_bg: bool = False, check_active_only=False, visualize: bool = False) -> np.ndarray:
    """
    Extracts the most fitting floodfill mask in the input matrix. It first extracts all unique
    submasks via extract_unique_masks_bbox, then finds the one with the area that is the most 
    similar to the input.

    Args:
        matrix (np.ndarray): The matrix to extract the most fitting submask from. All non-zero cells will be considered active.
        discard_bg (bool, optional): Whether to discard the background mask (the largest area mask). Defaults to False.
        check_active_only (bool, optional): Whether to only check floodfill masks on active pixels. Optimization for working with contours only. Defaults to False.
        visualize (bool, optional): Whether to visualize the bounding box, in extract_unique_masks_bbox. Defaults to False.

    Returns:
        np.ndarray: The most fitting unique mask.
    """
    matrix_bin = matrix2binary(matrix)
    input_sum = np.sum(matrix_bin)
    unique_submasks = extract_unique_masks_bbox(
        matrix_bin, discard_bg=discard_bg, check_active_only=check_active_only, visualize=visualize)
    submask_sums = np.zeros(len(unique_submasks))
    for i, mask in enumerate(unique_submasks):
        mask_bin = matrix2binary(mask)
        submask_sums[i] = np.sum(mask_bin)
    diffs = np.abs(submask_sums - input_sum)
    index_of_most_fitting = np.argmin(diffs)
    return unique_submasks[index_of_most_fitting]


def drop_small_submasks(matrix: np.ndarray, area_thresh: int = 20, discard_bg: bool = False, check_active_only=False, visualize: bool = False) -> np.ndarray:
    """
    Drops small floodfill submasks in the input matrix. It first extracts all unique
    submasks via extract_unique_masks_bbox, then drops the ones that have an area
    below area_thresh. Returns the binary summed matrix of the remaining submasks.

    Args:
        matrix (np.ndarray): The matrix to drop small submasks from. All non-zero cells will be considered active.
        area_thresh (int, optional): The area below which a submask should be dropped.
        discard_bg (bool, optional): Whether to discard the background mask (the largest area mask). Defaults to False.
        check_active_only (bool, optional): Whether to only check floodfill masks on active pixels. Optimization for working with contours only. Defaults to False.
        visualize (bool, optional): Whether to visualize the bounding box, in extract_unique_masks_bbox. Defaults to False.

    Returns:
        np.ndarray: The binary summed matrix of the remaining submasks.
    """
    matrix_bin = matrix2binary(matrix)
    unique_submasks = extract_unique_masks_bbox(
        matrix_bin, discard_bg=discard_bg, check_active_only=check_active_only, visualize=visualize)
    output_matrix = np.zeros_like(matrix_bin)
    for mask in unique_submasks:
        mask_bin = matrix2binary(mask)
        if np.sum(mask_bin) > area_thresh:
            output_matrix += mask_bin

    return matrix2binary(output_matrix)


def labels_keep_largest_submasks(labels_matrix: np.ndarray) -> np.ndarray:
    """
    Filters a labels matrix by keeping the result of get_largest_submask
    on all label matrices. Useful for filtering small patches of label noise,
    when only one continuous patch per label is expected to be found.

    Args:
        labels_matrix (np.ndarray): A labels matrix with a single plane where pixel values denote labels.

    Returns:
        np.ndarray: The filtered labels matrix.
    """
    unique_labels = [label for label in np.unique(labels_matrix) if label != 0]
    labels_matrix_filtered = np.zeros_like(labels_matrix)
    for label in unique_labels:
        label_matrix = np.where(labels_matrix == label, label, 0)
        largest_submask = get_largest_submask(label_matrix)
        label_matrix_filtered = np.where(
            largest_submask > 0, label, 0).astype(np.uint8)
        labels_matrix_filtered += label_matrix_filtered
    return labels_matrix_filtered


def labels_keep_most_fitting_submasks(labels_matrix: np.ndarray) -> np.ndarray:
    """
    Filters a labels matrix by keeping the result of get_most_fitting_submask
    on all label matrices. Useful for filtering small patches of label noise,
    when only one continuous patch per label is expected to be found.

    Args:
        labels_matrix (np.ndarray): A labels matrix with a single plane where pixel values denote labels.

    Returns:
        np.ndarray: The filtered labels matrix.
    """
    unique_labels = [label for label in np.unique(labels_matrix) if label != 0]
    labels_matrix_filtered = np.zeros_like(labels_matrix)
    for label in unique_labels:
        label_matrix = np.where(labels_matrix == label, label, 0)
        largest_submask = get_most_fitting_submask(label_matrix)
        label_matrix_filtered = np.where(
            largest_submask > 0, label, 0).astype(np.uint8)
        labels_matrix_filtered += label_matrix_filtered
    return labels_matrix_filtered


def unique_points(points: list) -> list:
    """
    Takes a list of points and keeps only the unique ones in the result.

    Args:
        points (list): tuples or lists of coordinates.

    Returns:
        list: The list of unique points.
    """
    points_unique = []
    for point in points:
        if len(points_unique) == 0:
            points_unique.append(point)
            continue
        num_equals = 0
        for unique_point in points_unique:
            if np.array_equal(np.array(unique_point), np.array(point)):
                num_equals += 1
                break
        if num_equals == 0:
            points_unique.append(point)

    return points_unique


def points2groups(points: np.array, radius: float = 7) -> tuple:
    """
    Groups points closer to each other than radius using c.
    Collects isolated points (points that have no neighbors within radius) into a
    separate list. Returns the the list with the isolated point ids, and the list
    of neighbor groups as a tuple.

    Args:
        points (np.array): The array of points.
        radius (float, optional): The query distance for the KDTree. Points that are closer to each other than this distance will be collected into a group. Defaults to 7.

    Returns:
        tuple: Two lists, the first is the list with the isolated points ids, the seconds is the list of neighbor groups (lists).
    """

    isolated_points_ids = []  # 1D list with ids of isolated points
    # 2D list with lists of point ids that are within radius to each other
    neighbor_groups = []

    kdtree = KDTree(points)

    for i in range(len(points)):
        group = kdtree.query_radius(np.array([points[i]]), r=radius)[0]
        # if the group only contains a single id then append to isolated_points_ids
        if len(group) == 1:
            isolated_points_ids.append(i)
        else:
            # if this is the first group, append
            if len(neighbor_groups) == 0:
                neighbor_groups.append(list(group))
                continue
            # if not the first group then compare it to all previous groups
            num_equals = 0
            for neighbor_group in neighbor_groups:
                if np.array_equal(np.array(group), np.array(neighbor_group)):
                    num_equals += 1
                    break
            # and only append to neighbor_groups if this group is unique so far
            if num_equals == 0:
                neighbor_groups.append(list(group))

    return isolated_points_ids, neighbor_groups


def groups2centers(groups: list, points: np.ndarray) -> list:
    """
    Calculate the 2D centers for groups of points. Returns a list
    of points where each point is a center of a respective group.

    Args:
        groups (list): The list of groups (lists of ids) to calculate centers for.
        points (np.ndarray): The array of points to which the ids correspond to.

    Returns:
        list: The list of 2D centers for each input group.
    """

    groups_centers = []

    # loop through groups
    for group in groups:
        # look up the coordinates for each id in the group
        group_xy = np.zeros((len(group), 2))
        for i, ind in enumerate(group):
            group_xy[i] = points[ind]
        # calculate center: average X and average Y
        avg_x = int(round(np.sum(group_xy[:, 0]) / len(group)))
        avg_y = int(round(np.sum(group_xy[:, 1]) / len(group)))
        # append to output list
        groups_centers.append([avg_x, avg_y])

    return groups_centers


def groups_filter_subsets(groups: list) -> list:
    """
    Filters smaller subsets of larger groups (e.g. if there is
    a [4, 5, 6], filter any [4, 5], [5, 6] or [4, 6]). Relies
    on set.issubset. Returns the list of groups that are not
    subsets of any other group.

    Args:
        groups (list): The list of groups (lists) to filter.

    Returns:
        list: The filtered list of groups that are not subsets of any other group.
    """

    largest_group_size = max([len(group) for group in groups])
    groups_reduced = []

    # loop through all possible group sizes from smaller to larger
    for size in range(largest_group_size-1):
        # loop through groups
        for group in groups:
            # if group size is fits what we are looking for
            if len(group) == size+2:
                num_subsets = 0
                # then check if it is a subset of any larger group
                for compare_group in groups:
                    if len(compare_group) > size+2:
                        if set(group).issubset(set(compare_group)):
                            num_subsets += 1
                            break
                # append to output list if it is not a subset
                if num_subsets == 0:
                    groups_reduced.append(group)

    return groups_reduced


def reduce_points(points: np.array, radius: float = 7) -> np.array:
    """
    Reduces a collection of points by clustering points (using sklearn.neighbors.KDTree)
    that are closer than radius to each other and replacing them with their geometrical
    center point.

    Args:
        points (np.array): The array of points to reduce.
        radius (float, optional): The query distance for the KDTree. Points that are closer to each other than this distance will be collected into a group. Defaults to 7.

    Returns:
        np.array: The array of reduced points.
    """
    # cluster points with KDTree using radius
    isolated_points, neighbors = points2groups(points, radius=radius)
    # if no clumps were made, just return input
    if len(neighbors) == 0:
        return points
    # filter subsets of groups
    neighbors_reduced = groups_filter_subsets(neighbors)
    # calculate centers for each group
    neighbors_centers = groups2centers(neighbors_reduced, points)
    # concatenate coordinates of isolated points with the centers of the clusters
    points_reduced = np.concatenate(
        (points[isolated_points], np.array(neighbors_centers)))

    return points_reduced


def points2pairs_closest(points: np.ndarray) -> list:
    """
    Organize an array of points into pairs of points based on proximity.
    Uses sklearn.neighbors.KDTree. Returns a list of pairs of indices.

    Args:
        points (np.ndarray): The points to organize into pairs.

    Returns:
        list: The pairs of point indices.
    """
    pairs_tree = KDTree(points)
    pair_groups = []
    # create pairs of close points (k=2)
    for i in range(len(points)):
        pair_inds = pairs_tree.query(
            np.array([points[i]]), k=2, return_distance=False)[0]
        if len(pair_groups) == 0:
            pair_groups.append(list(pair_inds))
            continue
        num_equals = 0
        for pair_group in pair_groups:
            if np.array_equal(np.sort(np.array(pair_group)), np.sort(np.array(pair_inds))):
                num_equals += 1
                break
        if num_equals == 0:
            pair_groups.append(list(pair_inds))

    return pair_groups


def points2pairs_percentile(points: np.ndarray, percentile: float = 99) -> list:
    """
    Organize an array of points into pairs based on the N percentile
    distance between all points. Uses sklearn.neighbors.KDTree.
    Returns a list of pairs of indices.

    Args:
        points (np.ndarray): The points to organize into pairs.
        percentile (float, optional): The percentile distance to use as a threshold for a valid pair. Defaults to 99.

    Returns:
        list: The pairs of point indices.
    """
    pairs_tree = KDTree(points)
    num_points = len(points)
    pair_groups = []
    distances = np.zeros((num_points, num_points))
    # create distance matrix
    for i in range(num_points):
        dist, ind = pairs_tree.query(np.array([points[i]]), k=num_points)
        sorter = np.argsort(ind[0])
        sorted_row = np.array(dist[0])[sorter]
        distances[i, :] = sorted_row
    # mean with outlier removal
    mean_dist = np.percentile(distances, percentile)
    # find all valid pairs:
    for y in range(num_points):
        for x in range(num_points):
            dist = distances[y, x]
            if dist == 0:
                continue
            if dist <= mean_dist:
                pair_groups.append([y, x])

    return pair_groups


def floodfill_from_point(matrix: np.ndarray, point: list) -> np.ndarray:
    """
    Convenience function for cv2.floodFill using a specific point as seed.
    Returns the sum of the input and the fill.

    Args:
        matrix (np.ndarray): The matrix to floodfill.
        point (list): The seed point for cv2.floodFill.

    Returns:
        np.ndarray: The binary matrix of the filled input (input + fill).
    """
    matrix_bin = matrix2binary(matrix)
    mask_buf = matrix_bin.astype(np.uint8)
    h, w = mask_buf.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(mask_buf, mask, tuple(point), (255, 255, 255))
    return np.where(mask_buf == 255, 1, 0) + matrix_bin
