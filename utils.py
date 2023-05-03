# %%
# imports

import scipy.io.wavfile as wav
from scipy import interpolate
import os
import pandas as pd
import numpy as np
import cv2
import librosa
import platform
from numba import jit, njit, prange
from musicalgestures._utils import roundup, generate_outfilename, MgProgressbar
from typing import List


# %%
# function: folder2dataset

def folder2dataset(folder: str, image_extension: str = ".tif") -> pd.DataFrame:
    """
    Parses a folder of images and returns a Pandas DataFrame, based on the file names and image properties.
    Uses `image2dataset` internally.

    Args:
        folder (str): Path to the folder where the images are located.
        image_extension (str, optional): The type of image file extension to look for. Defaults to ".tif".

    Returns:
        pd.DataFrame: The dataset of the images.
    """
    # list all files in the folder, filter for files that have the same extension as `image:extension`
    images = [file for file in sorted(os.listdir(folder)) if os.path.splitext(file)[
        1].lower() == image_extension]
    # prefix all file names in the list with the folder path to create full paths
    images_abspath = [os.path.join(folder, image) for image in images]
    # pass the list to `images2dataset` that will return a pd.DataFrame
    return images2dataset(images_abspath)


# %%
# function: image2dataset

def images2dataset(images: List[str]) -> pd.DataFrame:
    """
    Creates a dataset from a list of file paths. It parses the image file names and the image properties.
    It expects the following file naming scheme:
    date_step_MIP_time_fedvisitingpoint_colorspace.ext

    The resulting pd.DataFrame will have the following columns:
    date, stage, group, step, fed, visiting_point, color_space, width, height, pixel_min, pixel_max, path.

    Examples:
    "20210223_BST_MIP_PBSG3_RFP.tif" - that would be interpreted as
    date="20210223", stage="bst", group="pbsg", step=0, fed=0, visiting_point=3, color_space=rfp
    "20210223_ST_MIP_T01_PBSG2_GFP.tif" - that would be interpreted as
    date="20210223", stage="st", group="pbsg", step=1, fed=0, visiting_point=2, color_space=gfp.

    Args:
        images (List[str]): A list of strings with the paths to the images.

    Returns:
        pd.DataFrame: The dataset of the images.
    """
    # define column names
    column_names = ["date", "stage", "group", "step", "fed",
                    "visiting_point", "color_space", "width", "height", "pixel_min", "pixel_max", "path"]
    # create DataFrame with the columns
    df = pd.DataFrame(columns=column_names)
    # for each image in the input list
    for image_abspath in images:
        # create a new dataset entry
        new_sample = {}
        # take the file name (without path) and split it by the underscores
        image_basename = os.path.basename(image_abspath)
        filename_parts = os.path.splitext(image_basename)[0].split("_")
        # handle images with no/unknown colorspace indication
        if filename_parts[-1].lower() not in ["gfp", "rfp", "rgb"]:
            # check if it is missing and if so, add a placeholder "NA"
            if filename_parts[-1].lower().startswith("esf") or filename_parts[-1].lower().startswith("pbsg"):
                filename_parts.append("NA")
        # add date
        new_sample["date"] = [filename_parts[0]]
        # add stage (bst, st or rf)
        new_sample["stage"] = filename_parts[1].lower()
        # add group (esf, pbsg, pbsga, pbsgb, etc...)
        new_sample["group"] = filename_parts[-2][:-1].lower()
        # add step
        if filename_parts[1].lower() == "bst":
            new_sample["step"] = [0]
        else:
            new_sample["step"] = [int(filename_parts[3][1:])]
        # add fed and visiting_point
        fed_code, visiting_point = filename_parts[-2][:-
                                                      1], filename_parts[-2][-1]
        fed = 1 if fed_code.lower().startswith("esf") else 0
        new_sample["fed"] = [fed]
        new_sample["visiting_point"] = [int(visiting_point)]
        # add color space
        new_sample["color_space"] = [filename_parts[-1].lower()]
        # read matrix
        bitdepth = 16 if new_sample["color_space"][0].lower() in [
            "gfp", "rfp"] else 8
        read_param = -1 if bitdepth == 16 else 1
        image_matrix = cv2.imread(image_abspath, read_param)
        # add width height
        new_sample["height"], new_sample["width"] = image_matrix.shape[:2]
        # add min max
        new_sample["pixel_min"], new_sample["pixel_max"] = np.min(
            image_matrix), np.max(image_matrix)
        # add path
        new_sample["path"] = image_abspath
        new_sample_df = pd.DataFrame.from_dict(new_sample)
        df = pd.concat([df, new_sample_df], ignore_index=True)
    return df

# %%
# function: group_by_colorspace


def group_by_colorspace(images: List[str]) -> dict:
    """
    Take a list of file paths (to images) and group them into a dictionary based on colorspace. It expects 
    that the colorspace is indicated at the end of the file name, preceded by underscore ("_"). It looks for 
    3-letter codes of either "rgb", "gfp", or "rfp" (not case-sensitive).
    The resulting dictionary will have three keys, "gfp", "rfp", and "rgb", pointing to lists of file names.
    Examples:
    "20210223_ST_MIP_T01_PBSG1_RGB.tif" - will get sorted to the list of files under the "rgb" key.
    "20210223_BST_MIP_ESF4_RFP.tif" - will get sorted to the list of files under the "rfp" key.

    Args:
        images (List[str]): The list of image file paths.

    Returns:
        dict: A dictionary with keys: "gfp", "rfp", and "rgb", pointing to 3 corresponding lists of file paths.
    """
    # initialize list containers
    images_gfp = []
    images_rfp = []
    images_rgb = []
    # for each image in the list:
    for image in images:
        # split the file name in parts using underscores ("_")
        filename_parts = os.path.splitext(image)[0].split("_")
        # take the last part in lowercase as the colorspace
        colorspace = filename_parts[-1].lower()
        # append the file path to either of the lists if the lowercase colorspace is either "rgb", "gfp", or "rfp"
        if colorspace == "rgb":
            images_rgb.append(image)
        elif colorspace == "gfp":
            images_gfp.append(image)
        elif colorspace == "rfp":
            images_rfp.append(image)
    # return a dictionary with the 3 lists
    return {
        "gfp": images_gfp,
        "rfp": images_rfp,
        "rgb": images_rgb
    }


# %%
# function: view

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
    cv2.waitKey(0)
    # avoid hanging windows on Mac OS
    if platform.system() == "Darwin":
        cv2.destroyAllWindows()
        cv2.waitKey(1)


# %%
# function: stretch_contrast

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


# %%
# class: ImageSequence

class ImageSequence():
    """
    The representation of a sequence of images based on an input dataset (pd.DataFrame), typically acquired by `images2dataset`
    or `folder2dataset`.
    """

    def __init__(self, sequence: pd.DataFrame) -> None:
        """
        Initializes the ImageSequence object from a provided pd.DataFrame, typically acquired by `images2dataset`
        or `folder2dataset`.

        Args:
            sequence (pd.DataFrame): The dataset to create the image sequence from.
        """
        # save internal reference to the DataFrame
        self.sequence = sequence
        # determine number of steps
        self.num_steps = len(self.sequence.step.unique())
        # calculate minimum and maximum values in the three colorspaces, and save them into a dictionary
        self.minmax = {
            "gfp": self.get_minmax("gfp"),
            "rfp": self.get_minmax("rfp"),
            "rgb": self.get_minmax("rgb"),
        }

    def get_minmax(
        self,
        color_space: str = "gfp",
    ) -> List[int]:
        """
        Returns the minimum and maximum cell values in `color_space` across the dataset `self.sequence`.

        Args:
            color_space (str, optional): The colorspace to assess. Defaults to "gfp".

        Returns:
            List[int]: The minimum and maximum cell values in the specified colorspace across the dataset.
        """
        # filter dataset at specified colorspace
        seq_at_colorspace = self.sequence.loc[self.sequence["color_space"] == color_space.lower(
        )]
        # calculate minimum and maximum cell values and return them as a list
        ds_pixel_min = min(seq_at_colorspace["pixel_min"].unique())
        ds_pixel_max = max(seq_at_colorspace["pixel_max"].unique())
        return ds_pixel_min, ds_pixel_max

    def get_path(
        self,
        step: int,
        color_space: str,
        visiting_point: int,
    ) -> str:
        """
        Returns the path of an image at a specified `step`, `color_space` and `visiting_point`.

        Args:
            step (int): The step corresponding to the image.
            color_space (str): The colorspace of the image (gfp, rfp or rgb).
            visiting_point (int): The visiting point corresponding to the image.

        Returns:
            str: The path to the image.
        """
        # filter for matching step, colorspace and visiting point, then return the
        # first (assumbed to be the single) value at "path" column
        return self.sequence.loc[(self.sequence["step"] == step) & (
            self.sequence["color_space"] == color_space) & (self.sequence["visiting_point"] == visiting_point)]["path"].values[0]

    def get_matrix(
        self,
        path: str,
        color_space: str,
    ) -> np.ndarray:
        """
        Returns the image matrix read from an image file.

        Args:
            path (str): The path to the file to read.
            color_space (str): The colorspace of the image. If "gfp" or "rfp" is specified, the image will be read in 16 bits, otherwise in 8 bits.

        Returns:
            np.ndarray: The image matrix.
        """
        bitdepth = 16 if color_space.lower() in ["gfp", "rfp"] else 8
        read_param = -1 if bitdepth == 16 else 1
        image_matrix = cv2.imread(path, read_param)
        return image_matrix

    def get_width_height(self) -> List[int]:
        """
        Returns the width and height of the first image in the dataset (assumed to be true for the other images as well).

        Returns:
            List[int]: The width and the height.
        """
        return self.sequence["width"].unique()[0], self.sequence["height"].unique()[0]

    def view(
        self,
        step: int = 0,
        color_space: str = "gfp",
        visiting_point: int = 1,
        normalize: bool = True,
        norm_percentile: float = None,
        scale: float = 1.0,
    ) -> None:
        """
        View an image from the sequence based on the specified step, colorspace and visiting point.
        Optionally apply normalization (using the whole distribution or a percentile specified by
        `norm_percentile`), or scaling.

        Args:
            step (int, optional): The step corresponding to the image. Defaults to 0.
            color_space (str, optional): The colorspace corresponding to the image. Defaults to "gfp".
            visiting_point (int, optional): The visiting point corresponding to the image. Defaults to 1.
            normalize (bool, optional): Whether to normalize the image in view. Defaults to True.
            norm_percentile (float, optional): If specified, the normalization will be based on the distribution of `norm_percentile`. Defaults to None.
            scale (float, optional): The scale (size) of the image relative to its original size. A `scale` of 1 will show the image in its original size, a `scale` of 0.5 will halve the width and height of the image. Defaults to 1.0.
        """
        # get the path and then the matrix of the image
        image_path = self.get_path(step, color_space, visiting_point)
        image = self.get_matrix(image_path, color_space.lower())
        # normalize if necessary using stretch_contrast
        if normalize:
            color_min, color_max = self.minmax[color_space.lower()]
            image = stretch_contrast(
                image, in_min=color_min, in_max=color_max, out_max=255, in_percentile=norm_percentile)
        # scale the image to the specified scale
        # roundup is used to enforce dimensions being divisibile by 4 for avoiding some compatibility issues
        if scale != 1.0:
            width, height = [roundup(dim*scale, 4) for dim in image.shape[:2]]
            image = cv2.resize(image, (width, height))
        # show the image
        # cv2.imshow(f'Step {step} | {color_space}', image.astype(np.uint8))
        # changed to fixed window ID so it is not jumping on the screen
        cv2.imshow('View', image.astype(np.uint8))
        cv2.waitKey(0)

    def render_image(
        self,
        step: int = 0,
        color_space: str = "gfp",
        visiting_point: int = 1,
        normalize: bool = True,
        norm_percentile: float = None,
        scale: float = 1.0,
        target_name: str = None,
        overwrite: bool = False,
    ) -> str:
        """
        Export an image corresponding to a specified step, visiting point and colorspace, with optional normalization
        or scaling. Typically used for exporting normalized versions of the images.

        Args:
            step (int, optional): The step corresponding to the image. Defaults to 0.
            color_space (str, optional): The colorspace corresponding to the image. Defaults to "gfp".
            visiting_point (int, optional): The visiting point corresponding to the image. Defaults to 1.
            normalize (bool, optional): Whether to normalize the image. Defaults to True.
            norm_percentile (float, optional): If specified, the normalization will be based on the distribution of `norm_percentile`. Defaults to None.
            scale (float, optional): The scale (size) of the image relative to its original size. A `scale` of 1 will render the image in its original size, a `scale` of 0.5 will halve the width and height of the image. Defaults to 1.0.
            target_name (str, optional): The target path to the exported image. Can end up being different from what was specified if `overwrite=False`. If not specified, a generic output path will be used based on the path and name of the source image. Defaults to None.
            overwrite (bool, optional): If False, the method will avoid overwriting existing files by incrementing `target_name`. Defaults to False.

        Returns:
            str: The path to the rendered image.
        """
        # fetch the image path and then matrix
        image_path = self.get_path(step, color_space, visiting_point)
        image = self.get_matrix(image_path, color_space.lower())
        # normalize if necessary
        if normalize:
            color_min, color_max = self.minmax[color_space.lower()]
            image = stretch_contrast(
                image, in_min=color_min, in_max=color_max, out_max=255, in_percentile=norm_percentile)
        # scale if necessary
        if scale != 1.0:
            width, height = [roundup(dim*scale, 4) for dim in image.shape[:2]]
            image = cv2.resize(image, (width, height))
        # if target_name was not specified, use the source image path/name with a "_render" suffix
        if target_name == None:
            target_name = f'{os.path.splitext(image_path)[0]}_render{os.path.splitext(image_path)[1]}'
        # avoid overwriting if necessary
        if not overwrite:
            target_name = generate_outfilename(target_name)
        # write matrix as 8-bit image
        cv2.imwrite(target_name, image.astype(np.uint8))
        return target_name

    def render_video(
        self,
        color_space: str = "gfp",
        visiting_point: int = 1,
        normalize: bool = True,
        norm_percentile: float = None,
        fps: int = 24,
        target_name: str = None,
        overwrite: bool = False,
    ) -> str:
        """
        Export the image sequence as a video with with the consecutive steps being frames in the video. Optionally normalize
        the frames in the output.

        Args:
            color_space (str, optional): The colorspace corresponding to the images. Defaults to "gfp".
            visiting_point (int, optional): The visiting point corresponding to the images. Defaults to 1.
            normalize (bool, optional): Whether to normalize the images. Defaults to True.
            norm_percentile (float, optional): If specified, the normalization will be based on the distribution of `norm_percentile`. Defaults to None.
            fps (int, optional): How many frames per second to have in the output video. Defaults to 24.
            target_name (str, optional): The target path to the exported video. Can end up being different from what was specified if `overwrite=False`. If not specified, a generic output path will be used based on the path to the images and a name following the naming scheme of the images (date_fedvisingpoint_COLOR.avi). Defaults to None.
            overwrite (bool, optional): If False, the method will avoid overwriting existing files by incrementing `target_name`. Defaults to False.

        Returns:
            str: The path to the rendered video.
        """
        # if target_name is not specified,
        if target_name == None:
            sequence_filtered = self.sequence.loc[(self.sequence["color_space"] == color_space.lower()) & (
                self.sequence["visiting_point"] == visiting_point)]
            row_0 = sequence_filtered.iloc[0]
            # get the path to the images
            path_to = os.path.dirname(row_0["path"])
            # get fed state (ESF for fed and PBSG for starved)
            fed_state_code = "ESF" if row_0["fed"] == 1 else "PBSG"
            # generate target_name using date, fed state, visiting point and colorspace
            target_name = f'{row_0["date"]}_{fed_state_code}{visiting_point}_{color_space.upper()}.avi'
            # join path and name
            target_name = os.path.join(path_to, target_name)
        # enforce avi extension
        else:
            target_name = os.path.splitext(target_name)[0] + '.avi'
        # avoid overwriting existing files if necessary
        if not overwrite:
            target_name = generate_outfilename(target_name)

        # initialize video writer with MJPG codec and the width and height of the images in the sequence
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        width, height = self.get_width_height()
        # create either an RGB VideoWriter, or a grayscale one
        out = None
        if color_space.lower() == "rgb":
            out = cv2.VideoWriter(target_name, fourcc, fps, (width, height))
        else:
            out = cv2.VideoWriter(target_name, fourcc, fps, (width, height), 0)

        # create progress bar
        pb_prefix = "Rendering image sequence:"
        pb = MgProgressbar(total=self.num_steps, prefix=pb_prefix)
        ii = 0
        # create a preview window for the render
        cv2.namedWindow("Rendering in progress...", 0)
        cv2.resizeWindow("Rendering in progress...", (300, 300))
        # loop through steps
        for step in range(self.num_steps):
            # fetch image path and then matrix
            image_path = self.get_path(step, color_space, visiting_point)
            image = self.get_matrix(image_path, color_space)
            # normalize if necessary
            if normalize:
                color_min, color_max = self.minmax[color_space.lower()]
                image = stretch_contrast(
                    image, in_min=color_min, in_max=color_max, out_max=255, in_percentile=norm_percentile)
            # write frame as 8-bit
            out.write(image.astype(np.uint8))
            # progress progress bar
            pb.progress(ii)
            ii += 1
            # preview rendered frame in window
            cv2.imshow("Rendering in progress...", image.astype(np.uint8))
            # preview for a 100 milliseconds
            if cv2.waitKey(100) != -1:  # (-1 means no key)
                continue
        # destroy preview window, release file, return path
        cv2.destroyAllWindows()
        out.release()
        return target_name

    def save_dataset(self, target_name: str) -> None:
        """
        Save internal dataset (a pd.DataFrame) to a csv file using pd.DataFrame.to_csv.

        Args:
            target_name (str): The name of the output csv file.
        """
        self.sequence.to_csv(target_name)


# %%
# class: Sinetable

class Sinetable():
    """
    A wavetable oscillator that can generate arbitrary length sine buffers or wav files, with a chosen 
    interpolation method, and with a chosen windowing applied.
    """

    def __init__(
        self,
        samples: int = 4096,
        interp: str = "cubic"
    ) -> None:
        """
        Initialize the SineTable object with an internal sine wave buffer of `samples` length, and
        an interpolator of `interp` kind.

        Args:
            samples (int, optional): The length of the internal sine wave buffer. Defaults to 4096.
            interp (str, optional): The kind of the interpolator. Uses scipy.interpolate.inter1d that supports 'linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', or 'next'. For more information consult the scipy documentation. Defaults to "cubic".
        """
        self.samples = samples
        self.x = np.arange(0, samples)
        self.y = np.sin(2 * np.pi * self.x / np.max(self.x))
        self.f = interpolate.interp1d(self.x, self.y, kind=interp)

    def sample(self, index) -> np.ndarray:
        """
        Sample the internal sine table at a specified index.

        Args:
            index (float or np.ndarray): The index to sample at. If the index is a fractional number the sample value will be interpolated with the objects interpolator. The `index` can also be an array of indices.

        Returns:
            np.ndarray: The sampled index or indices.
        """
        return self.f(index)

    def generate(
        self,
        sr: int = 44100,
        length_s: float = 1,
        freq: float = 440,
        gain_db: float = 0,
        window: str = "hann",
    ) -> np.ndarray:
        """
        A semi-optimized method to generate an arbitrary length sine wave buffer.

        Args:
            sr (int, optional): The sampling rate of the buffer. Defaults to 44100.
            length_s (float, optional): The length of the generated buffer in seconds. Defaults to 1.
            freq (float, optional): The frequency of the sine wave in the buffer. Defaults to 440.
            gain_db (float, optional): The gain to apply to the buffer in dB. Defaults to 0.
            window (str, optional): The windowing to apply to the buffer. Can be "hann" for np.hanning, "hamm" for np.hamming, "blackman" for np.blackman, "kaiser" for np.kaiser. If the string isn't any of these no windowing will be used. Defaults to "hann".

        Returns:
            np.ndarray: The generated buffer.
        """
        # define windowing function
        window_function = np.ones  # if "none" or not any of the ones below
        if window.lower() == "hann":
            window_function = np.hanning
        elif window.lower() == "hamm":
            window_function = np.hamming
        elif window.lower() == "blackman":
            window_function = np.blackman
        elif window.lower() == "kaiser":
            window_function = np.kaiser

        # create windowing buffer, sample buffer, initialize index counter and increment
        window_buf = window_function(int(np.ceil(length_s * sr)))
        buffer = np.zeros(int(np.ceil(length_s * sr)), )
        index = 0
        increment = freq * self.samples / sr

        # optimized function to generate an array of indices for the sampler
        @jit(nopython=True)
        def fill_indices(buffer, index, increment, table_length):
            output = np.zeros_like(buffer)
            for i in range(buffer.shape[0]):
                output[i] = index
                index += increment
                index %= table_length-1
            return output

        # generate indices for sampler
        indices = fill_indices(buffer, index, increment, self.samples)
        # generate interpolated values for the indices
        buffer = self.sample(indices)
        # apply gain
        amp = 10 ** (gain_db / 20)
        # apply windowing and return
        return buffer * amp * window_buf

    def write(
        self,
        target_name: str,
        sr: int = 44100,
        length_s: float = 1,
        freq: float = 440,
        gain_db: float = 0,
        window: str = "hann",
        overwrite: bool = False,
    ) -> str:
        """
        Generate an arbitrary length sine wave buffer and write it to a wav file. This is basically a wrapper around `generate`
        that also saves the resulting buffer to a file using scipy.io.wavfile.

        Args:
            target_name (str): The target name of the output file. Can end up being different from what was specified if `overwrite=False`.
            sr (int, optional): The sample rate to use. Defaults to 44100.
            length_s (float, optional): The length of the output file in seconds. Defaults to 1.
            freq (float, optional): The frequency of the sine wave. Defaults to 440.
            gain_db (float, optional): The gain to apply in dB. Defaults to 0.
            window (str, optional): The windowing to apply to the file. Can be "hann" for np.hanning, "hamm" for np.hamming, "blackman" for np.blackman, "kaiser" for np.kaiser. If the string isn't any of these no windowing will be used. Defaults to "hann".
            overwrite (bool, optional): If False, the method will avoid overwriting existing files by incrementing `target_name`. Defaults to False.

        Returns:
            str: The path to the generated wav file.
        """
        # call generate with the provided parameters
        buffer = self.generate(sr, length_s, freq, gain_db, window)
        # avoid overwriting if necessary
        if not overwrite:
            target_name = generate_outfilename(target_name)
        # write file
        wav.write(target_name, sr, buffer.astype(np.float32))
        # return path to written file
        return target_name


# %%
# function: image2sines

def image2sines(
    image_path: str,
    target_name: str,
    out_length: float = 4.0,
    num_sines: int = 6,
    sr: int = 44100,
    lowest_freq: float = 50.0,
    highest_freq: float = 10000.0,
    db_range: float = 128,
    time_dim: str = "width",
    harmonic: bool = False,
    normalize: bool = True,
    overwrite: bool = False,
):
    """
    Sonify an image file using a bank of sine wave oscillators and write the result to a file. The function will read an image,
    scale its height or width (depending on `time_dim`) to the number of sine oscillators specified (`num_sines`), and then for
    each row generate a sine wave where pixel luminosity is mapped to loudness. The frequencies of the sine oscillators will be
    an equal-tempered distribution between `lowest_freq` and `highest_freq` if `harmonic==False` or a harmonic series over
    `lowest_freq` until `highest_freq` if `harmonic==True`. Either the width or the height dimension of the image (depending on 
    `time_dim`) will be scaled to the desired length of the output buffer (`out_length`). At each row pixel luminosity will be 
    mapped to the specified decibel range (`db_range`). The function is optimized in a way that each sine oscillator will create 
    a separate worker thread, potentially leveraging machines with a lot of logical processors. All internally used generator 
    functions are JIT-compiled using Numba. To be able to do this, contrary to the Sinetable class, this function is limited 
    to use linear interpolation.

    Args:
        image_path (str): The path to the image to sonify.
        target_name (str): The target name of the rendered wav file. Can end up being different from what was specified if `overwrite=False`.
        out_length (float, optional): The target length of the output wav file in seconds. Defaults to 4.0.
        num_sines (int, optional): The number of sine oscillators to use. Defaults to 6.
        sr (int, optional): The sample rate to use. Defaults to 44100.
        lowest_freq (float, optional): The frequency of the lowest sine oscillator. Defaults to 50.0.
        highest_freq (float, optional): The frequency of the highest sine oscillator. Defaults to 10000.0.
        db_range (float, optional): The decibel range to use when mapping 8-bit pixel luminosity. Defaults to 128.
        time_dim (str, optional): The dimension of the input image to interpret as the time dimension. Can be "width" or "height". If it is "width", then the width of the image will be mapped to `out_length` and the height to `num_sines`. Defaults to "width".
        harmonic (bool, optional): Whether to use a harmonic series as the array of frequencies. If True, the sine frequencies will become integer multiples of `lowest_freq` with frequencies higher than `highest_freq` being discarded. Defaults to False.
        normalize (bool, optional): Whether to normalize the output. Defaults to True.
        overwrite (bool, optional): If False, the method will avoid overwriting existing files by incrementing `target_name`. Defaults to False.
    """
    # read image file
    image_matrix = cv2.imread(image_path)
    # if it's rgb, convert to grayscale
    if len(image_matrix.shape) > 2:
        image_matrix = cv2.cvtColor(image_matrix, cv2.COLOR_BGR2GRAY)
    # if the time dimension if the height, then rotate the image counterclockwise 90 degrees.
    if time_dim.lower() == "height":
        image_matrix = np.transpose(image_matrix)

    # scale height to num_sines
    image_as_spectrogram = cv2.resize(
        image_matrix, (image_matrix.shape[1], num_sines))

    # normalize 8-bit image
    image_norm = image_as_spectrogram.astype(np.float64) / 255.0

    # calculate frequency range
    hz_range = None
    if harmonic:
        harm_series = generate_harmonic_series(num_sines, lowest_freq)
        hz_range = np.array(
            [freq for freq in harm_series if freq <= highest_freq])
    else:
        lowest_midi, highest_midi = librosa.hz_to_midi(
            [lowest_freq, highest_freq])
        midi_range = np.linspace(lowest_midi, highest_midi, num_sines)
        hz_range = librosa.midi_to_hz(midi_range)

    # rotate and mirror image so that time will flow left-to-right, and frequencies will range bottom-to-top
    image_norm = np.transpose(image_norm)
    image_norm = np.fliplr(image_norm)

    # call parallelized generator function
    output_rows = generate_rows_parallel(
        num_sines, image_norm, sr, out_length, hz_range, db_range)

    # rescale after summation
    all_rows = output_rows / num_sines
    # normalize if necessary
    if normalize:
        all_rows = librosa.util.normalize(all_rows)

    # avoid overwriting if necessary
    if not overwrite:
        target_name = generate_outfilename(target_name)
    # write output as 32-bit wave
    wav.write(target_name, sr, all_rows.astype(np.float32))


# %%
# function: generate_harmonic_series

@jit(nopython=True)
def generate_harmonic_series(num_freqs: int, base_freq: float) -> np.ndarray:
    """
    Generate a harmonic series of frequencies.

    Args:
        num_freqs (int): The number of frequencies to generate.
        base_freq (float): The base frequency to use.

    Returns:
        np.ndarray: The generated harmonic series.
    """
    # create output buffer
    out_freqs = np.zeros(num_freqs)
    # for each frequency,
    for i in range(num_freqs):
        # calculate and write to output
        out_freqs[i] = base_freq * (i+1)

    return out_freqs


# %%
# function: fill_indices

@jit(nopython=True)
def fill_indices(
    samples: int,
    index: float,
    increment: float,
    table_length: int
) -> np.ndarray:
    """
    Generate a buffer of sample indices to use with a wavetable oscillator.

    Args:
        samples (int): The length of the indices buffer to generate.
        index (float): The float index in the wavetable to start the generator at.
        increment (float): The index increment to use (will determine frequency of the output wave).
        table_length (int): The length of the internal wavetable in samples.

    Returns:
        np.ndarray: The generated indices buffer.
    """
    # create output buffer
    output = np.zeros(samples)
    # for each sample at input buffer,
    for i in range(samples):
        # write index to output
        output[i] = index
        # apply increment
        index += increment
        # wrap around wavetable if necessary
        index %= table_length-1
    # return indices buffer
    return output


# %%
# function: generate_sine_windowed

@jit(nopython=True)
def generate_sine_windowed(
    sr: int = 44100,
    length_s: float = 1,
    freq: float = 440,
    gain_db: float = 0,
    table_samples: int = 4096
) -> np.ndarray:
    """
    Generate a sine wave buffer based on an internal sine table using linear interpolation.
    Apply a hanning window on the output.

    Args:
        sr (int, optional): The sample rate to use. Defaults to 44100.
        length_s (float, optional): The length of the generated buffer in seconds. Defaults to 1.
        freq (float, optional): The frequency of the sine wave. Defaults to 440.
        gain_db (float, optional): The gain of the sine wave in dB. Defaults to 0.
        table_samples (int, optional): The length of the internal wavetable in samples. Defaults to 4096.

    Returns:
        np.ndarray: The generated sine buffer.
    """
    # generate internal sine table
    x = np.arange(0, table_samples)
    y = np.sin(2 * np.pi * x / np.max(x))
    # generate windowing and sample buffers
    window_buf = np.hanning(int(np.ceil(length_s * sr)))
    # create index variable and increment factor
    index = 0
    increment = freq * table_samples / sr
    # generate indices buffer for sampling
    indices = fill_indices(int(np.ceil(length_s * sr)),
                           index, increment, table_samples)
    # generate linearly interpolated sample buffer
    buffer = np.interp(indices, x, y)
    # apply gain
    amp = 10 ** (gain_db / 20)
    # apply windowing, return buffer
    return buffer * amp * window_buf


# %%
# function: generate_sine

@jit(nopython=True)
def generate_sine(
    sr: int = 44100,
    length_s: float = 1,
    freq: float = 440,
    gain_db: float = 0,
    table_length: int = 4096
) -> np.ndarray:
    """
    Generate a sine wave buffer based on an internal sine table using linear interpolation.

    Args:
        sr (int, optional): The sample rate to use. Defaults to 44100.
        length_s (float, optional): The length of the generated buffer in seconds. Defaults to 1.
        freq (float, optional): The frequency of the sine wave. Defaults to 440.
        gain_db (float, optional): The gain of the sine wave in dB. Defaults to 0.
        table_length (int, optional): The length of the internal wavetable in samples. Defaults to 4096.

    Returns:
        np.ndarray: The generated sine buffer.
    """
    # generate internal sine table
    x = np.arange(0, table_length)
    y = np.sin(2 * np.pi * x / np.max(x))
    # create index variable and increment factor
    index = 0
    increment = freq * table_length / sr
    # generate indices buffer for sampling
    indices = fill_indices(int(np.ceil(length_s * sr)),
                           index, increment, table_length)
    # generate linearly interpolated sample buffer
    buffer = np.interp(indices, x, y)
    # apply gain
    amp = 10 ** (gain_db / 20)
    # apply amp, return buffer
    return buffer * amp


# %%
# function: apply_curve

@jit(nopython=True)
def apply_curve(row: np.ndarray, curve: np.ndarray) -> np.ndarray:
    """
    Apply a curve to a row of samples. The curve is scaled to the length of the row.

    Args:
        row (np.ndarray): The row of samples to apply the curve to.
        curve (np.ndarray): The curve to apply to the row.

    Returns:
        np.ndarray: The row of samples with the curve applied.
    """
    # create x axis for curve
    curve_x = np.arange(0, len(curve))
    # create x axis for row
    row_x = np.arange(0, len(row))
    # scale row x axis to curve x axis
    interp_points = scale_array_auto(row_x, 0, len(curve)-1)
    # return interpolated curve applied to row
    return row * np.interp(interp_points, curve_x, curve)


# %%
# function: generate_row

@jit(nopython=True)
def generate_row(
    row_samples: np.ndarray,
    sr: int,
    row_length: float,
    frequency: float,
    db_range: float
) -> np.ndarray:
    """
    Generate a sine "row" buffer (a sequence of overlapped sines of varying amplitudes) of a given frequency and length.

    Args:
        num_hops (int): Number of hops (the number of sines that will be generated).
        num_cols (int): Number of columns (the number of cells in row_samples).
        hop_size_samps (int): The hop size in samples.
        row_samples (np.ndarray): The 1D row to synthesize as sines.
        sr (int): The sample rate to use.
        sine_length (float): The length of the internal sine wave (in seconds) at each hop.
        frequency (float): The frequency of the sine wave.
        db_range (float): The decibel range to map row_samples to.

    Returns:
        np.ndarray: The generated sine buffer.
    """
    # generate sine buffer
    sine_buffer = generate_sine(sr, row_length, frequency, 0, 4096)
    # map row_samples to decibel range
    row_db = scale_array(row_samples, 0, 1, -db_range, 0)
    # convert db to amp
    row_amp = np.power(10, row_db / 20)
    # apply curve to sine buffer and return
    return apply_curve(sine_buffer, row_amp)


# %%
# function: generate_rows_parallel

@njit(parallel=True)
def generate_rows_parallel(
    num_rows: int,
    matrix: np.ndarray,
    sr: int,
    row_length: float,
    hz_range: np.ndarray,
    db_range: float
) -> np.ndarray:
    """
    Generate sine rows from a 2D matrix. The process is parallelized, to take advantage of CPU-s with multiple cores.
    For every row in the `matrix` (for `num_rows`) call `generate_row` in a parallel thread. Then accumulate rows
    in an output row that is returned at the end of the process.

    Args:
        num_rows (int): Number or rows to be generated.
        matrix (np.ndarray): The 2D matrix to generate sine rows from.
        num_hops (int): Number of hops (the number of sines that will be generated in each row).
        num_cols (int): Number of columns (the number of cells in one row of the matrix).
        hop_size_samps (int): The hop size in samples.
        sr (int): The sample rate to use.
        sine_length (float): The length of the internal sine wave (in seconds) at each hop in each row.
        hz_range (np.ndarray): The array of frequencies to use for generating the rows.
        db_range (float): The decibel range to map each row to.

    Returns:
        np.ndarray: The buffer accumulated from all rows.
    """
    # create container
    output_rows = np.zeros(int(np.ceil(row_length * sr)), dtype=np.float64)
    # for each row,
    for row in prange(num_rows):
        # take the row from the matrix
        y = matrix[:, row]
        # if row is empty, just skip
        if np.sum(y) == 0:
            continue
        # otherwise generate the row
        output_row = generate_row(
            y, sr, row_length, hz_range[row], db_range)
        # accumulate result to output
        output_rows += output_row
    # return accumulated result
    return output_rows


# %%
# function: overlap_add

@jit(nopython=True)
def overlap_add(a1: np.ndarray, a2: np.ndarray, insertion_index: int) -> np.ndarray:
    """
    Implement the overlap-add technique. Concatenate the array `a2` with `a1` at `insertion_index`.
    The overlapping elements will be added together. Return the concatenated array.

    Args:
        a1 (np.ndarray): The "left" array.
        a2 (np.ndarray): The "right" array.
        insertion_index (int): The index in `a1` where `a2` should start.

    Returns:
        np.ndarray: The concatenated array.
    """
    # determine overlap length
    a1_overlap_length = 0
    if insertion_index >= 0:
        a1_overlap_length = len(a1) - insertion_index
    else:
        a1_overlap_length = insertion_index * -1

    # if in fact there is an overlap,
    if a1_overlap_length > 0:
        # take a1 until the overlapping area
        a1_before_overlap = a1[:insertion_index]
        a1_to_overlap = a1[insertion_index:]
        a2_to_overlap = a2[:a1_overlap_length]
        # accumulate overlapping area
        overlapped = a1_to_overlap + a2_to_overlap
        # take a2 after the overlapping area
        a2_after_overlap = a2[a1_overlap_length:]
        # concatenate all three
        return np.concatenate((a1_before_overlap, overlapped, a2_after_overlap))
    # if there is no overlapping area, just concatenate a1 and a2
    else:
        return np.concatenate((a1, a2))


# %%
# function: scale_array_auto

@jit(nopython=True)
def scale_array_auto(
    array: np.ndarray,
    out_low: float,
    out_high: float
) -> np.ndarray:
    """
    Scales an array linearly. The input range is automatically 
    retrieved from the array. Optimized by Numba.

    Args:
        array (np.ndarray): The array to be scaled.
        out_low (float): Minimum of output range.
        out_high (float): Maximum of output range.

    Returns:
        np.ndarray: The scaled array.
    """
    minimum, maximum = np.min(array), np.max(array)
    # if all values are the same, then return an array with the
    # same shape, all cells set to out_high
    if maximum - minimum == 0:
        return np.ones_like(array, dtype=np.float64) * out_high
    else:
        m = (out_high - out_low) / (maximum - minimum)
        b = out_low - m * minimum
        return m * array + b


# %%
# function: scale_array

@jit(nopython=True)
def scale_array(
    array: np.ndarray,
    in_low: float,
    in_high: float,
    out_low: float,
    out_high: float
) -> np.ndarray:
    """
    Scales an array linearly.

    Args:
        array (np.ndarray): The array to be scaled.
        in_low (float): Minimum of input range.
        in_high (float): Maximum of input range.
        out_low (float): Minimum of output range.
        out_high (float): Maximum of output range.

    Returns:
        np.ndarray: The scaled array.
    """
    if in_high == in_low:
        return np.ones_like(array, dtype=np.float64) * out_high
    else:
        return ((array - in_low) * (out_high - out_low)) / (in_high - in_low) + out_low

# %%

# function scale_array_exp


@jit(nopython=True)
def scale_array_exp(
    x: np.ndarray,
    in_low: float,
    in_high: float,
    out_low: float,
    out_high: float,
    exp: float = 1.0,
) -> np.ndarray:
    """
    Scales an array of values from one range to another. Based on the Max/MSP scale~ object.

    Args:
        x (np.ndarray): The array of values to scale.
        in_low (float): The lower bound of the input range.
        in_high (float): The upper bound of the input range.
        out_low (float): The lower bound of the output range.
        out_high (float): The upper bound of the output range.
        exp (float, optional): The exponent to use for the scaling. Defaults to 1.0.

    Returns:
        np.ndarray: The scaled array.
    """
    if in_high == in_low:
        return np.ones_like(x, dtype=np.float64) * out_high
    else:
        return np.where(
            (x-in_low)/(in_high-in_low) == 0,
            out_low,
            np.where(
                (x-in_low)/(in_high-in_low) > 0,
                out_low + (out_high-out_low) *
                ((x-in_low)/(in_high-in_low))**exp,
                out_low + (out_high-out_low) * -
                ((((-x+in_low)/(in_high-in_low)))**(exp))
            )
        )

# %%

# function: wrap


@jit(nopython=True)
def wrap(
    x: float,
    min: float,
    max: float,
) -> float:
    """
    Wrap a value between a minimum and maximum value.

    Args:
        x (float): The value to wrap.
        min (float): The minimum value.
        max (float): The maximum value.

    Returns:
        float: The wrapped value.
    """
    return (x - min) % (max - min) + min

# %%

# function: resize_interp


@jit(nopython=True)
def resize_interp(
    input: np.ndarray,
    size: int,
) -> np.ndarray:
    """
    Resize an array. Uses linear interpolation.

    Args:
        input (np.ndarray): Array to resize.
        size (int): The new size of the array.

    Returns:
        np.ndarray: The resized array.
    """
    # create x axis for input
    input_x = np.arange(0, len(input))
    # create array with sampling indices
    output_x = scale_array_auto(np.arange(size), 0, len(input_x)-1)
    # interpolate
    return np.interp(output_x, input_x, input).astype(np.float64)

# %%

# function: phasor


@jit(nopython=True)
def phasor(
    samples: int,
    sr: int,
    frequency: np.ndarray,
) -> np.ndarray:
    """
    Generate a phasor.

    Args:
        samples (int): The number of samples to generate.
        sr (int): The sample rate to use.
        frequency (np.ndarray): The frequency to use. Can be a single value or an array.

    Returns:
        np.ndarray: The generated phasor.
    """
    # create array to hold output
    output = np.zeros(samples, dtype=np.float64)
    frequency_resized = np.array([0], dtype=np.float64)
    if len(frequency) == 1:
        frequency_resized = np.repeat(frequency[0], samples).astype(np.float64)
    elif len(frequency) == samples:
        frequency_resized = frequency.astype(np.float64)
    else:
        # resize frequency array to match number of samples (-1 because we start at 0)
        frequency_resized = resize_interp(frequency, samples-1)
    # for each sample after the first
    for i in range(samples-1):
        # calculate increment
        increment = frequency_resized[i] / sr
        # calculate phasor value from last sample and increment
        output[i+1] = wrap(increment + output[i], 0, 1)
    return output

# %%

# function: samples2seconds


@jit(nopython=True)
def samples2seconds(
    samples: int,
    sr: int,
) -> float:
    """
    Convert samples to seconds.

    Args:
        samples (int): The number of samples.
        sr (int): The sample rate.

    Returns:
        float: The number of seconds.
    """
    return samples / sr

# %%

# function: seconds2samples


@jit(nopython=True)
def seconds2samples(
    seconds: float,
    sr: int,
) -> int:
    """
    Convert seconds to samples.

    Args:
        seconds (float): The number of seconds.
        sr (int): The sample rate.

    Returns:
        int: The number of samples.
    """
    return int(seconds * sr)

# %%

# function: array2broadcastable


@jit(nopython=True)
def array2broadcastable(
    array: np.ndarray,
    samples: int
) -> np.ndarray:
    """
    Convert an array to a broadcastable array. If the array has a single value or has
    the size == samples, the array is returned. Otherwise the array is resized with 
    linear interpolation (calling resize_interp) to match the number of samples.

    Args:
        array (np.ndarray): The array to convert.
        samples (int): The number of samples to generate.

    Returns:
        np.ndarray: The converted array.
    """
    if array.size == 1 or array.size == samples:
        return array
    else:
        return resize_interp(array, samples)

# %%

# function: sinewave


@jit(nopython=True)
def sinewave(
    samples: int,
    sr: int,
    frequency: np.ndarray,
) -> np.ndarray:
    """
    Generate a sine wave.

    Args:
        samples (int): The number of samples to generate.
        sr (int): The sample rate to use.
        frequency (np.ndarray): The frequency to use. Can be a single value or an array.

    Returns:
        np.ndarray: The generated sine wave.
    """
    # create phasor buffer
    phasor_buf = phasor(samples, sr, frequency)
    # calculate sine wave and return sine buffer
    return np.sin(2 * np.pi * phasor_buf)

# %%

# function: fm_synth


def fm_synth(
        samples: int,
        sr: int,
        carrier_frequency: np.ndarray,
        modulator_frequency: np.ndarray,
        modulator_amplitude: np.ndarray,
) -> np.ndarray:
    """
    Generate a frequency modulated signal.

    Args:
        samples (int): The number of samples to generate.
        sr (int): The sample rate to use.
        carrier_frequency (np.ndarray): The carrier frequency to use. Can be a single value or an array.
        modulator_frequency (np.ndarray): The modulator frequency to use. Can be a single value or an array.
        modulator_amplitude (np.ndarray): The modulator amplitude to use. Can be a single value or an array.

    Returns:
        np.ndarray: The generated frequency modulated signal.
    """
    # create modulator buffer
    modulator_buf = sinewave(samples, sr, modulator_frequency)
    # if modulator amplitude is a single value, multiply modulator buffer by that value
    if len(modulator_amplitude) == 1:
        modulator_buf *= modulator_amplitude[0]
    # if modulator amplitude is an array, resize it to match number of samples and multiply modulator buffer by it
    else:
        modulator_buf *= resize_interp(
            modulator_amplitude.astype(np.float64), samples)
    # calculate frequency modulated signal and return fm buffer
    return sinewave(samples, sr, carrier_frequency + modulator_buf)

# %%

# function: fm_synth_2 (with harmonicity)


def fm_synth_2(
        samples: int,
        sr: int,
        carrier_frequency: np.ndarray,
        harmonicity_ratio: np.ndarray,
        modulation_index: np.ndarray,
) -> np.ndarray:
    """
    Generate a frequency modulated signal.

    Args:
        samples (int): The number of samples to generate.
        sr (int): The sample rate to use.
        carrier_frequency (np.ndarray): The carrier frequency to use. Can be a single value or an array.
        harmonicity_ratio (np.ndarray): The harmonicity ratio to use. Can be a single value or an array.
        modulation_index (np.ndarray): The modulation index to use. Can be a single value or an array.

    Returns:
        np.ndarray: The generated frequency modulated signal.
    """
    # initialize parameter arrays
    _carrier_frequency = array2broadcastable(
        carrier_frequency.astype(np.float64), samples)
    _harmonicity_ratio = array2broadcastable(
        harmonicity_ratio.astype(np.float64), samples)
    _modulation_index = array2broadcastable(
        modulation_index.astype(np.float64), samples)

    # calculate modulator frequency
    modulator_frequency = _carrier_frequency * _harmonicity_ratio
    # create modulator buffer
    modulator_buf = sinewave(samples, sr, modulator_frequency)
    # create modulation amplitude buffer
    modulation_amplitude = modulator_frequency * _modulation_index
    # calculate frequency modulated signal and return fm buffer
    return sinewave(samples, sr, _carrier_frequency + (modulator_buf * modulation_amplitude))

# %%

# function: am_synth


def am_synth(
        samples: int,
        sr: int,
        carrier_frequency: np.ndarray,
        modulator_frequency: np.ndarray,
        modulator_amplitude: np.ndarray,
) -> np.ndarray:
    """
    Generate an amplitude modulated signal.

    Args:
        samples (int): The number of samples to generate.
        sr (int): The sample rate to use.
        carrier_frequency (np.ndarray): The carrier frequency to use. Can be a single value or an array.
        modulator_frequency (np.ndarray): The modulator frequency to use. Can be a single value or an array.
        modulator_amplitude (np.ndarray): The modulator amplitude to use. Can be a single value or an array.

    Returns:
        np.ndarray: The generated amplitude modulated signal.
    """
    # create modulator buffer
    modulator_buf = sinewave(samples, sr, modulator_frequency)
    mod_amp_resized = np.zeros(1, dtype=np.float64)
    # if modulator amplitude is a single value, multiply modulator buffer by that value
    if len(modulator_amplitude) == 1:
        mod_amp_resized = modulator_amplitude[0]
        modulator_buf *= modulator_amplitude[0]
    # if modulator amplitude is an array, resize it to match number of samples and multiply modulator buffer by it
    else:
        mod_amp_resized = resize_interp(
            modulator_amplitude.astype(np.float64), samples)
        modulator_buf *= mod_amp_resized
    # calculate amplitude modulated signal and return am buffer
    return sinewave(samples, sr, carrier_frequency) * (modulator_buf + 1 - mod_amp_resized)

# %%

# function: am_module


def am_module(
        samples: int,
        sr: int,
        modulator_frequency: np.ndarray,
        modulator_amplitude: np.ndarray,
) -> np.ndarray:
    """
    Generate an amplitude modulator signal.

    Args:
        samples (int): The number of samples to generate.
        sr (int): The sample rate to use.
        modulator_frequency (np.ndarray): The modulator frequency to use. Can be a single value or an array.
        modulator_amplitude (np.ndarray): The modulator amplitude to use. Can be a single value or an array.

    Returns:
        np.ndarray: The generated amplitude modulator signal.
    """
    # create modulator buffer
    modulator_buf = sinewave(samples, sr, modulator_frequency)
    mod_amp_resized = np.zeros(1, dtype=np.float64)
    # if modulator amplitude is a single value, multiply modulator buffer by that value
    if len(modulator_amplitude) == 1:
        mod_amp_resized = modulator_amplitude[0]
        modulator_buf *= modulator_amplitude[0]
    # if modulator amplitude is an array, resize it to match number of samples and multiply modulator buffer by it
    else:
        mod_amp_resized = resize_interp(
            modulator_amplitude.astype(np.float64), samples)
        modulator_buf *= mod_amp_resized
    # calculate amplitude modulator signal and return am buffer
    return modulator_buf + 1 - mod_amp_resized

# %%

# function: midi2frequency


@jit(nopython=True)
def midi2frequency(
        midi: np.ndarray,
        base_frequency: float = 440.0,
) -> np.ndarray:
    """
    Convert MIDI note number to frequency.

    Args:
        midi (np.ndarray): The MIDI note number. Can be a scalar or an array.
        base_frequency (float, optional): The base frequency (or "tuning") to use. Defaults to 440.0.

    Returns:
        np.ndarray: The frequency in Hz.
    """
    return base_frequency * 2 ** ((midi.astype(np.float64) - 69) / 12)

# %%

# function: frequency2midi


@jit(nopython=True)
def frequency2midi(
        frequency: np.ndarray,
        base_frequency: float = 440.0,
) -> np.ndarray:
    """
    Converts a frequency in Hz to a MIDI note number.

    Args:
        frequency: Frequency in Hz. Can be a scalar or a numpy array.
        base_frequency: Frequency of MIDI note 69. Defaults to 440.0.

    Returns:
        np.ndarray: MIDI note number.
    """

    return 69 + 12 * np.log2(frequency.astype(np.float64) / base_frequency)

# %%

# function: history


@jit(nopython=True)
def history(
        signal: np.ndarray,
) -> np.ndarray:
    """
    History signal. Shifts the input array by one sample to the right.

    Args:
        signal (np.ndarray): A signal.

    Returns:
        np.ndarray: A history signal.
    """
    # make history array
    history = np.zeros_like(signal, dtype=np.float64)
    history[1:] = signal[:-1]
    return history

# %%


# function: ramp2trigger

def ramp2trigger(
        ramp: np.ndarray,
) -> np.ndarray:
    """
    Convert a ramp to a trigger signal.

    Args:
        ramp (np.ndarray): A ramp signal.

    Returns:
        np.ndarray: A trigger signal.
    """
    # make output array
    trigger = np.zeros_like(ramp)
    # make history array
    history_sig = history(ramp)
    # calculate absolute proportional change
    abs_proportional_change = np.abs(np.divide(
        (ramp - history_sig), (ramp + history_sig), out=trigger, where=(ramp + history_sig) != 0))
    # convert to trigger
    trigger[abs_proportional_change > 0.5] = 1
    # remove duplicates
    trigger[1:] = np.diff(trigger)
    trigger = np.where(trigger > 0, 1, 0)

    return trigger

# %%

# function: ramp2slope


def ramp2slope(ramp: np.ndarray) -> np.ndarray:
    """
    Converts a ramp (0...1) to a slope (-0.5...0.5).

    Args:
        ramp (np.ndarray): ramp (0...1)

    Returns:
        np.ndarray: slope (-0.5...0.5)
    """
    delta = np.zeros_like(ramp)
    delta[1:] = np.diff(ramp)
    return wrap(delta, -0.5, 0.5)

# %%

# function: array2fluid_dataset


def array2fluid_dataset(
        array: np.ndarray,
) -> dict:
    """
    Convert a numpy array to a json format that's compatible with fluid.dataset~.

    Args:
        array (np.ndarray): The numpy array to convert. Should be a 2D array of (num_samples, num_features).

    Returns:
        dict: The json dataset.
    """
    num_cols = array.shape[1]
    out_dict = {}
    out_dict["cols"] = num_cols
    out_dict["data"] = {}
    for i in range(len(array)):
        out_dict["data"][str(i)] = array[i].tolist()
    return out_dict


# %%

# function: fluid_dataset2array


def fluid_dataset2array(
        dataset: dict,
) -> np.ndarray:
    """
    Convert a json dataset to a numpy array.

    Args:
        dataset (dict): The json dataset to convert.

    Returns:
        np.ndarray: The numpy array.
    """
    num_cols = dataset["cols"]
    num_rows = len(dataset["data"])
    out_array = np.zeros((num_rows, num_cols))
    for i in range(num_rows):
        out_array[i] = np.array(dataset["data"][str(i)])
    return out_array
