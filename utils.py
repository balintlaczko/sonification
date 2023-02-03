import scipy.io.wavfile as wav
from scipy import interpolate
import os
import pandas as pd
import numpy as np
import cv2
import librosa
# import pyo
from numba import jit, njit, prange
from musicalgestures._utils import roundup, generate_outfilename, MgProgressbar
from typing import List


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
    output_rows = generate_rows_fast(
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


@jit(nopython=True)
def generate_harmonic_series(num_freqs, base_freq):
    out_freqs = np.zeros(num_freqs)
    for i in range(num_freqs):
        out_freqs[i] = base_freq * (i+1)
    return out_freqs


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


@jit(nopython=True)
def generate_sine(
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


@jit(nopython=True)
def generate_sine_no_window(
    sr: int = 44100,
    length_s: float = 1,
    freq: float = 440,
    gain_db: float = 0,
    table_samples: int = 4096
) -> np.ndarray:
    """
    Generate a sine wave buffer based on an internal sine table using linear interpolation.

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
    # apply amp, return buffer
    return buffer * amp


# apply curve
@jit(nopython=True)
def apply_curve(row, curve):
    curve_x = np.arange(0, len(curve))
    interp_points = scale_array(np.arange(0, len(row)), 0, len(curve)-1)
    return row * np.interp(interp_points, curve_x, curve)


@jit(nopython=True)
def generate_row(
    num_hops: int,
    num_cols: int,
    hop_size_samps: int,
    row_samples: np.ndarray,
    sr: int,
    sine_length: float,
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
    # the indices for row_samples
    x = np.arange(0, row_samples.shape[0])
    # the insertion index for the overlap-add
    insertion_index = 0
    # translate hops to interpolated columns
    hops_as_cols = scale_array(np.arange(num_hops), 0, num_cols-1)
    # generate sine only once
    sine_buffer = generate_sine_no_window(sr, sine_length, frequency, 0, 4096)
    # for each hop,
    for hop in range(num_hops):
        # get linearly interpolated cell value from input
        cell_val = np.interp(hops_as_cols[hop], x, row_samples)
        # calculate gain
        gain_db_scaled = cell_val * db_range - db_range
        # generate sine buffer for hop
        # buffer = generate_sine(
        #     sr, sine_length, frequency, gain_db_scaled, 4096)
        amp = 10 ** (gain_db_scaled / 20)
        # if this is the first hop, then just add, else overlap-add
        if hop == 0:
            # output_row = buffer
            output_row = sine_buffer * amp
        else:
            # output_row = overlap_add(output_row, buffer, insertion_index)
            output_row = overlap_add(
                output_row, sine_buffer * amp, insertion_index)
        # increment insertion_index for next hop
        insertion_index += hop_size_samps
    # return row buffer
    return output_row


@jit(nopython=True)
def generate_row_fast(
    row_samples: np.ndarray,
    sr: int,
    row_length: float,
    frequency: float,
    db_range: float
) -> np.ndarray:

    sine_buffer = generate_sine_no_window(sr, row_length, frequency, 0, 4096)
    # row_db = scale_array(row_samples, -db_range, 0)
    row_db = scale_array_custom(row_samples, 0, 1, -db_range, 0)
    row_amp = np.power(10, row_db / 20)
    return apply_curve(sine_buffer, row_amp)


# generate rows / parallel version
@njit(parallel=True)
def generate_rows_fast(
    num_rows: int,
    matrix: np.ndarray,
    sr: int,
    row_length: float,
    hz_range: np.ndarray,
    db_range: float
) -> np.ndarray:
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
        output_row = generate_row_fast(
            y, sr, row_length, hz_range[row], db_range)
        # accumulate result to output
        output_rows += output_row
    # return accumulated result
    return output_rows


# generate rows / parallel version
@njit(parallel=True)
def generate_rows(
    num_rows: int,
    matrix: np.ndarray,
    num_hops: int,
    num_cols: int,
    hop_size_samps: int,
    sr: int,
    sine_length: float,
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
    output_rows = np.zeros(
        int(np.ceil(sine_length * sr) + (num_hops-1) * hop_size_samps), dtype=np.float64)
    # for each row,
    for row in prange(num_rows):
        # take the row from the matrix
        y = matrix[:, row]
        # if row is empty, just skip
        if np.sum(y) == 0:
            continue
        # otherwise generate the row
        output_row = generate_row(
            num_hops, num_cols, hop_size_samps, y, sr, sine_length, hz_range[row], db_range)
        # accumulate result to output
        output_rows += output_row
    # return accumulated result
    return output_rows


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


@jit(nopython=True)
def scale_array(
    array: np.ndarray,
    out_low: float,
    out_high: float
) -> np.ndarray:
    """
    Scales an array linearly. Optimized by Numba.

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


@jit(nopython=True)
def scale_array_custom(
    array: np.ndarray,
    in_low: float,
    in_high: float,
    out_low: float,
    out_high: float
) -> np.ndarray:
    """
    Scales an array linearly using a custom input range 
    (ignoring the arrays own range). Optimized by Numba.

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


