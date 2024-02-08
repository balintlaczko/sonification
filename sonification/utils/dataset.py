import os
from typing import List
import numpy as np
import pandas as pd
import cv2
import matrix
from misc import roundup, generate_outfilename
import tqdm


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
            image = matrix.stretch_contrast(
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
            image = matrix.stretch_contrast(
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
        pb = tqdm.tqdm(range(self.num_steps), desc=pb_prefix)
        # create a preview window for the render
        cv2.namedWindow("Rendering in progress...", 0)
        cv2.resizeWindow("Rendering in progress...", (300, 300))
        # loop through steps
        for step in pb:
            # fetch image path and then matrix
            image_path = self.get_path(step, color_space, visiting_point)
            image = self.get_matrix(image_path, color_space)
            # normalize if necessary
            if normalize:
                color_min, color_max = self.minmax[color_space.lower()]
                image = matrix.stretch_contrast(
                    image, in_min=color_min, in_max=color_max, out_max=255, in_percentile=norm_percentile)
            # write frame as 8-bit
            out.write(image.astype(np.uint8))
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