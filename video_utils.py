# %%
# imports

import numpy as np
import cv2
import os
import tqdm

# %%
# function to render an RGB video from up to 3 grayscale videos


def videos2planes(
        video2red: str = None,
        video2green: str = None,
        video2blue: str = None,
        target_name: str = None,
) -> str:
    """
    Render an RGB video from up to 3 grayscale videos, where
    the first video is the red channel, the second is the green,
    and the third is the blue. If an input video is not provided,
    that channel will be rendered as black.

    Args:
        video2red (str, optional): Path to the video to use for the red channel. Defaults to None.
        video2green (str, optional): Path to the video to use for the green channel. Defaults to None.
        video2blue (str, optional): Path to the video to use for the blue channel. Defaults to None.
        target_name (str, optional): The name of the output video. Defaults to None (which assumes that the filename "out_rgb.avi" should be used).

    Returns:
        str: Path to the rendered RGB video.
    """

    # dictionary of videos
    videos = {}

    # read video paths as grayscale
    colors = ["red", "green", "blue"]
    video_paths = {
        "red": video2red,
        "green": video2green,
        "blue": video2blue
    }

    for color in colors:
        video_path = video_paths[color]
        if video_path is not None:
            videos[color] = cv2.VideoCapture(video_path)
            assert videos[color].isOpened(
            ), f"Error opening video: {video_path}"

    # get video properties of first provided input video
    folder, fps, width, height, length = None, None, None, None, None
    for color in colors:
        if color in videos.keys():
            folder = os.path.dirname(video_paths[color])
            fps = videos[color].get(cv2.CAP_PROP_FPS)
            width = int(videos[color].get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(videos[color].get(cv2.CAP_PROP_FRAME_HEIGHT))
            length = int(videos[color].get(cv2.CAP_PROP_FRAME_COUNT))
            break
    print(f"Folder: {folder}")
    print(f"FPS: {fps}")
    print(f"Width: {width}")
    print(f"Height: {height}")
    print(f"Length: {length}")

    # create output video path
    output_name = None
    if target_name is None:
        output_name = os.path.join(folder, "out_rgb.avi")
    else:
        target_folder = os.path.dirname(target_name)
        if target_folder == "":
            target_folder = folder
        target_filename = os.path.basename(target_name)
        # enforce .avi extension
        if not target_filename.endswith(".avi"):
            target_filename = os.path.splitext(target_filename)[0] + ".avi"
        output_name = os.path.join(target_folder, target_filename)

    # create output video
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_name, fourcc, fps, (width, height))

    # initialize frame counter
    frame_counter = 0

    # iterate over frames
    for _ in tqdm.tqdm(range(length)):
        # read frames
        frames = {}
        for color in colors:
            if color in videos.keys():
                success, frame = videos[color].read()
                assert success, f"Error reading frame {frame_counter} from video: {video_paths[color]}"
                assert frame.shape[0] == height and frame.shape[
                    1] == width, f"Frame {frame_counter} from video {video_paths[color]} has incorrect dimensions: {frame.shape}"
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames[color] = frame
            else:
                frames[color] = np.zeros((height, width), dtype=np.uint8)

        # merge frames
        frame = np.stack(
            [frames["blue"], frames["green"], frames["red"]], axis=-1)
        assert frame.shape == (height, width, 3)
        # write frame
        out.write(frame)

        # increment frame counter
        frame_counter += 1

    # release videos
    for color in colors:
        if color in videos.keys():
            videos[color].release()
    out.release()

    return output_name


# %%
