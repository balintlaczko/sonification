import numpy as np
import cv2
import os
import tqdm
from typing import List
from .misc import generate_outfilename, str2sec


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


def video_from_images(images: List[str], images_folder: str, destination_folder: str = None, target_name: str = None, overwrite: bool = False) -> str:
    """
    Render a video from a folder of images.

    Args:
        images (List[str]): File names of the images.
        images_folder (str): Path to the folder containing the images.
        destination_folder (str, optional): Folder to put the rendered video into. If None, the images_folder will be used. Defaults to None.
        target_name (str, optional): Requested name of the video. Defaults to None.
        overwrite (bool, optional): Allow to overwrite exisiting files. If False, the target_name will be incremented to avoid overwriting an older file. Defaults to False.

    Returns:
        str: Path to the rendered video.
    """
    if destination_folder == None:
        destination_folder = images_folder

    if target_name == None:
        target_name = os.path.join(
            destination_folder, f'{os.path.basename(os.path.dirname(images_folder))}.avi')
    if not overwrite:
        target_name = generate_outfilename(target_name)

    concat = 'concat:'

    for image in images:
        concat += f'{os.path.join(images_folder, image)}|'

    cmd = ['ffmpeg', '-i', concat, '-c', 'mjpeg', '-q:v', '3', target_name]

    ffmpeg_cmd(cmd, len(images),
               pb_prefix="Rendering video from images:")

    return target_name


class FFmpegError(Exception):
    def __init__(self, message):
        self.message = message


def ffmpeg_cmd(command, total_time, pb_prefix='Progress', print_cmd=False, stream=True, pipe=None):
    """
    Run an ffmpeg command in a subprocess and show progress using an MgProgressbar.

    Args:
        command (list): The ffmpeg command to execute as a list. Eg. ['ffmpeg', '-y', '-i', 'myVid.mp4', 'myVid.mov']
        total_time (float): The length of the output. Needed mainly for the progress bar.
        pb_prefix (str, optional): The prefix for the progress bar. Defaults to 'Progress'.
        print_cmd (bool, optional): Whether to print the full ffmpeg command to the console before executing it. Good for debugging. Defaults to False.
        stream (bool, optional): Whether to have a continuous output stream or just (the last) one. Defaults to True (continuous stream).
        pipe (str, optional): Whether to pipe video frames from FFmpeg to numpy array. Possible to read the video frame by frame with pipe='read', to load video in memory with pipe='load', or to write the frames of a numpy array to a video file with pipe='write'. Defaults to None.

    Raises:
        KeyboardInterrupt: If the user stops the process.
        FFmpegError: If the ffmpeg process was unsuccessful.
    """
    import subprocess

    # pb = MgProgressbar(total=total_time, prefix=pb_prefix)

    # Hide banner and quiet report printing
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'quiet'] + command[1:]

    if print_cmd:
        if type(command) == list:
            print(' '.join(command))
        else:
            print(command)

    if pipe == 'read':
        # Define ffmpeg command and read frame by frame
        command = command + ['-f', 'image2pipe', '-pix_fmt', 'bgr24',
                             '-vcodec', 'rawvideo', '-preset', 'ultrafast', '-']
        process = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=-1)
        return process

    elif pipe == 'load':
        # Define ffmpeg command and load all video frames in memory
        command = command + ['-f', 'image2pipe', '-pix_fmt', 'bgr24',
                             '-vcodec', 'rawvideo', '-preset', 'ultrafast', '-']
        process = subprocess.run(command, stdout=subprocess.PIPE, bufsize=-1)
        return process

    elif pipe == 'write':
        # Write the frames of a numpy array to a video file
        process = subprocess.Popen(command, stdin=subprocess.PIPE, bufsize=-1)
        return process

    else:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        returncode = None
        all_out = ''

        try:
            while True:
                if stream:
                    out = process.stdout.readline()
                else:
                    out = process.stdout.read()
                all_out += out

                if out == '':
                    process.wait()
                    returncode = process.returncode
                    break

                elif out.startswith('frame='):
                    # try:
                    #     out_list = out.split()
                    #     time_ind = [elem.startswith('time=')
                    #                 for elem in out_list].index(True)
                    #     time_str = out_list[time_ind][5:]
                    #     time_sec = str2sec(time_str)
                    #     pb.progress(time_sec)
                    # except ValueError:
                    #     # New version of FFmpeg outputs N/A values
                    #     pass
                    pass

            if returncode in [None, 0]:
                # pb.progress(total_time)
                pass
            else:
                raise FFmpegError(all_out)

        except KeyboardInterrupt:
            try:
                process.terminate()
            except OSError:
                pass
            process.wait()
            raise KeyboardInterrupt
