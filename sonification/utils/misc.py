import os
import numpy as np


def scale_linear(x, in_low, in_high, out_low, out_high):
    return (x - in_low) / (in_high - in_low) * (out_high - out_low) + out_low


def midi2frequency(midi, base_frequency=440.0):
    return base_frequency * 2**((midi - 69) / 12)


def frequency2midi(frequency, base_frequency=440.0):
    return 69 + 12 * np.log2(frequency / base_frequency)


# taken from musicalgestures
def roundup(num, modulo_num):
    """
    Rounds up a number to the next integer multiple of another.

    Args:
        num (int): The number to round up.
        modulo_num (int): The number whose next integer multiple we want.

    Returns:
        int: The rounded-up number.
    """
    num, modulo_num = int(num), int(modulo_num)
    return num - (num % modulo_num) + modulo_num*((num % modulo_num) != 0)


# taken from musicalgestures
def generate_outfilename(requested_name):
    """Returns a unique filepath to avoid overwriting existing files. Increments requested 
    filename if necessary by appending an integer, like "_0" or "_1", etc to the file name.

    Args:
        requested_name (str): Requested file name as path string.

    Returns:
        str: If file at requested_name is not present, then requested_name, else an incremented filename.
    """
    requested_name = os.path.abspath(requested_name).replace('\\', '/')
    req_of, req_fex = os.path.splitext(requested_name)
    req_of = req_of.replace('\\', '/')
    req_folder = os.path.dirname(requested_name).replace('\\', '/')
    req_of_base = os.path.basename(req_of)
    req_file_base = os.path.basename(requested_name)
    out_increment = 0
    files_in_folder = os.listdir(req_folder)
    # if the target folder is empty, return the requested path
    if len(files_in_folder) == 0:
        return requested_name
    # filter files with same ext
    files_w_same_ext = list(filter(lambda x: os.path.splitext(x)[
                            1] == req_fex, files_in_folder))
    # if there are no files with the same ext
    if len(files_w_same_ext) == 0:
        return requested_name
    # filter for files with same start and ext
    files_w_same_start_ext = list(
        filter(lambda x: x.startswith(req_of_base), files_w_same_ext))
    # if there are no files with the same start and ext
    if len(files_w_same_start_ext) == 0:
        return requested_name
    # check if requested file is already present
    present = None
    try:
        ind = files_w_same_start_ext.index(req_file_base)
        present = True
    except ValueError:
        present = False
    # if requested file is not present
    if not present:
        return requested_name
    # if the original filename is already taken, check if there are incremented filenames
    files_w_increment = list(filter(lambda x: x.startswith(
        req_of_base+"_"), files_w_same_start_ext))
    # if there are no files with increments
    if len(files_w_increment) == 0:
        return f'{req_of}_0{req_fex}'
    # parse increments, discard the ones that are invalid, increment highest
    for file in files_w_increment:
        _of = os.path.splitext(file)[0]
        _only_incr = _of[len(req_of_base)+1:]
        try:
            found_incr = int(_only_incr)
            found_incr = max(0, found_incr)  # clip at 0
            out_increment = max(out_increment, found_incr+1)
        except ValueError:  # if cannot be converted to int
            pass
    # return incremented filename
    return f'{req_of}_{out_increment}{req_fex}'


# taken from: https://www.geeksforgeeks.org/python-program-for-quicksort/
# Function to find the partition position
def partition(array, low, high):

    # choose the rightmost element as pivot
    pivot = array[high]

    # pointer for greater element
    i = low - 1

    # traverse through all elements
    # compare each element with pivot
    for j in range(low, high):
        if array[j] <= pivot:

            # If element smaller than pivot is found
            # swap it with the greater element pointed by i
            i = i + 1

            # Swapping element at i with element at j
            (array[i], array[j]) = (array[j], array[i])

    # Swap the pivot element with the greater element specified by i
    (array[i + 1], array[high]) = (array[high], array[i + 1])

    # Return the position from where partition is done
    return i + 1

# function to perform quicksort


def quickSort(array, low, high):
    if low < high:

        # Find pivot element such that
        # element smaller than pivot are on the left
        # element greater than pivot are on the right
        pi = partition(array, low, high)

        # Recursive call on the left of pivot
        quickSort(array, low, pi - 1)

        # Recursive call on the right of pivot
        quickSort(array, pi + 1, high)


def str2sec(time_string):
    """
    Converts a time code string into seconds.

    Args:
        time_string (str): The time code to convert. Eg. '01:33:42'.

    Returns:
        float: The time code converted to seconds.
    """
    elems = [float(elem) for elem in time_string.split(':')]
    return elems[0]*3600 + elems[1]*60 + elems[2]


def kl_scheduler(epoch, cycle_period, ramp_up_phase=0.5):
    """
    A cyclical scheduler for the KL divergence weight.
    In the first half of the cycle, the weight increases linearly from 0 to 1.
    In the second half, the weight stays at 1.

    Args:
        epoch (int): current epoch
        cycle_period (int): number of epochs in a cycle

    Returns:
        float: the weight for the KL divergence loss
    """
    epoch_mod = epoch % cycle_period
    epoch_mod_norm = epoch_mod / cycle_period
    return np.clip(epoch_mod_norm * 1/ramp_up_phase, 0, 1)


# exponential moving average
def ema(last_val, new_val, alpha=0.99):
    return alpha*last_val + (1-alpha)*new_val