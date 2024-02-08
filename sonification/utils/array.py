import numpy as np
from numba import jit


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