import numpy as np
from numba import jit, njit, prange
import torch
import cv2
import librosa
import scipy.io.wavfile as wav
from scipy import interpolate
from musicalgestures._utils import generate_outfilename
from . import array


# function to calculate number of hops
def num_hops(
        buffer_length: int,
        hop_length: int
) -> int:
    return int(buffer_length / hop_length) + 1


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


def frequency2midi_tensor(
        frequency: torch.Tensor,
        base_frequency: float = 440.0,
) -> torch.Tensor:
    """
    Converts a frequency in Hz to a MIDI note number.

    Args:
        frequency: Frequency in Hz. Can be a scalar or an array as a torch.tensor.
        base_frequency: Frequency of MIDI note 69. Defaults to 440.0.

    Returns:
        np.ndarray: MIDI note number.
    """

    return 69 + 12 * torch.log2(frequency / base_frequency)


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
    interp_points = array.scale_array_auto(row_x, 0, len(curve)-1)
    # return interpolated curve applied to row
    return row * np.interp(interp_points, curve_x, curve)


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
    row_db = array.scale_array(row_samples, 0, 1, -db_range, 0)
    # convert db to amp
    row_amp = np.power(10, row_db / 20)
    # apply curve to sine buffer and return
    return apply_curve(sine_buffer, row_amp)


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
    return array.wrap(delta, -0.5, 0.5)


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
        frequency_resized = array.resize_interp(frequency, samples-1)
    # for each sample after the first
    for i in range(samples-1):
        # calculate increment
        increment = frequency_resized[i] / sr
        # calculate phasor value from last sample and increment
        output[i+1] = array.wrap(increment + output[i], 0, 1)
    return output


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
        modulator_buf *= array.resize_interp(
            modulator_amplitude.astype(np.float64), samples)
    # calculate frequency modulated signal and return fm buffer
    return sinewave(samples, sr, carrier_frequency + modulator_buf)


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
    _carrier_frequency = array.array2broadcastable(
        carrier_frequency.astype(np.float64), samples)
    _harmonicity_ratio = array.array2broadcastable(
        harmonicity_ratio.astype(np.float64), samples)
    _modulation_index = array.array2broadcastable(
        modulation_index.astype(np.float64), samples)

    # calculate modulator frequency
    modulator_frequency = _carrier_frequency * _harmonicity_ratio
    # create modulator buffer
    modulator_buf = sinewave(samples, sr, modulator_frequency)
    # create modulation amplitude buffer
    modulation_amplitude = modulator_frequency * _modulation_index
    # calculate frequency modulated signal and return fm buffer
    return sinewave(samples, sr, _carrier_frequency + (modulator_buf * modulation_amplitude))


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
        mod_amp_resized = array.resize_interp(
            modulator_amplitude.astype(np.float64), samples)
        modulator_buf *= mod_amp_resized
    # calculate amplitude modulated signal and return am buffer
    return sinewave(samples, sr, carrier_frequency) * (modulator_buf + 1 - mod_amp_resized)


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
        mod_amp_resized = array.resize_interp(
            modulator_amplitude.astype(np.float64), samples)
        modulator_buf *= mod_amp_resized
    # calculate amplitude modulator signal and return am buffer
    return modulator_buf + 1 - mod_amp_resized