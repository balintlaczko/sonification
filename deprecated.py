from utils import *
import math
import pyo


@jit(nopython=True)
def hop2col(hop, num_hops, num_cols):
    return hop * num_cols / num_hops


def image2sines_unoptimized(
    image_path: str,
    target_name: str,
    out_length: float = 4.0,
    num_sines: int = 6,
    sine_length: float = 1.0,
    overlap: int = 1,
    sr: int = 44100,
    lowest_freq: float = 50.0,
    highest_freq: float = 10000.0,
    time_dim: int = 0,
    overwrite: bool = False,
):
    # get image
    folder = os.path.dirname(image_path)
    image_matrix = cv2.imread(image_path)
    if len(image_matrix.shape) > 2:
        image_matrix = cv2.cvtColor(image_matrix, cv2.COLOR_BGR2GRAY)

    if time_dim == 0:
        image_matrix = np.transpose(image_matrix)

    print("image matrix:", image_matrix.shape)

    # scale height to num_sines
    image_as_spectrogram = cv2.resize(
        image_matrix, (num_sines, image_matrix.shape[-1]))
    # normalize 8-bit image
    image_norm = image_as_spectrogram.astype(np.float32) / 255.0
    print("image_norm:", image_norm.shape)

    num_cols = image_norm.shape[0]
    print("num_cols:", num_cols)
    print("sine length:", sine_length)
    hop_size = sine_length / (overlap+1)
    hop_size_samps = librosa.time_to_samples(hop_size, sr=sr)
    num_hops = math.ceil(out_length / hop_size)
    # sine_length = out_length / num_cols * 2
    print("hop_size:", hop_size)
    print("hop_size_samps:", hop_size_samps)
    print("num_hops:", num_hops)

    nyquist = sr / 2
    lowest_midi, highest_midi = librosa.hz_to_midi([lowest_freq, highest_freq])
    midi_range = np.linspace(lowest_midi, highest_midi, num_sines)
    hz_range = librosa.midi_to_hz(midi_range)
    output_row = None
    output_rows = None
    sine = Sinetable()

    x = np.arange(0, image_matrix.shape[-1])
    print("x", x)
    y = np.sin(2 * np.pi * x / np.max(x))
    for row in range(num_sines):
        insertion_index = 0
        print("row:", row)
        y = image_norm[:, row]
        # print(y.shape)
        sampler = interpolate.interp1d(x, y, kind="cubic")

        for hop in range(num_hops):
            col = hop2col(hop, num_hops, num_cols)
            print(f"col: {col}", end=" ")
            cell_val = sampler(col)
            # gain_db_scaled = image_norm[col, row] * 70 - 70
            gain_db_scaled = cell_val * 70 - 70
            buffer = sine.generate(
                sr, sine_length, hz_range[-row], gain_db_scaled)
            # insertion_index = -1 * len(buffer) / 2
            if hop == 0:
                output_row = buffer
            else:
                output_row = overlap_add(output_row, buffer, insertion_index)
            insertion_index += hop_size_samps
            print("output_row.shape:", output_row.shape)

        if row == 0:
            output_rows = np.zeros_like(output_row)
        output_rows += output_row

    all_rows = output_rows / num_sines

    if not overwrite:
        target_name = generate_outfilename(target_name)
    wav.write(target_name, sr, all_rows.astype(np.float32))


def image2sines_semi_optimized(
    image_path: str,
    target_name: str,
    out_length: float = 4.0,
    num_sines: int = 6,
    sine_length: float = 1.0,
    overlap: int = 1,
    sr: int = 44100,
    lowest_freq: float = 50.0,
    highest_freq: float = 10000.0,
    time_dim: int = 0,
    overwrite: bool = False,
):
    # get image
    image_matrix = cv2.imread(image_path)
    if len(image_matrix.shape) > 2:
        image_matrix = cv2.cvtColor(image_matrix, cv2.COLOR_BGR2GRAY)

    if time_dim == 0:
        image_matrix = np.transpose(image_matrix)

    print("image matrix:", image_matrix.shape)

    # scale height to num_sines
    image_as_spectrogram = cv2.resize(
        image_matrix, (num_sines, image_matrix.shape[-1]))
    # normalize 8-bit image
    image_norm = image_as_spectrogram.astype(np.float32) / 255.0
    print("image_norm:", image_norm.shape)

    num_cols = image_norm.shape[0]
    print("num_cols:", num_cols)
    print("sine length:", sine_length)
    hop_size = sine_length / (overlap+1)
    hop_size_samps = librosa.time_to_samples(hop_size, sr=sr)
    num_hops = math.ceil(out_length / hop_size)
    print("hop_size:", hop_size)
    print("num_hops:", num_hops)

    lowest_midi, highest_midi = librosa.hz_to_midi([lowest_freq, highest_freq])
    midi_range = np.linspace(lowest_midi, highest_midi, num_sines)
    hz_range = librosa.midi_to_hz(midi_range)
    output_row = None
    output_rows = None

    for row in range(num_sines):
        y = image_norm[:, row]
        output_row = generate_row(
            num_hops, num_cols, hop_size_samps, y, sr, sine_length, hz_range, row)
        if row == 0:
            output_rows = np.zeros_like(output_row)
        output_rows += output_row

    all_rows = output_rows / num_sines

    if not overwrite:
        target_name = generate_outfilename(target_name)
    wav.write(target_name, sr, all_rows.astype(np.float32))

# serial version


@jit(nopython=True)
def generate_rows_serial(num_rows, matrix, num_hops, num_cols, hop_size_samps, sr, sine_length, hz_range):
    output_rows = np.zeros(
        int(sine_length * sr + (num_hops-1) * hop_size_samps), dtype=np.float64)
    for row in range(num_rows):
        y = matrix[:, row]
        output_row = generate_row(
            num_hops, num_cols, hop_size_samps, y, sr, sine_length, hz_range[row])
        output_rows += output_row
    return output_rows

# image to spectrum using pyo -- requires Python 3.9 or lower...
def image2spectrum(
    image_path: str,
    target_name: str,
    out_length: float = 4.0,
    fft_size: int = 1024,
    overlap: int = 4,
    sr: int = 44100,
    time_dim: str = "width",
    overwrite: bool = False,
) -> None:
    """
    Sonify an image as an FFT spectrum using pyo.IFFTMatrix and write the result to a file. The function will read an image,
    scale its height or width (depending on `time_dim`) to the number of FFT bins (`fft_size` / 2) and then perform inverse
    FFT to synthesize the image as a spectrum of frequencies.

    Args:
        image_path (str): The path to the image to sonify.
        target_name (str): The target name of the rendered wav file. Can end up being different from what was specified if `overwrite=False`.
        out_length (float, optional): The target length of the output wav file in seconds. Defaults to 4.0.
        fft_size (int, optional): The FFT size in samples. Defaults to 1024.
        overlap (int, optional): The amount of temporal overlaps to make. Affects temporal resolution of the result. Defaults to 4.
        sr (int, optional): The sample rate to use. Defaults to 44100.
        time_dim (str, optional): The dimension of the input image to interpret as the time dimension. Can be "width" or "height". If it is "width", then the width of the image will be mapped to `out_length` and the height to `fft_size` / 2. Defaults to "width".
        overwrite (bool, optional): If False, the method will avoid overwriting existing files by incrementing `target_name`. Defaults to False.
    """
    # get image
    image_matrix = cv2.imread(image_path)
    # if it's rgb, convert to grayscale
    if len(image_matrix.shape) > 2:
        image_matrix = cv2.cvtColor(image_matrix, cv2.COLOR_BGR2GRAY)
    if time_dim.lower() == "height":
        image_matrix = np.transpose(image_matrix)

    # scale height to number of bins (fft_size/2)
    image_matrix_scaled = cv2.resize(
        image_matrix, (image_matrix.shape[1], int(fft_size/2)))

    # scale it into the range of -1...1
    image_scaled_sn = image_matrix_scaled / 255 * 2 - 1

    # convert to list of lists - necessary for pyo.NewMatrix.replace
    image_scaled_sn_list = list(image_scaled_sn)
    image_scaled_sn_list = [list(row) for row in image_scaled_sn_list]

    # boot server in offline mode
    s = pyo.Server(audio="offline", sr=sr).boot()

    # avoid overwriting existing files if necessary
    if not overwrite:
        target_name = generate_outfilename(target_name)

    # set up recording
    s.recordOptions(dur=out_length, filename=target_name,
                    fileformat=0, sampletype=1)

    # create new matrix container
    fft_matrix = pyo.NewMatrix(
        len(image_scaled_sn_list[0]), len(image_scaled_sn_list))
    # populate container with the image
    fft_matrix.replace(image_scaled_sn_list)
    # define the index that will ramp from 0 to 1 over the course of out_length
    index = pyo.Linseg([(0, 0), (out_length, 1)]).play()
    # define the phase of the iFFT synthesis - from pyo.IFFTMatrix docs:
    # "Try different signals like white noise or an oscillator with a frequency slightly
    # detuned in relation to the frequency of the FFT (sr / fftsize)."
    phase = pyo.Sine(freq=sr/fft_size*0.999, mul=1)
    # synthesize
    fout = pyo.IFFTMatrix(fft_matrix, index, phase, size=fft_size,
                          overlaps=overlap, wintype=2).mix(1).out()

    # starts the recording
    s.start()
    # using return seems to abruptly terminate the recording process,
    # that's why we print instead
    print(target_name)
