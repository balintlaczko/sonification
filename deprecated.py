from utils import *
import math


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
