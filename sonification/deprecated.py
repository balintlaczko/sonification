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
        output_row = generate_row_with_hops(
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
        output_row = generate_row_with_hops(
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


# function: generate_rows_with_hops

@njit(parallel=True)
def generate_rows_with_hops(
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
        output_row = generate_row_with_hops(
            num_hops, num_cols, hop_size_samps, y, sr, sine_length, hz_range[row], db_range)
        # accumulate result to output
        output_rows += output_row
    # return accumulated result
    return output_rows


# function: generate_row_with_hops

@jit(nopython=True)
def generate_row_with_hops(
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
    sine_buffer = generate_sine(sr, sine_length, frequency, 0, 4096)
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


class Tsynth_FmSynth(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.fm_vco = FmVCO(
            tuning=torch.tensor([0.0] * config.batch_size),
            mod_depth=torch.tensor([0.0] * config.batch_size),
            synthconfig=config,
            device=device,
        )
        self.modulator = SineVCO(
            tuning=torch.tensor([0.0] * config.batch_size),
            mod_depth=torch.tensor([0.0] * config.batch_size),
            initial_phase=torch.tensor([torch.pi / 2] * config.batch_size),
            synthconfig=config,
            device=device,
        )

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=80,
            normalized=True,
            f_min=20.0,
            f_max=8000.0,
        ).to(device)

    def forward(
            self,
            carrier_frequency: torch.Tensor,
            modulator_frequency: torch.Tensor,
            modulation_index: torch.Tensor
    ):
        # generate modulator signal
        modulator_signal = self.modulator(
            frequency2midi_tensor(modulator_frequency))

        # set modulation index
        self.fm_vco.set_parameter(
            "mod_depth", torch.clip(modulation_index, -96, 96))

        # generate fm signal
        fm_signal = self.fm_vco(frequency2midi_tensor(
            carrier_frequency), modulator_signal)

        # generate mel spectrogram
        mel_spec = self.mel_transform(fm_signal)

        return mel_spec
    

class Tsynth_FmSynth2Wave(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.fm_vco = FmVCO(
            tuning=torch.tensor([0.0] * config.batch_size),
            mod_depth=torch.tensor([0.0] * config.batch_size),
            synthconfig=config,
            device=device,
        )
        self.modulator = SineVCO(
            tuning=torch.tensor([0.0] * config.batch_size),
            mod_depth=torch.tensor([0.0] * config.batch_size),
            initial_phase=torch.tensor([torch.pi / 2] * config.batch_size),
            synthconfig=config,
            device=device,
        )

    def forward(
            self,
            carrier_frequency: torch.Tensor,
            modulator_frequency: torch.Tensor,
            modulation_index: torch.Tensor
    ):
        # generate modulator signal
        modulator_signal = self.modulator(
            frequency2midi_tensor(modulator_frequency))

        # set modulation index
        self.fm_vco.set_parameter(
            "mod_depth", torch.clip(modulation_index, -96, 96))

        # generate fm signal
        fm_signal = self.fm_vco(frequency2midi_tensor(
            carrier_frequency), modulator_signal)

        return fm_signal
    

class FM_Autoencoder_Wave(nn.Module):
    def __init__(self, config, device, z_dim=32, mlp_dim=512):
        super().__init__()

        self.encoder = Wave2MFCCEncoder(z_dim=z_dim)
        self.decoder = MLP(z_dim, mlp_dim, mlp_dim, 3)
        self.synthparams = nn.Linear(mlp_dim, 3)
        self.synth_act = nn.ReLU()
        self.synth = Tsynth_FmSynth2Wave(config, device)

    def forward(self, x):
        z = self.encoder(x)
        mlp_out = self.decoder(z)
        params = self.synthparams(mlp_out)
        params = self.synth_act(params)
        # print("params.shape", params.shape)
        # print("params", params)
        carrier_frequency = params[:, 0]
        # print("carrier_frequency", carrier_frequency)
        modulator_frequency = params[:, 1]
        # print("modulator_frequency", modulator_frequency)
        modulation_index = params[:, 2]
        # print("modulation_index", modulation_index)
        # print("mod index sum:", modulation_index.sum())
        fm_signal = self.synth(
            carrier_frequency, modulator_frequency, modulation_index)
        return fm_signal
    

class FM_Autoencoder_Wave2(nn.Module):
    def __init__(self, config, device, z_dim=32, mlp_dim=512):
        super().__init__()

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=80,
            normalized=True,
            f_min=20.0,
            f_max=8000.0,
        ).to(device)

        self.spec_encoder = MultiScaleEncoder(input_dim_h=80, input_dim_w=188)

        self.z = nn.Sequential(
            nn.Linear(128 * 20 * 47, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, z_dim),
            nn.ReLU(),
        )

        self.decoder = MLP(z_dim, mlp_dim, mlp_dim, 3)
        self.synthparams = nn.Linear(mlp_dim, 3)
        self.synth_act = nn.ReLU()
        self.synth = Tsynth_FmSynth2Wave(config, device)

    def forward(self, x):
        # print("x.shape", x.shape)
        mel_spec = self.mel_transform(x)
        # print("mel_spec.shape", mel_spec.shape)
        mel_spec = mel_spec.unsqueeze(1)
        encoded = self.spec_encoder(mel_spec)
        # print("encoded.shape", encoded.shape)
        encoded_flatten = encoded.view(encoded.shape[0], -1)
        z = self.z(encoded_flatten)
        # print("z.shape", z.shape)
        mlp_out = self.decoder(z)
        # print("mlp_out.shape", mlp_out.shape)
        params = self.synthparams(mlp_out)
        params = self.synth_act(params)
        # print("params.shape", params.shape)
        # print("params.shape", params.shape)
        # print("params", params)
        carrier_frequency = params[:, 0]
        # print("carrier_frequency", carrier_frequency)
        modulator_frequency = params[:, 1]
        # print("modulator_frequency", modulator_frequency)
        modulation_index = params[:, 2]
        # print("modulation_index", modulation_index)
        # print("mod index sum:", modulation_index.sum())
        fm_signal = self.synth(
            carrier_frequency, modulator_frequency, modulation_index)
        return fm_signal


class FM_Autoencoder(nn.Module):
    def __init__(
            self,
            config,
            device,
            num_mels=80,
            num_hops=188,
            num_params=3,
            dim=256
    ):
        super().__init__()

        self.num_mels = num_mels
        self.num_hops = num_hops
        self.num_params = num_params

        # self.encoder = Mel2Params(num_mels, num_hops, num_params, dim=dim)
        self.encoder = Mel2Params2(num_mels, num_hops, num_params, dim=dim)
        self.synth = Tsynth_FmSynth(config, device)

    def forward(self, mel_spec):
        params = self.encoder(mel_spec)
        carrier_frequency = params[:, 0]
        modulator_frequency = params[:, 1]
        modulation_index = params[:, 2]
        mel_spec_recon = self.synth(
            carrier_frequency,
            modulator_frequency,
            modulation_index
        )
        return mel_spec_recon, params
    

class FM_Param_Autoencoder(nn.Module):
    def __init__(
            self,
            config,
            device,
            num_mels=80,
            num_hops=188,
            num_params=3,
            dim=256
    ):
        super().__init__()

        self.num_mels = num_mels
        self.num_hops = num_hops
        self.num_params = num_params

        self.synth = Tsynth_FmSynth(config, device)
        self.decoder = Mel2Params2(num_mels, num_hops, num_params, dim=dim)
        # self.decoder = Mel2Params(num_mels, num_hops, num_params, dim=dim)

    def forward(self, params):
        mel_spec = self.synth(
            params[:, 0],  # carrier_frequency
            params[:, 1],  # modulator_frequency
            params[:, 2]  # modulation_index
        )
        params_recon = self.decoder(mel_spec)
        return params_recon
