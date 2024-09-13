import pyaudio
import numpy as np
from numba import jit, int32, float32
from numba.experimental import jitclass
from sonification.utils.dsp import sinewave
from sonification.utils import array
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import AsyncIOOSCUDPServer
import asyncio

# instantiate PyAudio
p = pyaudio.PyAudio()

print(p.get_default_input_device_info())
print(p.get_default_output_device_info())

sr = 48000
n_channels = 1
dtype = pyaudio.paFloat32
block = 128

# generate a sine wave to modulate the amplitude with
freq = 30
sine_dur = 1
sine_buf = sinewave(int(sr * sine_dur), sr, np.array([freq]))

# create a circular buffer
class CircularBuffer:
    def __init__(self, buffer, block):
        self.buffer = buffer
        self.block = block
        self.index = 0
        self.length = np.shape(buffer)[0]

    def sample(self):
        if self.index + self.block < self.length:
            out = self.buffer[self.index:self.index+self.block]
            self.index += self.block
            return out
        else:
            incomplete_block_length = self.length - self.index
            out = np.concatenate((self.buffer[self.index:], self.buffer[:self.block - incomplete_block_length]))
            self.index = self.block - incomplete_block_length
            return out

# create a circular buffer object
sine_buf = CircularBuffer(sine_buf, block)


# creted an optimized Phasor class that can work real-time
spec = [
    ('sr', int32),
    ('phase', float32),
]

@jitclass(spec)
class Phasor:
    def __init__(self, sr, phase=0):
        self.sr = sr
        self.phase = phase

    def compute(self, frequency: np.array):
        output = np.zeros_like(frequency)
        increment = frequency[0] / self.sr
        output[0] = array.wrap(self.phase + increment, 0, 1)
        for i in range(1, len(frequency)):
            increment = frequency[i] / self.sr
            output[i] = array.wrap(output[i-1] + increment, 0, 1)
        self.phase = output[-1]
        return output
    
@jit(nopython=True)
def phasor2sine(phasor):
    return np.sin(2 * np.pi * phasor)




# stream_in = p.open(format=dtype,
#                      channels=n_channels,
#                      rate=sr,
#                      input=True,
#                      frames_per_buffer=block)

# stream_out = p.open(format=dtype,
#                     channels=n_channels,
#                     rate=sr,
#                     output=True,
#                     frames_per_buffer=block)

# while True:
#     in_data = stream_in.read(block)
#     # convert the data to numpy array
#     in_data = np.frombuffer(in_data, dtype=np.float32)
#     # get the sine wave
#     sine = sine_buf.sample()
#     # modulate the input data with the sine wave
#     out_data = in_data * sine
#     # convert the data back to bytes
#     out_data = out_data.astype(np.float32).tobytes()
#     # write it back to the output stream
#     stream_out.write(out_data)
#     # print(sine_buf.index)



async def loop(callback):
    try:
        stream_out = p.open(format=dtype,
                            channels=n_channels,
                            rate=sr,
                            output=True,
                            frames_per_buffer=block,
                            stream_callback=callback)
        
        # keep main thread alive while playing
        while stream_out.is_active():
            await asyncio.sleep(1)

    # clean up when interrupted    
    # except KeyboardInterrupt:
    except asyncio.CancelledError:
        print(" Interrupted! Closing audio stream...")
        stream_out.close()
        p.terminate()
        print(" Audio stream closed.")

async def init_main():
    # create a phasor object
    phasor = Phasor(sr)
    # create a block of frequency values
    freq = 440
    frequency = np.ones(block) * freq
    def set_frequency(addr, freq):
        nonlocal frequency
        frequency = np.ones(block) * freq
    def callback(in_data, frame_count, time_info, status):
        out_data = phasor2sine(phasor.compute(frequency))
        out_data = out_data.astype(np.float32).tobytes()
        return (out_data, pyaudio.paContinue)
    # create an OSC receiver for the frequency
    disp = Dispatcher()
    disp.map("/frequency", set_frequency)
    ip = "127.0.0.1"
    port = 12345
    server = AsyncIOOSCUDPServer((ip, port), disp, asyncio.get_event_loop())
    transport, protocol = await server.create_serve_endpoint()
    await loop(callback=callback)
    transport.close()

asyncio.run(init_main())