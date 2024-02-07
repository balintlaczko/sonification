def scale_linear(x, in_low, in_high, out_low, out_high):
    return (x - in_low) / (in_high - in_low) * (out_high - out_low) + out_low


def midi2frequency(midi, base_frequency=440.0):
    return base_frequency * 2**((midi - 69) / 12)