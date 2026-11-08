import sys
import numpy as np
import soundfile as sf
import sys

from room_analysis import analyze_room

def load_mono(path):
    data, sample_rate = sf.read(path, dtype = 'float32')

    if data.ndim > 1:
        data = data.mean(axis = 1)

    if sample_rate != 44100:
        new_length = int(len(data) * 44100 / sample_rate)
        data = np.interp(
            np.linspace(0, len(data) -1, new_length),
            np.arange(len(data)),
            data
        ).astype('float32')

    return data

if len(sys.argv) != 3:
    print('usage: python3 test_room.py reference.wav measured.wav')
    sys.exit(1)

reference = load_mono(sys.argv[1])
measured = load_mono(sys.argv[2])

result = analyze_room(reference, measured, 44100)

if 'error' in result:
    print('error:', result['error'])
    sys.exit(1)

print(f"delay: {result['delay_ms']:.1f} ma ({result['distance_m']:.1f} m)")
print(f"direct/reverb: {result['direct_to_reverb_db']:+.1f} dB")
print(f"clarity C50: {result['clarity_db']:+.1f} dB")

for hz, value in result['reverb_times'].items():
    text = 'not reliable' if value is None else f"{value:.2f} s"
    print(f"reverb {hz:5d} Hz: {text}")
