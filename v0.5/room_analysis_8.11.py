import numpy as np
from scipy.signal import butter, sosfiltfilt, welch, csd

def estimate_impulse_response(reference, measured, sr, nperseeg = 262144):
    nperseg = min(nperseeg, len(reference))

    _, power_reference = welch(reference, fs = sr, nperseg = nperseg)
    _, power_measured = welch(measured, fs = sr, nperseg = nperseg)
    _, cross_power = csd(reference, measured, fs = sr, nperseg = nperseg)

    power_reference = np.maximum(power_reference, 1e-20)
    coherence_values = np.abs(cross_power) ** 2 / np.maximum(power_reference * power_measured, 1e-30)
    weight = np.clip((coherence_values - 0.05) / 0.1, 0.0, 1.0)

    transfer = cross_power / power_reference * weight
    impulse = np.fft.irfft(transfer, n = nperseg)

    start = int(np.argmax(np.abs(impulse)))
    return impulse[start:], start / sr * 1000

def noise_floor(energy):
    return np.mean(energy[int(len(energy) * 0.85):])

def band_reverb_time(impulse, sr, center_hz, skip_ms = 20, min_range_db = 33):
    sos = butter(
        4, [center_hz / np.sqrt(2), center_hz * np.sqrt(2)],
        'bandpass', fs = sr, output = 'sos'
    )
    energy = sosfiltfilt(sos, impulse) ** 2
    noise = noise_floor(energy)

    skip = int(skip_ms / 1000 * sr)
    usable = energy[skip:int(len(energy) * 0.85)]

    window = int(0.01 * sr)
    peak = np.max(np.convolve(usable, np.ones(window) / window, mode = 'valid'))

    if noise <= 0 or peak <= 0 or 10 * np.log10(peak / noise) < min_range_db:
        return None
    
    energy = np.maximum(usable - noise, 0)


    decay = np.cumsum(energy[::-1])[::-1]
    if len(decay) == 0 or decay[0] <= 0:
        return None

    decay_db = 10 * np.log10(decay / decay[0] + 1e-12)
    first = np.argmax(decay_db <= -5)
    second = np.argmax(decay_db <= -25)

    if second <= first + 10:
        return None

    slope = np.polyfit(np.arange(first, second) / sr, decay_db[first:second], 1)[0]
    return float(-60 / slope)

def direct_to_reverb_db(impulse, sr):
    energy = impulse ** 2
    noise = noise_floor(energy)

    direct = np.sum(np.maximum(energy[:int(0.002 * sr)] - noise, 0))
    reverb = np.sum(np.maximum(energy[int(0.005 * sr):int(len(energy) * 0.85)] - noise, 0))
    
    return float(10 * np.log10(max(direct, 1e-20) / max(reverb, 1e-20)))

def clarity_db(impulse, sr, early_ms = 50):
    energy = impulse ** 2
    noise = noise_floor(energy)
    energy = np.maximum(energy[:int(len(energy) * 0.85)] - noise, 0)

    split = int(early_ms / 1000 * sr)
    early = np.sum(energy[:split])
    late = np.sum(energy[split:])

    return float(10 * np.log10(max(early, 1e-20) / max(late, 1e-20)))
    
def analyze_room(reference, measured, sr, min_seconds = 25):
    reference = np.asarray(reference, dtype = float)
    measured = np.asarray(measured, dtype = float)

    size = min(len(reference), len(measured))
    reference = reference[:size]
    measured = measured[:size]

    if size < sr * min_seconds:
        return {'error': f"need at least {min_seconds} seconds of signal"}

    impulse, delay_ms = estimate_impulse_response(reference, measured, sr)

    if delay_ms > 1000:
        return {'error': 'measured signal comes before reference, check channel order'}

    return {
        'delay_ms': delay_ms,
        'distance_m': delay_ms / 1000 * 343,
        'direct_to_reverb_db': direct_to_reverb_db(impulse, sr),
        'clarity_db': clarity_db(impulse, sr),
        'reverb_times': {
            hz: band_reverb_time(impulse, sr, hz) for hz in (500, 1000, 4000)
        }
    }