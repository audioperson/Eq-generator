import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import time
import os
from scipy.signal import correlate, correlation_lags
from scipy.signal import welch 
from scipy.signal import find_peaks
from scipy.optimize import least_squares
from scipy.signal import freqz

#einstellungen
samplerate = 44100
dauer = 5
datei_kombiniert = 'kombiniert.wav'
bands = 8
min_gain_db = 0.5

def laufzeit_korrektion(y1, y2, sr, max_delay_ms=40):
    

    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)

    if len(y1) == 0 or len (y2) == 0:
        print('file is empty')

    n = min(len(y1), len(y2))
    x = y1[:n] - np.mean(y1[:n])
    y = y2[:n] - np.mean(y2[:n])

    if np.sum(x**2) < 1e-12 or np.sum(y**2) < 1e-12:
        print('to quiet')

    max_lag = int(sr * max_delay_ms / 1000)

    #kreutzkorrelation
    corr = correlate(x, y, mode='full', method='fft')
    lags = correlation_lags(len(x), len(y), mode='full')

    maske = np.abs(lags) <= max_lag
    corr = corr[maske]
    lags = lags[maske]

    norm = np.sqrt(np.sum(x**2) * np.sum(y**2))
    corr_norm = corr / norm 

    index = np.argmax(np.abs(corr_norm))
    lag = int(lags[index])
    korrelation = float(corr_norm[index])

    delay_ms = lag / sr * 1000

    print(f"time difference: {delay_ms}")

    if lag > 0:
       y1_aligned = y1[lag:]
       y2_aligned = y2[:-lag]
    elif lag < 0:
      n_versatz = -lag
      y1_aligned = y1[:-n_versatz] 
      y2_aligned = y2[n_versatz:]
    else:
       y1_aligned = y1.copy()
       y2_aligned = y2.copy()

    n_aligned = min(len(y1_aligned), len(y2_aligned))

    if n_aligned == 0:
       print('not the same lenght')

    return (
        y1_aligned[:n_aligned],
        y2_aligned[:n_aligned],
        delay_ms
    )


def peaking_eq_response(freqs, sr, freq, gain_db, Q):

    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * Q)
    cos_w0 = np.cos(w0)

    b = np.array([
        1 + alpha * A,
        -2 * cos_w0,
        1 - alpha * A
    ])

    a = np.array([
        1 + alpha / A,
        -2 * cos_w0,
        1 - alpha / A
    ])

    b = b / a[0]
    a = a / a[0]

    omega = 2 * np.pi * np.asarray(freqs) / sr
    _, h = freqz(b, a, worN=omega)

    magnitude_db = 20 * np.log10(
        np.maximum(np.abs(h), 1e-12)
    )

    return magnitude_db


def combined_eq_response(freqs, sr, params, bands):

    response = np.zeros_like(
        np.asarray(freqs, dtype=float)
    )

    for i in range(bands):

        log_freq, gain_db, log_Q = params[3*i:3*i+3]

        freq = 10 ** log_freq
        Q = 10 ** log_Q

        response += peaking_eq_response(
            freqs, sr, freq, gain_db, Q
        )

    return response


def optimize_eq(freqs, eq_db, sr, bands=bands,
                max_gain_db=20.0,
                q_min=0.2, q_max=8.0, min_gain_db=0.5):

    freqs = np.asarray(freqs, dtype=float)
    eq_db = np.asarray(eq_db, dtype=float)

    f_min = max(20.0, float(np.min(freqs)))
    f_max = min(20000.0, float(np.max(freqs)))

    bands = min(max(1, int(bands)), bands)

    valid = (
        np.isfinite(freqs) &
        np.isfinite(eq_db) &
        (freqs > 0) &
        (freqs < sr / 2)
    )

    freqs = freqs[valid]
    eq_db = eq_db[valid]

    if len(freqs) < 3:

        f_min = max(20.0, float(freqs.min()))
        f_max = min(
        20000.0,
        float(freqs.max()),
        sr / 2 * 0.999
    )

    peaks_pos, _ = find_peaks(eq_db, prominence=0.5)
    peaks_neg, _ = find_peaks(-eq_db, prominence=0.5 )

    candidates = list(peaks_pos) + list(peaks_neg)

    candidates = sorted(
        set(candidates),
        key=lambda i: abs(eq_db[i]),
        reverse=True
    )

    start_freqs = []

    for idx in candidates:

        f = float(freqs[idx])

        if all(
            abs(np.log2(f / old)) > 0.5
            for old in start_freqs
        ):
            start_freqs.append(f)

        if len(start_freqs) >= bands:
            break

    fallback_freqs = np.geomspace(
        f_min, f_max, bands + 2
    )[1:-1]

    for f in fallback_freqs:
        if len(start_freqs) >= bands:
            break

        if all(
            abs(np.log2(f / old)) > 0.20
            for old in start_freqs
        ):
            start_freqs.append(float(f))

    for f in fallback_freqs:

        if len(start_freqs) >= bands:
            break

        if not any(
            np.isclose(f, old) for old in start_freqs
        ):
            start_freqs.append(float(f))

    start_freqs = start_freqs[:bands]
    n_bands = len(start_freqs)

    if n_bands == 0:
        return 

    x0 = []

    for f in start_freqs:

        idx = np.argmin(np.abs(freqs - f))

        initial_gain = np.clip(
            eq_db[idx],
            -max_gain_db,
            max_gain_db
        )

        x0.extend([
            np.log10(f),
            initial_gain,
            np.log10(1.0)
        ])

    # Grenzen für die Optimierung
    lower = []
    upper = []

    for _ in range(n_bands):

        lower.extend([
            np.log10(f_min),
            -max_gain_db,
            np.log10(q_min)
        ])

        upper.extend([
            np.log10(f_max),
            max_gain_db,
            np.log10(q_max)
        ])

    # Abweichung zwischen Messkurve und EQ-Modell
    def residual(params):

        n_bands = len(params) // 3

        modeled = combined_eq_response(
            freqs, sr, params, n_bands
        )

        fit_error = modeled - eq_db

        log_freq = params[0::3]
        min_log_gap = np.log10(2 ** 0.5)

        penalties = []

        for i in range(n_bands):
            for j in range(i + 1, n_bands):
                distance = abs(log_freq[i] - log_freq[j])

                penalty = 60.0 * max(0.0, min_log_gap - distance)

                penalties.append(penalty)


        return np.concatenate([fit_error, np.asarray(penalties)])

    # EQ-Parameter optimieren
    result = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=3000
    )

    params = result.x.copy()

    keep = []

    # Parameter umwandlung

    for i in range(len(params) // 3):
        gain = params[3* i + 1]

        if abs(gain) >= min_gain_db:
            keep.extend([3 * i, 3 * i +1, 3 * i +2])
        else:
            print:(f"Removing unnecessary frequency band" 
                   f"{10**params[3*i]:.1f} Hz"
                   f"{gain:+2f} db"
                   )
    if not keep:
        print:('no relevant bands found')
        return[]

    params = params[keep]
    lower = np.asarray(lower)[keep]
    upper = np.asarray(upper)[keep]

    result_final = least_squares(
        residual,
        params,
        bounds=(lower, upper),
        loss='soft_l1',
        f_scale=1.0,
        max_nfev=3000
    )
    params = result_final.x

    eq_bands = []

    for i in range(len(params) // 3):
        log_freq, gain_db, log_Q = params[3*i:3*i+3]

        freq = 10 ** log_freq
        Q = 10 ** log_Q

        eq_bands.append((freq, gain_db, Q))

        eq_bands.sort(key=lambda band: band[0])

    return eq_bands    

    

    
 
    

def analyze_audio_files(datei1, datei2):
    print('original:', datei1)
    print('comparison:', datei2)

    y1, sr1 = librosa.load(datei1, sr = None, duration = dauer)
    y2, sr2 = librosa.load(datei2, sr = sr1, duration = dauer)

    print('files loaded')


def main(datei1 = None, datei2 = None):
    print("==================================================")
    print("        AUDIO-COMPARISON & EQ-MATCHING TOOL        ")
    print("==================================================")
    print("Pick the mode:")
    print(" two existing files")
    print(" live record")
    
    #wahl = input("your choise (1 oder 2): ").strip()
    if datei1 is not None and datei2 is not None:
        wahl='1'
    else:
        wahl = input("your choise (1 oder 2): ").strip()
    
    y1, y2, sr = None, None, samplerate

    # Modus 1
    if wahl == "1":
        print("\n--- mode: two existing files---")

        if datei1 is None or datei2 is None:
            datei1 = input("Name of file 1 (z.B. 1.wav): ").strip()
            datei2 = input("Name of file 2 (z.B. 2.wav): ").strip()
        
        # Dateiprüfen
        if not os.path.exists(datei1) or not os.path.exists(datei2):
            print(f"\nfiles not found")
            print(f"make shure that '{datei1}' and '{datei2}' are in the folder")
            exit(1)
            
        print('\nloading files...')
        y1, sr1 = librosa.load(datei1, sr=None, duration=dauer)
        y2, sr2 = librosa.load(datei2, sr=sr1, duration=dauer)
        sr = sr1
        
        # Stereo-Dateien zur Sicherheit in Mono umwandeln
        if len(y1.shape) > 1: y1 = librosa.to_mono(y1)
        if len(y2.shape) > 1: y2 = librosa.to_mono(y2)

    #Modus2
    elif wahl == "2":
        print("\n--- mode: Live-record ---")
        
       
        #print("\nVerfügbare Audiogeräte mit Eingängen:")
        verfügbare_ids = []
        for i, d in enumerate(sd.query_devices()):
            if d['max_input_channels'] > 0:
                print(f"  ID {i}: {d['name']} (channels: {d['max_input_channels']})")
                verfügbare_ids.append(i)
        
        # Standard-Eingangsgerät als Vorschlag ermitteln
        standard_id = sd.default.device[0]
        id_eingabe = input(f"\nenter device ID (normaly {standard_id}): ").strip()
        
        # Wenn der Nutzer einfach Enter drückt, nimm den Standard
        if id_eingabe == "":
            GERAET_ID = standard_id
        else:
            try:
                GERAET_ID = int(id_eingabe)
            except ValueError:
                print("\n[FEHLER] pleas chose a number")
                exit(1)
        
        #dateien löschen
        if os.path.exists(datei_kombiniert):
            try:
                os.remove(datei_kombiniert)
                print(f"-> old file'{datei_kombiniert}' has been deleted.")
            except Exception as e:
                print(f"-> mistake while deleting: {e}")
                
        print(f"\n=== starting Recording ({dauer}s) ===")
        
        try:
            anzahl_kanäle = int(sd.query_devices(GERAET_ID)['max_input_channels'])
            if anzahl_kanäle < 2:
                print("\nthe device hasent 2 input channels")
                anzahl_kanäle = 2
                
            #print(f"Nimmt von Gerät ID {GERAET_ID} mit {anzahl_kanäle} Kanälen zeitgleich auf...")
            daten = sd.rec(int(dauer * samplerate), samplerate=samplerate, channels=anzahl_kanäle, device=GERAET_ID, blocking=True)
            sf.write(datei_kombiniert, daten, samplerate)
            print("-> record saved")
        except Exception as e:
            print(f"\nmistake while recording {e}")
            print("pleas make shure its the corect device")
            exit(1)
            
        time.sleep(1.5)
        
        #print('Lade aufgezeichnete Spuren for den Vergleich...')
        y, sr = librosa.load(datei_kombiniert, sr=None, mono=False, duration=dauer)
        
        if len(y.shape) > 1 and y.shape[0] >= 2:
            y1 = y[0]  # Kanal 1 (Mikrofon 1)
            y2 = y[1]  # Kanal 2 (Mikrofon 2)
        else:
            #print("\n[FEHLER] Die Aufnahme hat keine 2 Kanäle geliefert.")
            exit(1)
            
    else:
        print("\npleas restart.")
        exit(1)


    min_len = int(min(len(y1), len(y2)))
    y1 = y1[:min_len]
    y2 = y2[:min_len]

    y1, y2, delay_ms = laufzeit_korrektion(y1, y2, sr, max_delay_ms=20)

    print('Gain corection')

    rms1 = np.sqrt(np.mean(y1**2))
    rms2 = np.sqrt(np.mean(y2**2))

    if rms1 < 1e-8 or rms2 < 1e-8:
        raise ValueError('one signal is to quied')

    gain_faktor = rms1 / rms2
    gain_db = 20 * np.log10(gain_faktor)
    
    y2 = y2 * gain_faktor

    rms1_after = np.sqrt(np.mean(y1**2))
    rms2_after = np.sqrt(np.mean(y2**2))
    print(f"Gain factor: {gain_faktor:.4f}")
    print(f"Gain correction: {gain_db:+.2f} db")

    print('FFT')
    min_len = min(len(y1), len(y2))
    y1 = y1[:min_len]
    y2 = y2[:min_len]

    nperseg = min(16384, min_len)

    freqs, leistung1 = welch(
        y1, fs=sr, nperseg=nperseg, scaling='density'
    )
    _, leistung2 = welch(
        y2, fs=sr, nperseg=nperseg, scaling ='density'
    )

    leistung1 = np.maximum(leistung1, 1e-12)
    leistung2 = np.maximum(leistung2, 1e-12)

    eq_kurve_db = 10 * np.log10(leistung1 / leistung2)

    offset_maske = (
        (freqs >= 100) &
        (freqs <= min(10000, sr / 2)) 
    )             

    eq_offset_db = np.median(eq_kurve_db[offset_maske])

    eq_kurve_db = eq_kurve_db - eq_offset_db

    maske = (freqs >= 20) & (freqs <= min(20000, sr / 2))
    freqs = freqs[maske]
    eq_kurve_db = eq_kurve_db[maske]
    

    anzahl_punkte = 500
    freqs_log = np.geomspace(20, min(20000, sr / 2), anzahl_punkte)

    eq_log = np.interp(
        np.log10(freqs_log),
        np.log10(freqs),
        eq_kurve_db
    )

    fenster_groesse = 5
    glatte_eq_kurve = np.convolve(eq_log, np.ones(fenster_groesse) / fenster_groesse, mode='same')

    eq_bands = optimize_eq(
        freqs_log,
        glatte_eq_kurve,
        sr,
        bands=bands,
        max_gain_db=20.0,
        q_min=0.2,
        q_max=8.0,
        min_gain_db=min_gain_db
    )

    print('--- Parametric EQ settings ---')

    for i, (freq, gain, Q) in enumerate(eq_bands, start=1):
      print(
          f"band {i}: "
          f"frequency = {freq:.0f} Hz, "
          f"Gain = {gain:.1f} db, "
          f"Q = {Q:.2f}"
            )

    modell_params = np.array([p for freq, gain, Q in eq_bands for p in (np.log10(freq), gain, np.log10(Q))])
    modell_eq = combined_eq_response(freqs_log, sr, modell_params, len(eq_bands))
    #Graphen
    """plt.figure(figsize=(10, 5))
    plt.plot(freqs_log, glatte_eq_kurve, color='tab:green', linewidth=2, label='Corection EQ', zorder=2)
    plt.plot(freqs_log, modell_eq, color='tab:blue', linewidth=2, linestyle='--', label='Optemized', zorder=3)
    plt.legend()
    plt.fill_between(freqs_log, glatte_eq_kurve, 0, color='tab:green', alpha=0.3, zorder=2)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1, zorder=1)

    plt.xscale('log')
    plt.xlim(20, 20000)
    plt.ylim(-30, 30)

    plt.title('Eq 2->1 (corection EQ)')
    plt.xlabel('Hz')
    plt.ylabel('dB')

    plt.xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000], 
               ["20", "50", "100", "200", "500", "1k", "2k", "5k", "10k", "20k"])

    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.show(block=True)"""
    return freqs_log, glatte_eq_kurve, modell_eq, eq_bands

if __name__ == '__main__':
    main()