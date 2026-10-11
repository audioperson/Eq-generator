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

#einstellungen
samplerate = 44100
dauer = 5
datei_kombiniert = 'kombiniert.wav'

def laufzeit_korrektion(y1, y2, sr, max_delay_ms=20):
    

    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)

    if len(y1) == 0 or len (y2) == 0:
        print('datei ist leer')

    n = min(len(y1), len(y2))
    x = y1[:n] - np.mean(y1[:n])
    y = y2[:n] - np.mean(y2[:n])

    if np.sum(x**2) < 1e-12 or np.sum(y**2) < 1e-12:
        print('ein signal ist zu leise')

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

    print(f"Zeitdifferenz: {delay_ms}")

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
       print('Keine gemeinsarme signallänge')

    return (
        y1_aligned[:n_aligned],
        y2_aligned[:n_aligned],
        delay_ms
    )

if __name__ == "__main__":
    print("==================================================")
    print("        AUDIO-VERGLEICH & EQ-MATCHING TOOL        ")
    print("==================================================")
    print("Bitte wähle den Modus:")
    print(" Zwei bestehende Mono-Dateien vergleichen")
    print(" Neue Live-Aufnahme starten (zum einmessen der PA)")
    
    wahl = input("Deine Auswahl (1 oder 2): ").strip()
    
    y1, y2, sr = None, None, samplerate

    # Modus 1
    if wahl == "1":
        print("\n--- Modus: Bestehende Dateien vergleichen ---")
        datei1 = input("Name von Datei 1 (z.B. 1.wav): ").strip()
        datei2 = input("Name von Datei 2 (z.B. 2.wav): ").strip()
        
        # Dateiprüfen
        if not os.path.exists(datei1) or not os.path.exists(datei2):
            print(f"\n[FEHLER] Eine oder beide Dateien wurden nicht gefunden!")
            print(f"Stelle sicher, dass '{datei1}' und '{datei2}' im Ordner liegen.")
            exit(1)
            
        print('\nLade Audiodateien...')
        y1, sr1 = librosa.load(datei1, sr=None, duration=dauer)
        y2, sr2 = librosa.load(datei2, sr=sr1, duration=dauer)
        sr = sr1
        
        # Stereo-Dateien zur Sicherheit in Mono umwandeln
        if len(y1.shape) > 1: y1 = librosa.to_mono(y1)
        if len(y2.shape) > 1: y2 = librosa.to_mono(y2)

    #Modus2
    elif wahl == "2":
        print("\n--- Modus: Live-Aufnahme ---")
        
       
        print("\nVerfügbare Audiogeräte mit Eingängen:")
        verfügbare_ids = []
        for i, d in enumerate(sd.query_devices()):
            if d['max_input_channels'] > 0:
                print(f"  ID {i}: {d['name']} (Kanäle: {d['max_input_channels']})")
                verfügbare_ids.append(i)
        
        # Standard-Eingangsgerät als Vorschlag ermitteln
        standard_id = sd.default.device[0]
        id_eingabe = input(f"\nBitte Geräte-ID eingeben (Standard ist {standard_id}): ").strip()
        
        # Wenn der Nutzer einfach Enter drückt, nimm den Standard
        if id_eingabe == "":
            GERAET_ID = standard_id
        else:
            try:
                GERAET_ID = int(id_eingabe)
            except ValueError:
                print("\n[FEHLER] Ungültige Eingabe! Bitte eine Zahl eingeben.")
                exit(1)
        
        #dateien löschen
        if os.path.exists(datei_kombiniert):
            try:
                os.remove(datei_kombiniert)
                print(f"-> Alte Datei '{datei_kombiniert}' wurde gelöscht.")
            except Exception as e:
                print(f"-> Warnung beim Löschen: {e}")
                
        print(f"\n=== STARTE SYNCHRONE LIVE-AUFNAHME ({dauer}s) ===")
        
        try:
            anzahl_kanäle = int(sd.query_devices(GERAET_ID)['max_input_channels'])
            if anzahl_kanäle < 2:
                print("\n[HARDWARE-WARNUNG] Das gewählte Gerät unterstützt weniger als 2 Kanäle.")
                anzahl_kanäle = 2
                
            print(f"Nimmt von Gerät ID {GERAET_ID} mit {anzahl_kanäle} Kanälen zeitgleich auf...")
            daten = sd.rec(int(dauer * samplerate), samplerate=samplerate, channels=anzahl_kanäle, device=GERAET_ID, blocking=True)
            sf.write(datei_kombiniert, daten, samplerate)
            print("-> Aufnahme erfolgreich gespeichert.")
        except Exception as e:
            print(f"\n[HARDWARE-FEHLER] Aufnahme fehlgeschlagen: {e}")
            print("Bitte überprüfe dein Audio-MIDI-Setup und die eingegebene ID.")
            exit(1)
            
        time.sleep(1.5)
        
        print('Lade aufgezeichnete Spuren for den Vergleich...')
        y, sr = librosa.load(datei_kombiniert, sr=None, mono=False, duration=dauer)
        
        if len(y.shape) > 1 and y.shape[0] >= 2:
            y1 = y[0]  # Kanal 1 (Mikrofon 1)
            y2 = y[1]  # Kanal 2 (Mikrofon 2)
        else:
            print("\n[FEHLER] Die Aufnahme hat keine 2 Kanäle geliefert.")
            exit(1)
            
    else:
        print("\nUngültige Auswahl! Bitte starte das Programm neu.")
        exit(1)


    min_len = int(min(len(y1), len(y2)))
    y1 = y1[:min_len]
    y2 = y2[:min_len]

    y1, y2, delay_ms = laufzeit_korrektion(y1, y2, sr, max_delay_ms=20)

    print('Gainanpassung')
    rms1 = np.sqrt(np.mean(y1**2))
    rms2 = np.sqrt(np.mean(y2**2))

    if rms1 < 1e-5: rms1 = 1e-5
    if rms2 < 1e-5: rms2 = 1e-5

    gain_faktor = rms1 / rms2
    y2 = y2 * gain_faktor
    print(f'-> Signal 2 wurde um den Gain-Faktor {gain_faktor:.2f} angepasst.')

    print('Frequenzanalyse (FFT)')
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

    maske = (freqs >= 20) & (freqs <= 20000)
    freqs = freqs[maske]
    eq_kurve_db = eq_kurve_db[maske]
    
    print('Glätte das Frequenzspektrum')

    anzahl_punkte = 500
    freqs_log = np.geomspace(20, min(20000, sr / 2), anzahl_punkte)

    eq_log = np.interp(
        np.log10(freqs_log),
        np.log10(freqs),
        eq_kurve_db
    )

    fenster_groesse = 5
    glatte_eq_kurve = np.convolve(eq_log, np.ones(fenster_groesse) / fenster_groesse, mode='same')
    
    #Graphen
    plt.figure(figsize=(10, 5))
    plt.plot(freqs_log, glatte_eq_kurve, color='tab:green', linewidth=2, label='Korektur Kurve', zorder=2)
    plt.fill_between(freqs_log, glatte_eq_kurve, 0, color='tab:green', alpha=0.3, zorder=2)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1, zorder=1)

    plt.xscale('log')
    plt.xlim(20, 20000)
    plt.ylim(-30, 30)

    plt.title('Eq 2->1 (Vorgeschlagene Korrekturkurve)')
    plt.xlabel('Hz')
    plt.ylabel('dB')

    plt.xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000], 
               ["20", "50", "100", "200", "500", "1k", "2k", "5k", "10k", "20k"])

    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.show(block=True)

   