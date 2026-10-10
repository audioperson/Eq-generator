import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def vergleiche_audios(d1, d2):
    print('lade Audiodateien')
    y1, sr1 = librosa.load(d1, sr=None, duration = 5)
    y2, sr2 = librosa.load(d2, sr=None, duration = 5)

    min_len = int(min(len(y1), len(y2)))

    y1 = y1[:min_len]
    y2 = y2[:min_len]

    print('Frequenzanalyse')

    fft1 = np.abs(np.fft.rfft(y1))
    fft2 = np.abs(np.fft.rfft(y2))
    freqs = np.fft.rfftfreq(min_len, d=1/sr1)

    fft1 = np.maximum(fft1, 1e-5)
    fft2 = np.maximum(fft2, 1e-5)

    #verbesserungs eqs
    eq_kurve_db = 20*np.log10(fft1/fft2)

    #kurve glätten
    fenster_groesse = int(len(eq_kurve_db) / 120)
    if fenster_groesse % 2 == 0: fenster_groesse +=1 
    glatte_eq_kurve = np.convolve(eq_kurve_db, np.ones(fenster_groesse) / fenster_groesse, mode='same')

    plt.figure(figsize=(10, 5))

    plt.plot(freqs, glatte_eq_kurve, color='tab:green', linewidth=2, label='Korrektur-Kurve', zorder=2)

    plt.fill_between(freqs, glatte_eq_kurve, 0, color='tab:green', alpha=0.3, zorder=2)

    plt.axhline(y=0, color='black', linestyle='-', linewidth=1, zorder=1)

    #logarithmische achsenkonfig

    plt.xscale('log')
    plt.xlim(20, 20000)

    #scale +30 -30
    plt.ylim(-30, 30)

    plt.title('Eq 2->1 (vorgeschlagen)')
    plt.xlabel('Hz')
    plt.ylabel('db')

    plt.xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])

    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()

    print('graphikfenster öfffnet')
    plt.show(block=True)
    


vergleiche_audios('1.mp3', '2.mp3')