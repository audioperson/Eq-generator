import sys
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QVBoxLayout,
    QPushButton,
    QFileDialog,
    QTableWidget,
    QTableWidgetItem,
    QHBoxLayout,
    QHeaderView,
    QTabWidget,
    QComboBox,
    QMessageBox
    
)
from PySide6.QtGui import QFont
from PySide6.QtCore import (
    QTimer, QThread, Signal
)
from concurrent.futures import ProcessPoolExecutor
from scipy.signal import welch, csd

import sounddevice as sd
import numpy as np
import pyqtgraph as pg
import soundfile as sf
import time
import traceback
import threading
from collections import deque



import vergleich

def prune_bands(freqs_log, curve, eq_bands, sr, min_improvement_db = 0.3):
    bands = list(eq_bands)

    def fit_error(band_list):
        if not band_list:
            return np.sqrt(np.mean(curve ** 2))
        params = []
        for freq, gain, q in band_list:
            params.extend([np.log10(freq), gain, np.log10(q)])
        model = vergleich.combined_eq_response(freqs_log, sr, params, len(band_list))
        return np.sqrt(np.mean((model - curve) ** 2))

    while bands:
        base_error = fit_error(bands)
        increases = [fit_error(bands[:i] + bands[i + 1:]) - base_error for i in range(len(bands))]
        weakest = int(np.argmin(increases))
        if increases[weakest] < min_improvement_db:
            bands.pop(weakest)
        else:
            break
    return bands

def band_fit_error(freqs_log, curve, band_list, sr):
    if not band_list:
        return np.sqrt(np.mean(curve ** 2))
    params = []
    for freq, gain, q in band_list:
        params.extend([np.log10(freq), gain, np.log10(q)])
    model = vergleich.combined_eq_response(freqs_log, sr, params, len(band_list))
    return np.sqrt(np.mean((model - curve) ** 2))

def choose_bands(freqs_log, curve, sr, max_bands, min_gain_db, tolerance_db = 0.3):
    best_bands = []
    best_error = band_fit_error(freqs_log, curve, [], sr)

    for count in range(1, max_bands + 1):
        candidate = vergleich.optimize_eq(
            freqs_log,
            curve,
            sr,
            bands = count,
            max_gain_db = 20.0,
            q_min = 0.2,
            q_max = 3.0,
            min_gain_db = min_gain_db
        ) or []

        error = band_fit_error(freqs_log, curve, candidate, sr)

        if best_error - error < tolerance_db:
            break

        best_bands = candidate
        best_error = error

    return best_bands

def apply_lag(reference, measured, lag):
    if lag >0:
        reference = reference[lag:]
        measured = measured[:-lag]
    elif lag < 0:
        reference = reference[:lag]
        measured =measured[-lag:]

    size = min(len(reference), len(measured))
    return reference[:size], measured[:size]

def analyze_block(reference, measured, sr, bands, min_gain_db, previous_curve = None, blend = 0.25, fixed_delay_ms = None):
    if fixed_delay_ms is None:
        reference, measured, delay_ms = vergleich.laufzeit_korrektion(
            reference, measured, sr, max_delay_ms = 250
        )
    else:
        delay_ms = fixed_delay_ms
        lag = int(round(delay_ms * sr / 1000))
        reference, measured = apply_lag(reference, measured, lag)

    rms_reference = np.sqrt(np.mean(reference ** 2))
    rms_measured = np.sqrt(np.mean(measured ** 2))

    if rms_reference < 1e-8 or rms_measured < 1e-8:
            return None

    measured = measured * (rms_reference / rms_measured)

    size = min(len(reference), len(measured))
    reference = reference[:size]
    measured = measured[:size]

    segment_length = min(65536, size)

    freqs, power_reference = welch(
        reference, fs = sr, nperseg = segment_length, scaling = 'density'
    )
    _, power_measured = welch(
        measured, fs = sr, nperseg = segment_length, scaling = 'density'
    )
    _, cross_power = csd(
        reference, measured, fs = sr, nperseg = segment_length, scaling = 'density'
    )
    power_reference = np.maximum(power_reference, 1e-12); power_measured = np.maximum(power_measured, 1e-12)
    cross_magnitude = np.maximum(np.abs(cross_power), 1e-12)
    coherence_values = cross_magnitude ** 2 / (power_reference * power_measured)
    reliable = coherence_values >= 0.8

    #print(f"coherence: reliable {reliable.mean() * 100:.0f} % of bins, median {np.median(coherence_values):.2f}")
    #for check_hz in (50, 100, 500, 1000, 5000, 10000, 15000):
        #check_index = np.argmin(np.abs(freqs - check_hz))
        #print(f' {check_hz} Hz: {coherence_values[check_index]:.2f}')
    eq_db = 20 * np.log10(power_reference / cross_magnitude)
    offset_mask = (freqs >= 100) & (freqs <= min(10000, sr / 2)) & reliable
    if np.any(offset_mask):
        eq_db = eq_db - np.median(eq_db[offset_mask])
    weight = np.clip((coherence_values - 0.7) / 0.15, 0.0, 1.0)
    eq_db = eq_db * weight

    

    range_mask = (freqs >= 30) & (freqs <= min(16000, sr / 2))

    freqs = freqs[range_mask]
    eq_db = eq_db[range_mask]

    freqs_log = np.geomspace(30, min(16000, sr / 2), 500)
    eq_log = np.interp(np.log10(freqs_log), np.log10(freqs), eq_db)

    window_size = 9
    smooth_curve = np.convolve(
        eq_log, np.ones(window_size) / window_size, mode = 'same'
    )

    if previous_curve is not None and len(previous_curve) == len(smooth_curve):
        smooth_curve = blend * smooth_curve + (1 - blend) * np.asarray(previous_curve)


    eq_bands = choose_bands(freqs_log, smooth_curve, sr, bands, min_gain_db)
    eq_bands = prune_bands(freqs_log, smooth_curve, eq_bands, sr)

    return eq_bands, freqs_log, smooth_curve, delay_ms

class EqWorker(QThread):
    result_ready = Signal(object)
    failed = Signal(str)

    def __init__(self, pool, reference, measured, previous_curve = None, fixed_delay_ms = None):
        super().__init__()
        self.pool = pool
        self.reference = reference
        self.measured = measured
        self.previous_curve = previous_curve
        self.fixed_delay_ms = fixed_delay_ms

    def run(self):
        try:
            start_time = time.perf_counter()

            future = self.pool.submit(
                analyze_block,
                self.reference,
                self.measured,
                44100,
                vergleich.bands,
                vergleich.min_gain_db,
                self.previous_curve,
                0.25,
                self.fixed_delay_ms
            )
            result = future.result()

            print(f'eq fit took {time.perf_counter() - start_time:.2f} s')
            
            if result is not None:
                self.result_ready.emit(result)

        except Exception:
            self.failed.emit(traceback.format_exc())           




class eq_generator(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setStyleSheet("""
            QMainWindow {
                background-color: #15171c;
            }

            QWidget {
                color: #E8EAF0;
                font-family: Arial;
                font-size: 13px;
            }

            QPushButton {
                background-color: #292D36;
                color: #FFFFFF;
                border: 1px solid #414754;
                border-radius: 6px;
                padding: 10px 16px;
                min-height: 18px;
            }
            QPushButton:hover {
                background-color: #383E4A;
                border-color: #6C8DFF;
            }
            QPushButton:pressed {
                background-color: #20232A;
            }
            QLabel {
                background-color: transparent;
            }
            QStatusBar {
                background-color: #15171c;
                color: #9299A8
            }

        """)   

        self.setWindowTitle('Auto EQ')
        self.resize(1200, 800)

        zentral = QWidget()
        main_layout = QHBoxLayout(zentral)
        sidebar = QVBoxLayout()
        content = QVBoxLayout()

        sidebar_widget = QWidget()
        sidebar_widget.setLayout(sidebar)
        sidebar_widget.setFixedWidth(240)

        main_layout.addWidget(sidebar_widget)
        main_layout.addLayout(content, 1)

        self.settings_tabs = QTabWidget()
        self.settings_tabs.setStyleSheet("""
            QTabWidget::pane {border: 1px solid #414754; background: #15171c;}
            QTabBar::tab {background: #292D36; color: #9299A8; padding: 8px 14px;}
            QTabBar::tab:selected {background: #383E4A; color: #FFFFFF;}
            QComboBox {background-color: #292D36; border: 1px solid #414754; border-radius: 6px; padding: 6px 10px;}
            QComboBox QAbstractItemView {background-color: #20232A; selection-background-color: #383E4A;}
        """) 
        self.audio_tab = QWidget()
        self.audio_tab_layout = QVBoxLayout(self.audio_tab)
        self.device_box = QComboBox()
        self.reference_chanel_box = QComboBox()
        self.measured_chanel_box = QComboBox()
        self.device_info_label = QLabel('')
        self.device_info_label.setWordWrap(True)
        self.device_info_label.setStyleSheet('color: #9299A8')
        self.refresh_button = QPushButton('Refresh devices')
        self.refresh_button.clicked.connect(self.refresh_devices)
        self.device_box.currentIndexChanged.connect(self.device_changed)

        self.audio_tab_layout.addWidget(QLabel('Input device'))
        self.audio_tab_layout.addWidget(self.device_box)
        self.audio_tab_layout.addWidget(QLabel('Reference channel'))
        self.audio_tab_layout.addWidget(self.reference_chanel_box)
        self.audio_tab_layout.addWidget(QLabel('Measured channel'))
        self.audio_tab_layout.addWidget(self.measured_chanel_box)
        self.audio_tab_layout.addWidget(self.device_info_label)
        self.audio_tab_layout.addWidget(self.refresh_button)
        self.audio_tab_layout.addStretch()
         
        self.settings_tabs.addTab(self.audio_tab, 'Audio')

        self.settings_panel = QWidget()
        self.settings_panel.setFixedWidth(280)
        self.settings_panel.setVisible(False)
        settings_layout = QVBoxLayout(self.settings_panel)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.addWidget(self.settings_tabs)

        self.close_settings_button = QPushButton('Close')
        self.close_settings_button.clicked.connect(self.close_settings)
        settings_layout.addWidget(self.close_settings_button)

        main_layout.addWidget(self.settings_panel)

        titel = QLabel('Auto EQ')
        titel.setFont(QFont('Arial', 28, QFont.Weight.Bold))
        titel.setStyleSheet("""
            color: #FFFFFF;
            padding: 8px 0px 2px 0px;
        """)

        subtitel = QLabel('Audio Analysis & Equalization')
        subtitel.setStyleSheet("""
            color: #9299A8;
            font-size: 13px;
            padding-bottom: 12px;
        """)
        subtitel.setWordWrap(True)
    

        self.live_button = QPushButton('Start Live Analysis')
        self.recorded_button = QPushButton('Compare Audio Files')
        self.recorded_button.clicked.connect(self.compare_audio_files)
        self.stop_live_button = QPushButton('Stop Live Analysis')
        self.live_button.clicked.connect(self.start_live_analysis)
        self.stop_live_button.clicked.connect(self.stop_live_analysis)
        self.test_files_button = QPushButton('Load Test Files')
        self.test_files_button.clicked.connect(self.load_test_files)
        self.test_start_button = QPushButton('Start Test Playback')
        self.test_start_button.clicked.connect(self.start_test_playback)
        self.settings_button = QPushButton('Settings')
        self.settings_button.setCheckable(True)
        self.settings_button.clicked.connect(self.toggle_settings)
        self.smooth_button = QPushButton('Smooth: on')
        self.smooth_button.setCheckable(True)
        self.smooth_button.setChecked(True)
        self.smooth_button.setFixedWidth(100)
        self.smooth_button.clicked.connect(self.toggle_smoothing)
        self.smooth_button.setStyleSheet("""
            QPushButton {padding: 4px 10px;}
            QPushButton:checked {border-color: #34d399; color: #34d399;}
        """)

        sidebar.addWidget(titel)

        subtitle_row = QHBoxLayout()
        subtitle_row.addWidget(subtitel, 1)
        subtitle_row.addWidget(self.smooth_button)
        sidebar.addLayout(subtitle_row)
        sidebar.addSpacing(20)

        sidebar.addWidget(self.live_button)
        sidebar.addWidget(self.stop_live_button)
        sidebar.addWidget(self.recorded_button)
        sidebar.addWidget(self.test_files_button)
        sidebar.addWidget(self.test_start_button)
        sidebar.addStretch()
        sidebar.addWidget(self.settings_button)


        self.setCentralWidget(zentral)
        self.live_timer = QTimer(self)
        self.live_timer.setInterval(100)
        self.live_timer.timeout.connect(self.update_live_analysis)
        self.eq_timer = QTimer(self)
        self.eq_timer.setInterval(4000)
        self.eq_timer.timeout.connect(self.update_live_eq)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setMouseEnabled(x = False, y = False)
        self.plot_widget.setLabel('bottom', 'Frequency (Hz)')
        self.plot_widget.setLabel('left', 'Level', units = 'dB')
        self.plot_widget.setLogMode(x = True, y = False)
        self.plot_widget.showGrid(x = True, y = True, alpha = 0.3)
        self.plot_widget.disableAutoRange()
        self.plot_widget.setYRange(-140, 0, padding = 0)
        self.plot_widget.setXRange(np.log10(20), np.log10(20000), padding = 0)
        self.plot_widget.getAxis('bottom').setTicks([[
            (np.log10(20), '20'),
            (np.log10(100), '100'),
            (np.log10(1000), '1000'),
            (np.log10(10000), '10000'), 
            (np.log10(20000), '20000')
        ]])

        self.style_plot(self.plot_widget, 'Live spectrum')
        self.plot_widget.addLegend(offset = (-10, 10))

        content.addWidget(self.plot_widget, 3)
        self.eq_table = QTableWidget(0, 4)
        self.eq_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self.eq_table.setHorizontalHeaderLabels([
            '#', 'Hz', 'dB', 'Q'
        ])
        self.eq_table.setAlternatingRowColors(True)
        self.eq_table.setShowGrid(True)
        self.eq_table.verticalHeader().setVisible(False)
        self.eq_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        sidebar.addWidget(QLabel('EQ bands'))
        sidebar.addWidget(self.eq_table)

        self.eq_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )

        self.diff_plot = pg.PlotWidget()
        self.diff_plot.setMouseEnabled(x = False, y = False)
        self.diff_plot.setLabel('bottom', 'Frequency (Hz)')
        self.diff_plot.setLabel('left', 'Difference', units = 'dB')
        self.diff_plot.setLogMode(x = True, y = False)
        self.diff_plot.showGrid(x = True, y = True, alpha = 0.3)
        self.diff_plot.disableAutoRange()
        self.diff_plot.setYRange(-20, 20, padding = 0)
        self.diff_plot.setXRange(np.log10(20), np.log10(20000), padding = 0)
        self.diff_plot.addItem(
            pg.InfiniteLine(pos = 0, angle = 0, pen = pg.mkPen('#9299A8', width = 1))
        )
        
        content.addWidget(self.diff_plot, 2)

        self.diff_plot.getAxis('bottom').setTicks([[
            (np.log10(20), '20'),
            (np.log10(100), '100'),
            (np.log10(1000), '1000'),
            (np.log10(10000), '10000'),
            (np.log10(20000), '20000')
        ]])
        self.diff_curve = self.diff_plot.plot(
            pen = pg.mkPen((52, 211, 153, 70), width = 1)
        )
        self.diff_smooth_curve = self.diff_plot.plot(
            pen = pg.mkPen('#34d399', width = 2)
        )
        self.analysis_curve = self.diff_plot.plot(
            pen = pg.mkPen('#f59e0b', width = 1)
        )
        self.model_curve = self.diff_plot.plot(
            pen = pg.mkPen('#6C8DFF', width = 2)
        )

        self.style_plot(self.diff_plot, 'Difference')

        self.eq_worker = None
        self.eq_pool = ProcessPoolExecutor(max_workers = 1)

        self.previous_curve = None
        self.fixed_delay_ms = None

        self.audio_stream = None
        self.test_reference = None
        self.test_measured = None
        self.test_position = 0
        self.test_mode = False
        self.test_timer = QTimer(self)
        self.test_timer.setInterval(10)
        self.test_timer.timeout.connect(self.feed_test_audios)
        self.lastest_peak = 0.0
        self.lastest_rms = 0.0
        self.buffer_size = 16384
        self.samples_recived = 0
        self.ring_buffer = np.zeros(self.buffer_size, dtype='float32') #0,37s
        self.peak_hold_db = None
        self.peak_decay_db_per_s = 6.0
        self.peak_smooth_bins = 3
        self.peak_smooth_strength = 0.3
        self.band_centers = 20 * 2 ** (np.arange(60) / 6)
        self.diff_smooth_db = None
        self.diff_smooth_alpha = 0.03
        self.live_min_db = 1.0
        self.min_level_db = -60.0

        self.display_smoothing_on = True
        self.display_split_hz = 400
        self.display_bands_per_octave = 24
        fft_freqs = np.fft.rfftfreq(self.buffer_size, d= 1 / 44100)
        half_band = 2 ** (1 / (2 * self.display_bands_per_octave))
        band_count = int(np.floor(
            self.display_bands_per_octave
            * np.log2(fft_freqs[-1] / self.display_split_hz)
        )) + 1
        display_centers = self.display_split_hz * 2 ** (
            np.arange(band_count) / self.display_bands_per_octave
        )
        band_edges = np.append(
            display_centers / half_band, display_centers[-1] * half_band
        )
        edge_index = np.searchsorted(fft_freqs, band_edges)
        edge_index[-1] = len(fft_freqs)
        self.display_edge_index = edge_index
        self.display_raw_count = int(edge_index[0])
        self.display_freqs = np.concatenate(
            (fft_freqs[:self.display_raw_count], display_centers)
        )
        self.raw_freqs = fft_freqs

        #live mode buffer
        self.analysis_seconds = 6
        self.analysis_size = 44100 * self.analysis_seconds
        self.analysis_ref_chunks = deque()
        self.analysis_meas_chunks = deque()
        self.analysis_chunk_samples = 0
        self.analysis_lock = threading.Lock()

        self.input_device = None
        self.reference_channel = 0
        self.measured_channel = 1
        self.reference_buffer = np.zeros(self.buffer_size, dtype = 'float32')
        self.show_audio_devices()
        self.refresh_devices()
        self.audio_status_message = None




    def compare_audio_files(self):
        self.original_file, _ = QFileDialog.getOpenFileName(
            self,
            'Select Reference Audio',
            '',
            'Audio Files (*.wav *.flac *.aiff);;All Files (*)'
        )    

        if not self.original_file:
            return

        self.comparison_file, _ = QFileDialog.getOpenFileName(
            self,
            'Select Comparison Audio',
            '',
            'Audio Files (*.wav *.flac *.aiff);;All Files (*)'
        )

        if not self.comparison_file:
            return

        result = vergleich.main(
            self.original_file,
            self.comparison_file
        )

        if result is None:
            return

        freqs_log, glatte_eq_kurve, modell_eq, eq_bands = result
 
        self.fill_eq_table(eq_bands)

        self.eq_table.resizeColumnsToContents()
        self.plot_widget.clear()
        self.plot_widget.enableAutoRange(axis = 'y')
        self.plot_widget.setLabel('bottom', 'Frequency', units = 'Hz')
        self.plot_widget.setLogMode(x = True, y = False)
        self.plot_widget.setXRange(np.log10(20), np.log10(20000), padding = 0)

        self.plot_widget.plot(
            freqs_log,
            glatte_eq_kurve,
            pen = pg.mkPen('g', width = 2),
            name = 'Correction EQ'
        )

        self.plot_widget.plot(
            freqs_log,
            modell_eq,
            pen = pg.mkPen('b', width = 2, style = pg.QtCore.Qt.PenStyle.DashLine),
            name = 'Otimized EQ'
        )


    def push_audio(self, reference, measured):
        n = len(measured)
        self.samples_recived += n

        with self.analysis_lock:
            self.analysis_ref_chunks.append(
                np.array(reference, dtype = 'float32')
            )
            self.analysis_meas_chunks.append(
                np.array(measured, dtype = 'float32')
            )
            self.analysis_chunk_samples += n

            while (
                self.analysis_chunk_samples - len(self.analysis_ref_chunks[0])
                >= self.analysis_size
            ):
                self.analysis_chunk_samples -= len(self.analysis_ref_chunks[0])
                self.analysis_ref_chunks.popleft()
                self.analysis_meas_chunks.popleft()

        self.reference_buffer = np.roll(self.reference_buffer, -n)
        self.reference_buffer[-n:] = reference
        self.ring_buffer = np.roll(self.ring_buffer, -n)
        self.ring_buffer[-n:] = measured

    def audio_callback(self, indata, frames, time, status):
        if status:
            self.audio_status_message = str(status)

        self.push_audio(
            indata[:, self.reference_channel],
            indata[:, self.measured_channel]
        )

    def start_live_analysis(self):

        try:
            if self.audio_stream is not None:
                return

            if self.input_device is None:
                self.show_message('no input device with 2 or more channels', True)
                return

            reference_channel = self.reference_chanel_box.currentIndex()
            measured_channel = self.measured_chanel_box.currentIndex()

            if reference_channel == measured_channel:
                self.show_message('reference and measured channel msut be different')
                return

            self.reference_channel = reference_channel
            self.measured_channel = measured_channel

            self.prepare_live_plot()
            
            self.audio_stream = sd.InputStream(
            samplerate = 44100,
            channels = max(reference_channel, measured_channel) + 1,
            device = self.input_device,
            dtype = 'float32',
            callback = self.audio_callback
            )
            

            self.audio_stream.start()
            self.live_timer.start()
            self.eq_timer.start()


        except Exception as e:
            self.show_message('audio input error: {e}', True )
            QMessageBox.warning(self, 'Audio input error', str(e))


    def stop_live_analysis(self):
        self.live_timer.stop()
        self.eq_timer.stop()

        if self.audio_stream is not None:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None 

        self.show_message('Live analysis stopped')

        self.test_timer.stop()
        self.test_mode = False
        

    def show_audio_devices(self):
        print(sd.query_devices())

    def update_live_analysis(self):
        try:
            if self.audio_stream is None and not self.test_mode:
                return

            if self.samples_recived < self.buffer_size:
                return
            
            audio_data = self.ring_buffer.copy()
            self.update_spectrum(audio_data)

            ref_freqs, ref_db = self.compute_spectrum(
                self.reference_buffer.copy()
            )
            in_range = (ref_freqs >= 20) & (ref_freqs <= 20000)
            self.reference_live_curve.setData(
                ref_freqs[in_range], ref_db[in_range]
            )

            difference_db = ref_db - self.live_spectrum_db
            self.diff_curve.setData(
                ref_freqs[in_range], difference_db[in_range]
            )

            band_db = self.band_average(ref_freqs, difference_db)

            if self.diff_smooth_db is None:
                self.diff_smooth_db = band_db
            else:
                alpha = self.diff_smooth_alpha
                self.diff_smooth_db = (
                    alpha * band_db + (1 - alpha) * self.diff_smooth_db
                )

            self.diff_smooth_curve.setData(
                self.band_centers, self.diff_smooth_db
            )

            self.lastest_peak = float(np.max(np.abs(audio_data)))
            self.lastest_rms = float(np.sqrt(np.mean(audio_data ** 2)))

            pass

        except Exception as e:
            self.show_message(f'live analysis error: {e}', True)

    def update_spectrum(self, audio_data):
        audio_data = np.asarray(audio_data).flatten()

        if len(audio_data) == 0:
            return

        freqs, spectrum_db = self.compute_spectrum(audio_data)

        self.live_spectrum_db = spectrum_db

        maske = (freqs >= 20) & (freqs <= 20000)

        aktuell = spectrum_db[maske]

        if self.peak_hold_db is None:
            self.peak_hold_db = aktuell.copy()
        else:
            dt = self.live_timer.interval() / 1000
            decay = self.peak_decay_db_per_s * dt
            falling = self.peak_hold_db - decay

            kernel_size = self.peak_smooth_bins
            padded = np.pad(falling, kernel_size // 2, mode = 'edge')
            smoothed = np.convolve(
                padded, np.ones(kernel_size) / kernel_size, mode = 'valid'
            )
            smoothed = falling + self.peak_smooth_strength * (smoothed - falling)
            falling = np.minimum(falling, smoothed)
            self.peak_hold_db = np.maximum(falling, aktuell)

            
        self.peak_curve.setData(freqs[maske], self.peak_hold_db)

        self.spectrum_curve.setData(
            freqs[maske], 
            spectrum_db[maske]
        )

    def load_mono(self, path):
        data, sample_rate = sf.read(path, dtype = 'float32')

        if data.ndim > 1:
            data = data.mean(axis = 1)

        if sample_rate != 44100:
            new_length = int(len(data) * 44100 / sample_rate)
            data = np.interp(
                np.linspace(0, len(data) - 1, new_length),
                np.arange(len(data)),
                data
            ).astype('float32')

        return data
    def load_test_files(self):
        reference_path, _= QFileDialog.getOpenFileName(
            self, 'Select Reference File (in1)', '',
            'Audio Files (*.wav *.flac *.aiff);;All Files(*)'
        )
        measured_path, _= QFileDialog.getOpenFileName(
            self, 'Select  File (in2)', '',
            'Audio Files (*.wav *.flac *.aiff);;All Files(*)'
        )
        if not reference_path or not measured_path:
            return

        self.test_reference = self.load_mono(reference_path)
        self.test_measured = self.load_mono(measured_path)

        common_lenght = min(len(self.test_reference), len(self.test_measured))
        self.test_reference = self.test_reference[:common_lenght]
        self.test_measured = self.test_measured[:common_lenght]
        self.test_position = 0

        self.show_message('test files loaded')

    def feed_test_audios(self):
        n = int(44100 * self.test_timer.interval() / 1000)
        steps = self.test_position + np.arange(n)

        reference_chunk = self.test_reference[steps % len(self.test_reference)]
        measured_chunk = self.test_measured[steps % len(self.test_measured)]
        self.test_position += n

        self.push_audio(reference_chunk, measured_chunk)

    def start_test_playback(self):
        if self.test_reference is None or self.test_measured is None:
            self.show_message('load test files first', True)
            return

        self.stop_live_analysis()

        self.prepare_live_plot()

        self.test_mode = True
        self.test_timer.start()
        self.live_timer.start()
        self.eq_timer.start()

    def compute_spectrum(self, audio_data):
        audio_data = np.asarray(audio_data).flatten()
        window = np.hanning(len(audio_data))
        spectrum = np.abs(np.fft.rfft(audio_data * window)) / (np.sum(window) / 2)
        freqs = np.fft.rfftfreq(len(audio_data), d = 1 / 44100)
        spectrum_db = 20 * np.log10(np.maximum(spectrum, 1e-12))

        if not self.display_smoothing_on:
            return self.raw_freqs, spectrum_db
        

        raw_count = self.display_raw_count
        power = spectrum[raw_count:] ** 2
        starts = self.display_edge_index[:-1] - raw_count
        counts = np.maximum(np.diff(self.display_edge_index), 1)
        band_power = np.add.reduceat(power, starts) / counts
        band_db = 10 * np.log10(np.maximum(band_power, 1e-24))


        return self.display_freqs, np.concatenate(
            (spectrum_db[:raw_count], band_db)
        )

    def toggle_smoothing(self, checked):
        self.peak_hold_db = None
        self.display_smoothing_on = checked
        self.smooth_button.setText('Smooth: on' if checked else 'Smooth : off')

    def toggle_settings(self, checked):
        self.settings_panel.setVisible(checked)

    def close_settings(self):
        self.settings_panel.setVisible(False)
        self.settings_button.setChecked(False)

    def show_message(self, text, error = False):
        print(text)
        self.statusBar().setStyleSheet(
            'color: #f87171' if error else 'color: #9299A8'
        )
        self.statusBar().showMessage(text, 15000 if error else 6000)
    
    def prepare_live_plot(self):
        self.plot_widget.clear()
        self.plot_widget.setYRange(-140, 0, padding = 0)
        self.peak_hold_db = None
        self.diff_smooth_db = None
        self.previous_curve = None
        self.fixed_delay_ms = None
        self.samples_recived = 0
        self.analysis_ref_chunks.clear()
        self.analysis_meas_chunks.clear()
        self.analysis_chunk_samples = 0
        self.reference_live_curve = self.plot_widget.plot(
            pen = pg.mkPen(
                '#f59e0b',
                width = 1
            ),
            name = 'Reference (in 1)'
        )
        self.peak_curve = self.plot_widget.plot(
            pen = pg.mkPen('#8b5cf6', width = 1),
            name = 'Peak hold'
        )
        self.spectrum_curve = self.plot_widget.plot(
            pen = pg.mkPen('c', width = 1),
            name = 'Measured (in 2)'
        )

    def band_average(self, freqs, values_db):
        result = np.zeros(len(self.band_centers))

        for i, center in enumerate(self.band_centers):
            low = center / 2 ** (1 / 12)
            high = center * 2 ** (1 / 12)
            selected = (freqs >= low) & (freqs < high)

            if np.any(selected):
                result[i] = np.mean(values_db[selected])
            else:
                result[i] = np.interp(center, freqs, values_db)

        return result

    def style_plot(self, plot, title):
        plot.setBackground('#15171c')
        plot.setTitle(title, color = '#E8EAF0', size = '11pt')

        for axis_name in ('bottom', 'left'):
            axis = plot.getAxis(axis_name)
            axis.setPen(pg.mkPen('#414754'))
            axis.setTextPen(pg.mkPen('#9299A8'))

    def fill_eq_table(self, eq_bands):
        self.eq_table.setRowCount(len(eq_bands))

        for i, (freqs, gain, q) in enumerate(eq_bands):
            self.eq_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.eq_table.setItem(i, 1, QTableWidgetItem(f"{freqs:.1f}"))
            self.eq_table.setItem(i, 2, QTableWidgetItem(f"{gain:+.2f}"))
            self.eq_table.setItem(i, 3, QTableWidgetItem(f"{q:.2f}"))


    def update_live_eq(self):
        try:
            if self.eq_worker is not None and self.eq_worker.isRunning():
                return

            block = self.get_analysis_block()

            if block is None:
                return

            if self.audio_status_message is not None:
                self.show_message(f"Audio: {self.audio_status_message}", True)
                self.audio_status_message = None

            reference_db = 20 * np.log10(max(np.sqrt(np.mean(block[0] ** 2)), 1e-12))
            measured_db = 20 * np.log10(max(np.sqrt(np.mean(block[0] ** 2)), 1e-12))

            if reference_db < self.min_level_db:
                self.show_message(f"reference singnal too quied ({reference_db:.0f} dBFS)", True)
                return

            if measured_db < self.min_level_db:
                self.show_message(f"microphone signal too quied ({measured_db:.0f} dBFS)", True)
                return

            self.eq_worker = EqWorker(self.eq_pool, block[0], block[1], self.previous_curve, self.fixed_delay_ms)
            self.eq_worker.result_ready.connect(self.show_live_eq)
            self.eq_worker.failed.connect(self.on_eq_failed)
            self.eq_worker.start()

        except Exception as e:
            print('live eq error:', e)
            traceback.print_exc()

    def on_eq_failed(self, message):
        print(message)
        last_line = message.strip().splitlines()[-1]
        self.show_message(f"EQ calculation failed: {last_line}", True)


    def closeEvent(self, event):
        self.stop_live_analysis()
        if self.eq_worker is not None:
            self.eq_worker.wait()

        self.eq_pool.shutdown(wait = False, cancel_futures = True)

        event.accept()

    def show_live_eq(self, result):
        eq_bands, freqs_log, smooth_curve, delay_ms = result
        self.previous_curve = smooth_curve

        self.analysis_curve.setData(freqs_log, smooth_curve)
        if self.fixed_delay_ms is None:
            self.fixed_delay_ms = delay_ms
            self.show_message(f"delay fixed at {delay_ms:.1f} ms")

        if eq_bands:
            model_params = np.array([
                value
                for freq, gain, q in eq_bands
                for value in (np.log10(freq), gain, np.log10(q))
            ])
            model_curve = vergleich.combined_eq_response(
                freqs_log, 44100, model_params, len(eq_bands)
            )
            self.model_curve.setData(freqs_log, model_curve)
        else:
            self.model_curve.setData([], [])

        strong_bands = [
            band for band in eq_bands
            if abs(band[1]) >= self.live_min_db
        ]
        self.fill_eq_table(strong_bands)

    def get_analysis_block(self):
        with self.analysis_lock:
            if self.analysis_chunk_samples < self.analysis_size:
                return None

            reference = np.concatenate(self.analysis_ref_chunks)
            measured = np.concatenate(self.analysis_meas_chunks)

        return reference[-self.analysis_size:], measured[-self.analysis_size:]

    def refresh_devices(self):
        self.input_devices = []
        self.device_box.blockSignals(True)
        self.device_box.clear()

        for index, device in enumerate(sd.query_devices()):
            if device['max_input_channels'] >= 2:
                self.input_devices.append(index)
                self.device_box.addItem(device['name'])

        self.device_box.blockSignals(False)

        if self.input_devices:
            self.device_box.setCurrentIndex(0)
            self.device_changed(0)
        else:
            self.input_device = None
            self.device_info_label.setText(
                'No input device with 2 or more channels found'
            )

    def device_changed(self, box_index):
        if box_index < 0 or box_index >= len(self.input_devices):
            return

        self.input_device = self.input_devices[box_index]
        device = sd.query_devices(self.input_device)
        channel_count = device['max_input_channels']

        for box in (self.reference_chanel_box, self.measured_chanel_box):
            box.clear()
            box.addItems([str(number) for number in range(1, channel_count + 1)])

        self.reference_chanel_box.setCurrentIndex(0)
        self.measured_chanel_box.setCurrentIndex(1)
        self.device_info_label.setText(
            f"{channel_count} inputs, {device['default_samplerate']:.0f} Hz"
        )
        

    
if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = eq_generator()
    window.show()

    sys.exit(app.exec())
