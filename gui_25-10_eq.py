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
    QHeaderView
    
)
from PySide6.QtGui import QFont
from PySide6.QtCore import (
    QTimer, QThread, Signal
)

import sounddevice as sd
import numpy as np
import pyqtgraph as pg
import soundfile as sf
import time


import vergleich

#print(dir(vergleich))

import inspect
#print(inspect.getsource(vergleich.optimize_eq))
#print(inspect.getsource(vergleich.main))

#Notiz an mich selbst, die rechenleistung des programmen reduzieren, und effizienter machen, dazu mal rechaschieren

class EqWorker(QThread):
    result_ready = Signal(object)
    failed = Signal(str)

    def __init__(self, freqs_log, curve):
        super().__init__()
        self.freqs_log = freqs_log
        self.curve = curve

    def run(self):
        try:
            start_time = time.perf_counter()

            eq_bands = vergleich.optimize_eq(
                self.freqs_log,
                self.curve,
                44100,
                bands = vergleich.bands,
                max_gain_db = 20.0,
                q_min = 0.2,
                q_max = 8.0,
                min_gain_db = vergleich.min_gain_db
            )

            print(f'eq fit took {time.perf_counter() - start_time:.2f} s')
            print('eq result:', eq_bands)

            self.result_ready.emit(eq_bands or [])

        except Exception as e:
            self.failed.emit(str(e))           




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
        subtitel.setStyleSheet
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

        sidebar.addWidget(titel)
        sidebar.addWidget(subtitel)
        sidebar.addSpacing(20)

        sidebar.addWidget(self.live_button)
        sidebar.addWidget(self.stop_live_button)
        sidebar.addWidget(self.recorded_button)
        sidebar.addWidget(self.test_files_button)
        sidebar.addWidget(self.test_start_button)
        sidebar.addStretch()


        self.setCentralWidget(zentral)
        self.live_timer = QTimer(self)
        self.live_timer.setInterval(100)
        self.live_timer.timeout.connect(self.update_live_analysis)
        self.eq_timer = QTimer(self)
        self.eq_timer.setInterval(2000)
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
            'band', 'Frequency (Hz)', 'Gain (dB)', 'Q'
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
        self.diff_plot.setLabel('left', 'Diffrence', units = 'dB')
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

        self.style_plot(self.diff_plot, 'Difference')

        self.eq_worker = None

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

        self.input_device = None
        self.reference_buffer = np.zeros(self.buffer_size, dtype = 'float32')
        self.show_audio_devices()

        self.reference_freqs = None
        self.reference_db = None


    def select_live_audio(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            'Select Live Audio',
            '',
            'Audio Files (*.wav *.mp3);;All Files (*)'
        )

        if file_path:
            self.original_file = file
            print('Live Audio:', file_path)


    def compare_audio_files(self):
        self.original_file, _ = QFileDialog.getOpenFileName(
            self,
            'Select Reference Audio',
            '',
            'Audio Files (*.wav *.flac *.aiff);;All Files (*)'
        )    

        if not self.original_file:
            return

        original_file = self.original_file

        self.comparison_file, _ = QFileDialog.getOpenFileName(
            self,
            'Select Comparison Audio',
            '',
            'Audio Files (*.wav *.flac *.aiff);;All Files (*)'
        )

        comparison_file = self.comparison_file

        if not comparison_file:
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

        self.reference_buffer = np.roll(self.reference_buffer, -n)
        self.reference_buffer[-n:] = reference
        self.ring_buffer = np.roll(self.ring_buffer, -n)
        self.ring_buffer[-n:] = measured

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)

        self.push_audio(indata[:, 0], indata[:, 1])

    def start_live_analysis(self):

        try:
            if self.audio_stream is not None:
                return

            self.prepare_live_plot()
            
            self.audio_stream = sd.InputStream(
            samplerate = 44100,
            channels = 1,
            dtype = 'float32',
            callback = self.audio_callback
            )

            self.audio_stream.start()
            self.live_timer.start()
            self.eq_timer.start()


        except Exception as e:
            print('audio input error:', e)


    def stop_live_analysis(self):
        self.live_timer.stop()
        self.eq_timer.stop()

        if self.audio_stream is not None:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None 

        print('Live analysis stopped')

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
            print('live analysis error:', e)

    def update_spectrum(self, audio_data):
        audio_data = np.asarray(audio_data).flatten()

        if len(audio_data) == 0:
            return

        freqs, spectrum_db = self.compute_spectrum(audio_data)

        self.live_freqs = freqs
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

    def compute_reference_spectrum(self, path):
        data, sample_rate = sf.read(pa)

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
        if not measured_path:
            return

        self.test_reference = self.load_mono(reference_path)
        self.test_measured = self.load_mono(measured_path)
        self.test_position = 0

        print('test files loaded:', len(self.test_reference), len(self.test_measured))

    def feed_test_audios(self):
        n = int(44100 * self.test_timer.interval() / 1000)
        steps = self.test_position + np.arange(n)

        reference_chunk = self.test_reference[steps % len(self.test_reference)]
        measured_chunk = self.test_measured[steps % len(self.test_measured)]
        self.test_position += n

        self.push_audio(reference_chunk, measured_chunk)

    def start_test_playback(self):
        if self.test_reference is None or self.test_measured is None:
            print('load test files first')
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
        return freqs, spectrum_db

    def prepare_live_plot(self):
        self.plot_widget.clear()
        self.peak_hold_db = None
        self.diff_smooth_db = None
        self.samples_recived = 0
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
        if self.diff_smooth_db is None:
            return

        try:
            levels = self.diff_smooth_db.copy()

            offset_mask = (self.band_centers >= 100) & (self.band_centers <= 10000)
            levels = levels - np.median(levels[offset_mask])

            freqs_log = np.geomspace(20, 20000, 500)
            curve = np.interp(
                np.log10(freqs_log),
                np.log10(self.band_centers),
                levels
            )

            window_size = 5
            smooth_curve = np.convolve(
                curve, np.ones(window_size) / window_size, mode = 'same'
            )

            if self.eq_worker is not None and self.eq_worker.isRunning():
                return
            self.eq_worker = EqWorker(freqs_log, smooth_curve)
            self.eq_worker.result_ready.connect(self.fill_eq_table)
            self.eq_worker.failed.connect(
                lambda message: print('live eq error:', message)

            )
            self.eq_worker.start()

        except Exception as e:
           print('live eq error:', e)
 

    def closeEvent(self, event):
        self.stop_live_analysis()
        if self.eq_worker is not None:
            self.eq_worker.wait()

        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = eq_generator()
    window.show()

    sys.exit(app.exec())
