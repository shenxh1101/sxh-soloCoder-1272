import numpy as np
from scipy.signal import welch
from scipy.signal.windows import kaiser
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox,
    QDoubleSpinBox, QLabel, QSizePolicy,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT


WINDOW_FUNCTIONS = {
    "Hanning": np.hanning,
    "Hamming": np.hamming,
    "Blackman": np.blackman,
}

FFT_SIZES = ["Auto", "256", "512", "1024", "2048", "4096", "8192", "16384", "32768", "65536"]

MAX_FFT_SAMPLES = 262144


class SpectrumWorker(QThread):
    result_ready = pyqtSignal(str, np.ndarray, np.ndarray)

    def __init__(self, signal, sr, window_name, beta, nfft, view):
        super().__init__()
        self.signal = signal
        self.sr = sr
        self.window_name = window_name
        self.beta = beta
        self.nfft = nfft
        self.view = view

    def run(self):
        n_samples = len(self.signal)
        window = self._get_window(n_samples)
        nfft = self.nfft

        if self.view == "Magnitude":
            windowed = self.signal * window
            fft_result = np.fft.rfft(windowed, n=nfft)
            magnitude_db = 20 * np.log10(np.abs(fft_result) + 1e-12)
            freqs = np.fft.rfftfreq(nfft, d=1.0 / self.sr)
            step = max(1, len(freqs) // 8000)
            self.result_ready.emit("Magnitude", freqs[::step], magnitude_db[::step])
        elif self.view == "Phase":
            windowed = self.signal * window
            fft_result = np.fft.rfft(windowed, n=nfft)
            phase = np.unwrap(np.angle(fft_result))
            freqs = np.fft.rfftfreq(nfft, d=1.0 / self.sr)
            step = max(1, len(freqs) // 8000)
            self.result_ready.emit("Phase", freqs[::step], phase[::step])
        elif self.view == "Power Spectral Density (PSD)":
            nperseg = min(n_samples, 8192)
            win = window[:nperseg] if len(window) >= nperseg else window
            freqs_w, psd = welch(
                self.signal, fs=self.sr, window=win,
                nperseg=nperseg, nfft=max(nfft, nperseg), scaling="density",
            )
            psd_db = 10 * np.log10(psd + 1e-12)
            self.result_ready.emit("PSD", freqs_w, psd_db)

    def _get_window(self, n):
        if self.window_name == "Kaiser":
            return kaiser(n, self.beta)
        return WINDOW_FUNCTIONS.get(self.window_name, np.hanning)(n)


class SpectrumPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._audio_data = None
        self._worker = None
        self._pending_view = None
        self._setup_ui()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        controls = QHBoxLayout()

        controls.addWidget(QLabel("View:"))
        self._view_combo = QComboBox()
        self._view_combo.addItems(["Magnitude", "Phase", "Power Spectral Density (PSD)"])
        self._view_combo.currentIndexChanged.connect(self._refresh)
        controls.addWidget(self._view_combo)

        controls.addWidget(QLabel("Window:"))
        self._window_combo = QComboBox()
        self._window_combo.addItems(list(WINDOW_FUNCTIONS.keys()) + ["Kaiser"])
        self._window_combo.currentIndexChanged.connect(self._on_window_changed)
        controls.addWidget(self._window_combo)

        self._beta_label = QLabel("Beta:")
        self._beta_label.setVisible(False)
        controls.addWidget(self._beta_label)
        self._beta_spin = QDoubleSpinBox()
        self._beta_spin.setRange(0.0, 100.0)
        self._beta_spin.setValue(14.0)
        self._beta_spin.setSingleStep(0.5)
        self._beta_spin.setVisible(False)
        self._beta_spin.valueChanged.connect(self._refresh)
        controls.addWidget(self._beta_spin)

        controls.addWidget(QLabel("FFT Size:"))
        self._fft_combo = QComboBox()
        self._fft_combo.addItems(FFT_SIZES)
        self._fft_combo.setCurrentIndex(0)
        self._fft_combo.currentIndexChanged.connect(self._refresh)
        controls.addWidget(self._fft_combo)

        controls.addWidget(QLabel("Channel:"))
        self._channel_combo = QComboBox()
        self._channel_combo.currentIndexChanged.connect(self._refresh)
        controls.addWidget(self._channel_combo)

        controls.addStretch()
        root.addLayout(controls)

        self._figure = Figure(facecolor="#1e1e1e")
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #ffcc00; font-size: 11px; padding: 2px 6px;")
        self._status_label.setVisible(False)

        root.addWidget(self._toolbar)
        root.addWidget(self._canvas)
        root.addWidget(self._status_label)

    def _on_window_changed(self, index):
        is_kaiser = self._window_combo.currentText() == "Kaiser"
        self._beta_label.setVisible(is_kaiser)
        self._beta_spin.setVisible(is_kaiser)
        self._refresh()

    def _get_window(self, n):
        name = self._window_combo.currentText()
        if name == "Kaiser":
            return kaiser(n, self._beta_spin.value())
        return WINDOW_FUNCTIONS[name](n)

    def _get_nfft(self, n_samples):
        text = self._fft_combo.currentText()
        if text == "Auto":
            effective_n = min(n_samples, MAX_FFT_SAMPLES)
            n = 1
            while n < effective_n:
                n <<= 1
            return n
        return int(text)

    def set_data(self, audio_data):
        self._audio_data = audio_data
        self._channel_combo.blockSignals(True)
        self._channel_combo.clear()
        for i in range(audio_data.num_channels):
            label = audio_data.channel_names[i] if i < len(audio_data.channel_names) else f"Channel {i}"
            self._channel_combo.addItem(label)
        self._channel_combo.blockSignals(False)
        self._refresh()

    def _refresh(self):
        if self._audio_data is None:
            return
        ch_idx = self._channel_combo.currentIndex()
        if ch_idx < 0:
            return
        signal = self._audio_data.get_channel(ch_idx)
        sr = self._audio_data.sample_rate
        n_samples = len(signal)
        view = self._view_combo.currentText()

        if n_samples > MAX_FFT_SAMPLES and view != "Power Spectral Density (PSD)":
            signal = signal[-MAX_FFT_SAMPLES:]
            n_samples = len(signal)

        nfft = self._get_nfft(n_samples)
        window_name = self._window_combo.currentText()
        beta = self._beta_spin.value()

        self._show_computing(True)

        if self._worker is not None and self._worker.isRunning():
            self._worker.quit()
            self._worker.wait(100)

        self._worker = SpectrumWorker(signal, sr, window_name, beta, nfft, view)
        self._worker.result_ready.connect(self._on_result_ready)
        self._worker.start()

    def _show_computing(self, show):
        if show:
            self._status_label.setText("Computing spectrum…")
            self._status_label.setVisible(True)
        else:
            self._status_label.setVisible(False)

    def _on_result_ready(self, view, x_data, y_data):
        self._show_computing(False)
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        ax.set_facecolor("#2d2d2d")
        ax.tick_params(colors="white", which="both")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("#555555")

        if view == "Magnitude":
            ax.plot(x_data, y_data, color="#00ccff", linewidth=0.8)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Magnitude (dB)")
            ax.set_title("Magnitude Spectrum")
        elif view == "Phase":
            ax.plot(x_data, y_data, color="#ff9900", linewidth=0.8)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Phase (radians)")
            ax.set_title("Phase Spectrum")
        elif view == "PSD":
            ax.plot(x_data, y_data, color="#66ff66", linewidth=0.8)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("PSD (dB/Hz)")
            ax.set_title("Power Spectral Density")

        ax.grid(True, which="both", linestyle="--", linewidth=0.5, color="#555555", alpha=0.7)
        self._figure.tight_layout()
        self._canvas.draw()

    def export_csv_data(self):
        if self._audio_data is None:
            raise ValueError("No data loaded")
        ch_idx = self._channel_combo.currentIndex()
        if ch_idx < 0:
            raise ValueError("No channel selected")
        signal = self._audio_data.get_channel(ch_idx)
        sr = self._audio_data.sample_rate
        n_samples = len(signal)
        if n_samples > MAX_FFT_SAMPLES:
            signal = signal[-MAX_FFT_SAMPLES:]
            n_samples = len(signal)
        nfft = self._get_nfft(n_samples)
        window = self._get_window(n_samples)
        windowed = signal * window
        fft_result = np.fft.rfft(windowed, n=nfft)
        freqs = np.fft.rfftfreq(nfft, d=1.0 / sr)
        view = self._view_combo.currentText()
        headers = ["Frequency(Hz)"]
        arrays = [freqs]
        if view == "Magnitude":
            magnitude_db = 20 * np.log10(np.abs(fft_result) + 1e-12)
            headers.append("Magnitude(dB)")
            arrays.append(magnitude_db)
        elif view == "Phase":
            phase = np.unwrap(np.angle(fft_result))
            headers.append("Phase(rad)")
            arrays.append(phase)
        elif view == "Power Spectral Density (PSD)":
            nperseg = min(n_samples, 8192)
            freqs_w, psd = welch(
                signal, fs=sr,
                window=window[:nperseg] if len(window) >= nperseg else window,
                nperseg=nperseg, nfft=max(nfft, nperseg), scaling="density",
            )
            psd_db = 10 * np.log10(psd + 1e-12)
            headers = ["Frequency(Hz)", "PSD(dB/Hz)"]
            arrays = [freqs_w, psd_db]
        return arrays, headers

    def get_figure(self):
        return self._figure
