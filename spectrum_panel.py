import numpy as np
from scipy.signal import welch
from scipy.signal.windows import kaiser
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox,
    QDoubleSpinBox, QLabel, QSizePolicy,
)
from PyQt6.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT


WINDOW_FUNCTIONS = {
    "Hanning": np.hanning,
    "Hamming": np.hamming,
    "Blackman": np.blackman,
}

FFT_SIZES = ["Auto", "256", "512", "1024", "2048", "4096", "8192", "16384", "32768", "65536"]


class SpectrumPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._audio_data = None
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

        root.addWidget(self._toolbar)
        root.addWidget(self._canvas)

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
            n = 1
            while n < n_samples:
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
        nfft = self._get_nfft(n_samples)
        window = self._get_window(n_samples)
        view = self._view_combo.currentText()

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
            windowed = signal * window
            fft_result = np.fft.rfft(windowed, n=nfft)
            magnitude_db = 20 * np.log10(np.abs(fft_result) + 1e-12)
            freqs = np.fft.rfftfreq(nfft, d=1.0 / sr)
            step = max(1, len(freqs) // 8000)
            ax.plot(freqs[::step], magnitude_db[::step], color="#00ccff", linewidth=0.8)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Magnitude (dB)")
            ax.set_title("Magnitude Spectrum")
        elif view == "Phase":
            windowed = signal * window
            fft_result = np.fft.rfft(windowed, n=nfft)
            phase = np.unwrap(np.angle(fft_result))
            freqs = np.fft.rfftfreq(nfft, d=1.0 / sr)
            step = max(1, len(freqs) // 8000)
            ax.plot(freqs[::step], phase[::step], color="#ff9900", linewidth=0.8)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Phase (radians)")
            ax.set_title("Phase Spectrum")
        elif view == "Power Spectral Density (PSD)":
            nperseg = min(n_samples, 8192)
            freqs_w, psd = welch(
                signal,
                fs=sr,
                window=window[:nperseg] if len(window) >= nperseg else window,
                nperseg=nperseg,
                nfft=max(nfft, nperseg),
                scaling="density",
            )
            psd_db = 10 * np.log10(psd + 1e-12)
            ax.plot(freqs_w, psd_db, color="#66ff66", linewidth=0.8)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("PSD (dB/Hz)")
            ax.set_title("Power Spectral Density")

        ax.grid(True, which="both", linestyle="--", linewidth=0.5, color="#555555", alpha=0.7)
        self._figure.tight_layout()
        self._canvas.draw()

    def get_figure(self):
        return self._figure
