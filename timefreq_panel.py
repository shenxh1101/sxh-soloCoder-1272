import numpy as np
import pywt
from scipy.signal import stft
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QSpinBox,
    QDoubleSpinBox, QLabel, QFormLayout, QGroupBox, QSizePolicy,
)
from PyQt6.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT


DARK_STYLE = """
QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-size: 12px;
}
QGroupBox {
    border: 1px solid #45475a;
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 14px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}
QComboBox {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 3px;
    padding: 3px 6px;
    min-width: 100px;
}
QComboBox::drop-down {
    border: none;
}
QComboBox QAbstractItemView {
    background-color: #313244;
    color: #cdd6f4;
    selection-background-color: #585b70;
}
QSpinBox, QDoubleSpinBox {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 3px;
    padding: 3px 6px;
}
QLabel {
    color: #cdd6f4;
}
"""


class TimeFreqPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.audio_data = None
        self._cbar = None
        self.setStyleSheet(DARK_STYLE)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(10)

        mode_box = QGroupBox("Analysis Mode")
        mode_lay = QFormLayout(mode_box)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["STFT Spectrogram", "Wavelet Scalogram"])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_lay.addRow("Mode:", self.mode_combo)

        ch_box = QGroupBox("Channel")
        ch_lay = QFormLayout(ch_box)
        self.ch_combo = QComboBox()
        self.ch_combo.currentIndexChanged.connect(self._recompute)
        ch_lay.addRow("Channel:", self.ch_combo)

        self.stft_box = QGroupBox("STFT Parameters")
        stft_lay = QFormLayout(self.stft_box)
        self.win_size_combo = QComboBox()
        self.win_size_combo.addItems(["256", "512", "1024", "2048", "4096"])
        self.win_size_combo.setCurrentIndex(2)
        self.win_size_combo.currentIndexChanged.connect(self._recompute)
        stft_lay.addRow("Window size:", self.win_size_combo)

        self.overlap_spin = QSpinBox()
        self.overlap_spin.setRange(0, 95)
        self.overlap_spin.setValue(75)
        self.overlap_spin.setSuffix("%")
        self.overlap_spin.valueChanged.connect(self._recompute)
        stft_lay.addRow("Overlap:", self.overlap_spin)

        self.win_func_combo = QComboBox()
        self.win_func_combo.addItems(["Hanning", "Hamming", "Blackman"])
        self.win_func_combo.currentIndexChanged.connect(self._recompute)
        stft_lay.addRow("Window func:", self.win_func_combo)

        self.wavelet_box = QGroupBox("Wavelet Parameters")
        wav_lay = QFormLayout(self.wavelet_box)
        self.wav_family_combo = QComboBox()
        self.wav_family_combo.addItems(["morl", "cmor", "mexh", "cgau"])
        self.wav_family_combo.currentIndexChanged.connect(self._recompute)
        wav_lay.addRow("Wavelet:", self.wav_family_combo)

        self.scale_min_spin = QSpinBox()
        self.scale_min_spin.setRange(1, 512)
        self.scale_min_spin.setValue(1)
        self.scale_min_spin.valueChanged.connect(self._recompute)
        wav_lay.addRow("Scale min:", self.scale_min_spin)

        self.scale_max_spin = QSpinBox()
        self.scale_max_spin.setRange(2, 1024)
        self.scale_max_spin.setValue(128)
        self.scale_max_spin.valueChanged.connect(self._recompute)
        wav_lay.addRow("Scale max:", self.scale_max_spin)

        range_box = QGroupBox("Time Range")
        range_lay = QFormLayout(range_box)
        self.t_start_spin = QDoubleSpinBox()
        self.t_start_spin.setRange(0.0, 99999.0)
        self.t_start_spin.setValue(0.0)
        self.t_start_spin.setSuffix(" s")
        self.t_start_spin.setDecimals(3)
        self.t_start_spin.valueChanged.connect(self._recompute)
        range_lay.addRow("Start:", self.t_start_spin)

        self.t_end_spin = QDoubleSpinBox()
        self.t_end_spin.setRange(0.0, 99999.0)
        self.t_end_spin.setValue(0.0)
        self.t_end_spin.setSuffix(" s")
        self.t_end_spin.setDecimals(3)
        self.t_end_spin.valueChanged.connect(self._recompute)
        range_lay.addRow("End:", self.t_end_spin)

        ctrl_row.addWidget(mode_box)
        ctrl_row.addWidget(ch_box)
        ctrl_row.addWidget(self.stft_box)
        ctrl_row.addWidget(self.wavelet_box)
        ctrl_row.addWidget(range_box)
        ctrl_row.addStretch()

        self.figure = Figure(figsize=(8, 4), facecolor="#1e1e2e")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.ax = self.figure.add_subplot(111)
        self._style_ax()

        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        root.addLayout(ctrl_row)
        root.addWidget(self.toolbar)
        root.addWidget(self.canvas, stretch=1)

        self._on_mode_changed(0)

    def _style_ax(self):
        self.ax.set_facecolor("#1e1e2e")
        self.ax.tick_params(colors="#cdd6f4", which="both")
        self.ax.xaxis.label.set_color("#cdd6f4")
        self.ax.yaxis.label.set_color("#cdd6f4")
        self.ax.title.set_color("#cdd6f4")
        for spine in self.ax.spines.values():
            spine.set_color("#45475a")

    def _on_mode_changed(self, _index):
        is_stft = self.mode_combo.currentIndex() == 0
        self.stft_box.setVisible(is_stft)
        self.wavelet_box.setVisible(not is_stft)
        self._recompute()

    def set_data(self, audio_data):
        self.audio_data = audio_data
        self.ch_combo.blockSignals(True)
        self.ch_combo.clear()
        if audio_data is not None:
            for i in range(audio_data.num_channels):
                name = (
                    audio_data.channel_names[i]
                    if audio_data.channel_names and i < len(audio_data.channel_names)
                    else f"Channel {i}"
                )
                self.ch_combo.addItem(name, i)
            duration = len(audio_data.signal) / audio_data.sample_rate
            self.t_start_spin.setMaximum(duration)
            self.t_end_spin.setMaximum(duration)
            self.t_start_spin.setValue(0.0)
            self.t_end_spin.setValue(duration)
        self.ch_combo.blockSignals(False)
        self._recompute()

    def _recompute(self):
        if self.audio_data is None:
            return
        ch_idx = self.ch_combo.currentData()
        if ch_idx is None:
            return
        signal = self.audio_data.get_channel(ch_idx).astype(np.float64)
        fs = self.audio_data.sample_rate

        t_start = self.t_start_spin.value()
        t_end = self.t_end_spin.value()
        if t_end <= t_start:
            return
        i_start = int(t_start * fs)
        i_end = int(t_end * fs)
        i_start = max(0, min(i_start, len(signal)))
        i_end = max(i_start + 1, min(i_end, len(signal)))
        signal = signal[i_start:i_end]

        self.ax.clear()
        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None

        if self.mode_combo.currentIndex() == 0:
            self._plot_stft(signal, fs, t_start)
        else:
            self._plot_wavelet(signal, fs, t_start)

        self._style_ax()
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _plot_stft(self, signal, fs, t_offset):
        nperseg = int(self.win_size_combo.currentText())
        noverlap = int(nperseg * self.overlap_spin.value() / 100.0)
        win_name = self.win_func_combo.currentText().lower()
        if win_name == "hanning":
            window = "hann"
        elif win_name == "hamming":
            window = "hamm"
        else:
            window = "blackman"

        freqs, times, Zxx = stft(signal, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
        power = np.abs(Zxx) ** 2
        power_db = 10.0 * np.log10(power + 1e-12)
        times = times + t_offset

        pcm = self.ax.pcolormesh(times, freqs, power_db, shading="gouraud", cmap="viridis")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Frequency (Hz)")
        self.ax.set_title("STFT Spectrogram")
        self._cbar = self.figure.colorbar(pcm, ax=self.ax, pad=0.02)
        self._cbar.set_label("Power (dB)", color="#cdd6f4")
        self._cbar.ax.yaxis.set_tick_params(color="#cdd6f4")
        for label in self._cbar.ax.get_yticklabels():
            label.set_color("#cdd6f4")

    def _plot_wavelet(self, signal, fs, t_offset):
        wavelet = self.wav_family_combo.currentText()
        smin = self.scale_min_spin.value()
        smax = self.scale_max_spin.value()
        if smax <= smin:
            return
        scales = np.arange(smin, smax + 1)

        coeffs, freqs = pywt.cwt(signal, scales, wavelet, 1.0 / fs)
        power = np.abs(coeffs) ** 2
        power_db = 10.0 * np.log10(power + 1e-12)
        duration = len(signal) / fs
        times = np.linspace(t_offset, t_offset + duration, power_db.shape[1])

        pcm = self.ax.pcolormesh(times, freqs, power_db, shading="gouraud", cmap="magma")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Frequency (Hz)")
        self.ax.set_title("Wavelet Scalogram")
        self._cbar = self.figure.colorbar(pcm, ax=self.ax, pad=0.02)
        self._cbar.set_label("Power (dB)", color="#cdd6f4")
        self._cbar.ax.yaxis.set_tick_params(color="#cdd6f4")
        for label in self._cbar.ax.get_yticklabels():
            label.set_color("#cdd6f4")

    def get_figure(self):
        return self.figure
