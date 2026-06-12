import numpy as np
import pywt
from scipy.signal import stft
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QSpinBox,
    QDoubleSpinBox, QLabel, QFormLayout, QGroupBox, QSizePolicy, QPushButton,
)
from PyQt6.QtCore import Qt, pyqtSignal
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from results_manager import AnalysisResult


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
    save_result_requested = pyqtSignal(dict, dict, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.audio_data = None
        self._cbar = None
        self._last_result = None
        self._analysis_range = None
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
        self.wav_family_combo.addItems([
            "morl", "cmor1.5-1.0", "cmor3.0-1.0",
            "mexh", "cgau1", "cgau2", "cgau3", "cgau4", "cgau5", "cgau6", "cgau7", "cgau8",
        ])
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

        self._save_btn = QPushButton("💾 Save")
        self._save_btn.setFixedHeight(24)
        self._save_btn.setStyleSheet("""
            QPushButton { background-color: #0e639c; color: white; border: none;
            border-radius: 3px; padding: 4px 12px; font-size: 11px; }
            QPushButton:hover { background-color: #1177bb; }
        """)
        self._save_btn.clicked.connect(self._on_save_result)
        ctrl_row.addWidget(self._save_btn)

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
        self._analysis_range = None
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
            max_init = min(30.0, duration)
            self.t_end_spin.setValue(max_init)
        self.ch_combo.blockSignals(False)
        self._recompute()

    def set_analysis_range(self, t0, t1):
        if self.audio_data is None:
            return
        self._analysis_range = (t0, t1)
        self.t_start_spin.blockSignals(True)
        self.t_end_spin.blockSignals(True)
        self.t_start_spin.setValue(t0)
        self.t_end_spin.setValue(t1)
        self.t_start_spin.blockSignals(False)
        self.t_end_spin.blockSignals(False)
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
        win_map = {"hanning": "hann", "hamming": "hamming", "blackman": "blackman"}
        window = win_map.get(win_name, "hann")

        freqs, times, Zxx = stft(signal, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
        power = np.abs(Zxx) ** 2
        power_db = 10.0 * np.log10(power + 1e-12)
        times = times + t_offset

        self._last_result = {
            "type": "stft",
            "times": times.copy(),
            "freqs": freqs.copy(),
            "power_db": power_db.copy(),
            "params": {
                "nperseg": nperseg,
                "noverlap": noverlap,
                "window": window,
            },
        }

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

        try:
            coeffs, freqs = pywt.cwt(signal, scales, wavelet, 1.0 / fs)
        except Exception as e:
            self.ax.text(0.5, 0.5, f"Wavelet error: {e}", transform=self.ax.transAxes,
                         ha="center", va="center", color="#f38ba8", fontsize=11)
            return

        power = np.abs(coeffs) ** 2
        power_db = 10.0 * np.log10(power + 1e-12)
        duration = len(signal) / fs
        times = np.linspace(t_offset, t_offset + duration, power_db.shape[1])

        self._last_result = {
            "type": "wavelet",
            "times": times.copy(),
            "freqs": freqs.copy(),
            "power_db": power_db.copy(),
            "params": {
                "wavelet": wavelet,
                "scales": scales.copy(),
            },
        }

        pcm = self.ax.pcolormesh(times, freqs, power_db, shading="gouraud", cmap="magma")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Frequency (Hz)")
        self.ax.set_title("Wavelet Scalogram")
        self._cbar = self.figure.colorbar(pcm, ax=self.ax, pad=0.02)
        self._cbar.set_label("Power (dB)", color="#cdd6f4")
        self._cbar.ax.yaxis.set_tick_params(color="#cdd6f4")
        for label in self._cbar.ax.get_yticklabels():
            label.set_color("#cdd6f4")

    def _on_save_result(self):
        if self._last_result is None:
            return
        res = self._last_result
        ch_idx = self.ch_combo.currentData()
        ch_name = self.ch_combo.currentText()
        data = {
            "type": res["type"],
            "times": res["times"],
            "freqs": res["freqs"],
            "power_db": res["power_db"],
            "channel": ch_name,
        }
        params = {
            "params": res["params"],
            "range": str(self._analysis_range) if self._analysis_range else "full",
        }
        mode = "STFT" if res["type"] == "stft" else "Wavelet"
        self.save_result_requested.emit(data, params, f"{mode} {ch_name}")

    def restore_result(self, result: AnalysisResult):
        d = result.data
        rtype = d.get("type", "stft")
        times = np.array(d.get("times", []))
        freqs = np.array(d.get("freqs", []))
        power_db = np.array(d.get("power_db", []))
        self._last_result = {
            "type": rtype,
            "times": times,
            "freqs": freqs,
            "power_db": power_db,
            "params": d.get("params", {}),
        }

        self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentIndex(0 if rtype == "stft" else 1)
        self.mode_combo.blockSignals(False)
        self.stft_box.setVisible(rtype == "stft")
        self.wavelet_box.setVisible(rtype != "stft")

        if d.get("channel"):
            idx = self.ch_combo.findText(d["channel"])
            if idx >= 0:
                self.ch_combo.blockSignals(True)
                self.ch_combo.setCurrentIndex(idx)
                self.ch_combo.blockSignals(False)

        params = d.get("params", {})
        if rtype == "stft":
            if "nperseg" in params:
                idx = self.win_size_combo.findText(str(params["nperseg"]))
                if idx >= 0:
                    self.win_size_combo.blockSignals(True)
                    self.win_size_combo.setCurrentIndex(idx)
                    self.win_size_combo.blockSignals(False)
            if "noverlap" in params and "nperseg" in params and params["nperseg"] > 0:
                pct = int(params["noverlap"] / params["nperseg"] * 100)
                self.overlap_spin.blockSignals(True)
                self.overlap_spin.setValue(pct)
                self.overlap_spin.blockSignals(False)
            if "window" in params:
                wmap = {"hann": "Hanning", "hamming": "Hamming", "blackman": "Blackman"}
                wname = wmap.get(params["window"], "Hanning")
                idx = self.win_func_combo.findText(wname)
                if idx >= 0:
                    self.win_func_combo.blockSignals(True)
                    self.win_func_combo.setCurrentIndex(idx)
                    self.win_func_combo.blockSignals(False)
        else:
            if "wavelet" in params:
                idx = self.wav_family_combo.findText(params["wavelet"])
                if idx >= 0:
                    self.wav_family_combo.blockSignals(True)
                    self.wav_family_combo.setCurrentIndex(idx)
                    self.wav_family_combo.blockSignals(False)
            if "scales" in params and len(params["scales"]) >= 2:
                s = np.array(params["scales"])
                self.scale_min_spin.blockSignals(True)
                self.scale_max_spin.blockSignals(True)
                self.scale_min_spin.setValue(int(s.min()))
                self.scale_max_spin.setValue(int(s.max()))
                self.scale_min_spin.blockSignals(False)
                self.scale_max_spin.blockSignals(False)

        self.ax.clear()
        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None
        cmap = "viridis" if rtype == "stft" else "magma"
        pcm = self.ax.pcolormesh(times, freqs, power_db, shading="gouraud", cmap=cmap)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Frequency (Hz)")
        self.ax.set_title("STFT Spectrogram" if rtype == "stft" else "Wavelet Scalogram")
        self._cbar = self.figure.colorbar(pcm, ax=self.ax, pad=0.02)
        self._cbar.set_label("Power (dB)", color="#cdd6f4")
        self._cbar.ax.yaxis.set_tick_params(color="#cdd6f4")
        for label in self._cbar.ax.get_yticklabels():
            label.set_color("#cdd6f4")
        self._style_ax()
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def export_csv_data(self):
        if self._last_result is None:
            raise ValueError("No time-frequency data computed yet. Wait for the analysis to finish first.")
        res = self._last_result
        times = res["times"]
        freqs = res["freqs"]
        power_db = res["power_db"]
        nf, nt = power_db.shape

        if res["type"] == "stft":
            w = res["params"]["window"]
            header_row1 = ["# Type: STFT Spectrogram",
                           f"# Window: {w}",
                           f"# Shape: Freqs={nf} x Times={nt}",
                           "# Format: Each row = [Time(s), Freq(Hz), Power(dB)]"]
            headers = ["Time(s)", "Frequency(Hz)", "Power(dB)"]
        else:
            wv = res["params"]["wavelet"]
            header_row1 = ["# Type: Wavelet Scalogram",
                           f"# Wavelet: {wv}",
                           f"# Shape: Freqs={nf} x Times={nt}",
                           "# Format: Each row = [Time(s), Freq(Hz), Power(dB)]"]
            headers = ["Time(s)", "Frequency(Hz)", "Power(dB)"]

        t_col = np.repeat(times, nf)
        f_col = np.tile(freqs, nt)
        p_col = power_db.flatten()

        return [t_col, f_col, p_col], headers

    def get_last_result(self):
        return self._last_result

    def get_figure(self):
        return self.figure
