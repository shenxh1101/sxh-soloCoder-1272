import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks, windows
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QDoubleSpinBox, QSpinBox,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
)
from PyQt6.QtCore import Qt, pyqtSignal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from results_manager import AnalysisResult


class AnalysisPanel(QWidget):
    save_result_requested = pyqtSignal(dict, dict, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.audio_data = None
        self._last_result = None
        self._analysis_range = None
        self._build_ui()
        self._apply_style()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        controls = QGridLayout()
        controls.setSpacing(6)

        row = 0
        controls.addWidget(QLabel("Channel:"), row, 0)
        self.channel_combo = QComboBox()
        self.channel_combo.setMinimumWidth(120)
        controls.addWidget(self.channel_combo, row, 1)

        controls.addWidget(QLabel("Window:"), row, 2)
        self.window_combo = QComboBox()
        self.window_combo.addItems(["Hanning", "Hamming", "Blackman", "Kaiser"])
        controls.addWidget(self.window_combo, row, 3)

        row += 1
        controls.addWidget(QLabel("Height threshold (dB):"), row, 0)
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(-200, 0)
        self.height_spin.setValue(-60)
        self.height_spin.setDecimals(1)
        self.height_spin.setSingleStep(1.0)
        controls.addWidget(self.height_spin, row, 1)

        controls.addWidget(QLabel("Min distance (bins):"), row, 2)
        self.distance_spin = QSpinBox()
        self.distance_spin.setRange(1, 10000)
        self.distance_spin.setValue(10)
        controls.addWidget(self.distance_spin, row, 3)

        row += 1
        controls.addWidget(QLabel("Prominence (dB):"), row, 0)
        self.prominence_spin = QDoubleSpinBox()
        self.prominence_spin.setRange(0, 200)
        self.prominence_spin.setValue(10)
        self.prominence_spin.setDecimals(1)
        self.prominence_spin.setSingleStep(1.0)
        controls.addWidget(self.prominence_spin, row, 1)

        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.setFixedHeight(32)
        self.analyze_btn.clicked.connect(self._on_analyze)
        controls.addWidget(self.analyze_btn, row, 2, 1, 1)

        self._save_btn = QPushButton("💾 Save")
        self._save_btn.setFixedHeight(32)
        self._save_btn.setStyleSheet("""
            QPushButton { background-color: #555; color: white; border: none;
            border-radius: 3px; padding: 4px 12px; }
            QPushButton:hover { background-color: #666; }
        """)
        self._save_btn.clicked.connect(self._on_save_result)
        controls.addWidget(self._save_btn, row, 3, 1, 1)

        row += 1
        self.thd_label = QLabel("THD: —")
        self.thd_label.setStyleSheet("font-size:14px; font-weight:bold;")
        controls.addWidget(self.thd_label, row, 0, 1, 2)

        self.thdn_label = QLabel("THD+N: —")
        self.thdn_label.setStyleSheet("font-size:14px; font-weight:bold;")
        controls.addWidget(self.thdn_label, row, 2, 1, 2)

        root.addLayout(controls)

        self.peak_table = QTableWidget(0, 3)
        self.peak_table.setHorizontalHeaderLabels(["Peak#", "Frequency (Hz)", "Magnitude (dB)"])
        self.peak_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.peak_table.setMaximumHeight(160)
        self.peak_table.verticalHeader().setVisible(False)
        root.addWidget(self.peak_table)

        self.figure = Figure(figsize=(8, 5), facecolor="#1e1e1e")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes_spectrum = self.figure.add_subplot(2, 1, 1)
        self.axes_thd = self.figure.add_subplot(2, 1, 2)
        self._style_axes()
        self.figure.tight_layout(pad=2.0)
        root.addWidget(self.canvas, stretch=1)

    def _apply_style(self):
        self.setStyleSheet("""
            QWidget { background-color: #1e1e1e; color: #d4d4d4; }
            QComboBox, QDoubleSpinBox, QSpinBox {
                background-color: #2d2d2d; color: #d4d4d4;
                border: 1px solid #555; border-radius: 3px; padding: 3px 6px;
            }
            QPushButton {
                background-color: #0e639c; color: #fff;
                border: none; border-radius: 3px; padding: 5px 14px;
            }
            QPushButton:hover { background-color: #1177bb; }
            QTableWidget {
                background-color: #252526; color: #d4d4d4;
                gridline-color: #3c3c3c; border: 1px solid #3c3c3c;
            }
            QHeaderView::section {
                background-color: #333; color: #d4d4d4;
                border: 1px solid #3c3c3c; padding: 3px;
            }
            QLabel { color: #d4d4d4; }
        """)

    def _style_axes(self):
        for ax in (self.axes_spectrum, self.axes_thd):
            ax.set_facecolor("#1e1e1e")
            ax.tick_params(colors="#aaa")
            ax.xaxis.label.set_color("#ccc")
            ax.yaxis.label.set_color("#ccc")
            ax.title.set_color("#ccc")
            for spine in ax.spines.values():
                spine.set_color("#555")

    def set_data(self, audio_data):
        self.audio_data = audio_data
        self._analysis_range = None
        self._last_result = None
        self.channel_combo.clear()
        for i in range(audio_data.num_channels):
            name = audio_data.channel_names[i] if i < len(audio_data.channel_names) else f"Ch {i}"
            self.channel_combo.addItem(name)

    def set_analysis_range(self, t0, t1):
        if self.audio_data is None:
            return
        self._analysis_range = (t0, t1)
        self._on_analyze()

    def _get_range_signal(self, channel_idx):
        if self.audio_data is None or channel_idx < 0:
            return None
        sig = self.audio_data.get_channel(channel_idx)
        if self._analysis_range is None:
            return sig
        t0, t1 = self._analysis_range
        i0 = int(t0 * self.audio_data.sample_rate)
        i1 = int(t1 * self.audio_data.sample_rate)
        i0 = max(0, min(i0, len(sig)))
        i1 = max(i0 + 1, min(i1, len(sig)))
        return sig[i0:i1]

    def _get_window(self, n):
        name = self.window_combo.currentText()
        if name == "Hanning":
            return windows.hann(n)
        elif name == "Hamming":
            return windows.hamming(n)
        elif name == "Blackman":
            return windows.blackman(n)
        elif name == "Kaiser":
            return windows.kaiser(n, beta=14)
        return windows.hann(n)

    def _on_analyze(self):
        if self.audio_data is None:
            return

        ch = self.channel_combo.currentIndex()
        signal = self._get_range_signal(ch)
        if signal is None:
            return
        signal = signal.astype(np.float64)
        sr = self.audio_data.sample_rate
        n = len(signal)

        max_fft = 65536
        if n > max_fft:
            step = n // max_fft
            signal = signal[::step]
            n = len(signal)
            sr = sr / step

        window = self._get_window(n)
        windowed = signal * window
        spectrum = np.abs(rfft(windowed)) * 2.0 / np.sum(window)
        freqs = rfftfreq(n, d=1.0 / sr)

        spectrum_db = 20 * np.log10(np.maximum(spectrum, 1e-12))

        height = self.height_spin.value()
        distance = self.distance_spin.value()
        prominence = self.prominence_spin.value()

        peak_indices, properties = find_peaks(
            spectrum_db, height=height, distance=distance, prominence=prominence
        )

        self._last_result = {
            "freqs": freqs,
            "spectrum_db": spectrum_db,
            "peak_indices": peak_indices,
            "channel": self.channel_combo.currentText(),
        }

        self._plot_spectrum(freqs, spectrum_db, peak_indices)
        self._fill_peak_table(freqs, spectrum_db, peak_indices)
        self._compute_thd(freqs, spectrum, spectrum_db, peak_indices, sr)

        self.figure.tight_layout(pad=2.0)
        self.canvas.draw()

    def _plot_spectrum(self, freqs, spectrum_db, peak_indices):
        ax = self.axes_spectrum
        ax.clear()
        ax.plot(freqs, spectrum_db, color="#4fc3f7", linewidth=0.8)
        if len(peak_indices) > 0:
            ax.scatter(freqs[peak_indices], spectrum_db[peak_indices],
                       color="#ff5252", s=40, zorder=5)
            for idx in peak_indices:
                ax.annotate(
                    f"{freqs[idx]:.0f}",
                    xy=(freqs[idx], spectrum_db[idx]),
                    xytext=(0, 8), textcoords="offset points",
                    fontsize=7, color="#ff8a80", ha="center",
                )
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.set_title("Spectrum with Detected Peaks")
        ax.set_xlim(left=0)
        ax.grid(True, color="#333", linewidth=0.5, alpha=0.5)
        self._style_axes()

    def _fill_peak_table(self, freqs, spectrum_db, peak_indices):
        self.peak_table.setRowCount(len(peak_indices))
        for i, idx in enumerate(peak_indices):
            self.peak_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.peak_table.setItem(i, 1, QTableWidgetItem(f"{freqs[idx]:.2f}"))
            self.peak_table.setItem(i, 2, QTableWidgetItem(f"{spectrum_db[idx]:.2f}"))

    def _compute_thd(self, freqs, spectrum_linear, spectrum_db, peak_indices, sr):
        ax = self.axes_thd
        ax.clear()

        if len(peak_indices) == 0:
            self.thd_label.setText("THD: —")
            self.thdn_label.setText("THD+N: —")
            ax.set_title("THD Breakdown")
            ax.grid(True, color="#333", linewidth=0.5, alpha=0.5)
            self._style_axes()
            return

        magnitudes = spectrum_db[peak_indices]
        fundamental_local = np.argmax(magnitudes)
        fundamental_idx = peak_indices[fundamental_local]
        f0 = freqs[fundamental_idx]
        v_fundamental = spectrum_linear[fundamental_idx]

        harmonic_indices = []
        harmonic_labels = []
        harmonic_values = []

        for h in range(2, 20):
            target_freq = f0 * h
            if target_freq > freqs[-1]:
                break
            target_bin = int(round(target_freq / (freqs[1] - freqs[0]))) if len(freqs) > 1 else 0
            search_range = max(5, int(round(f0 / (freqs[1] - freqs[0])) * 0.05)) if len(freqs) > 1 else 5
            low = max(0, target_bin - search_range)
            high = min(len(spectrum_linear), target_bin + search_range + 1)
            if low >= high:
                continue
            local_idx = low + np.argmax(spectrum_linear[low:high])
            if spectrum_db[local_idx] > self.height_spin.value():
                harmonic_indices.append(local_idx)
                harmonic_labels.append(f"H{h}")
                harmonic_values.append(spectrum_linear[local_idx])

        v_harmonics_sq = sum(v ** 2 for v in harmonic_values)
        thd = (np.sqrt(v_harmonics_sq) / v_fundamental * 100) if v_fundamental > 0 else 0.0
        self.thd_label.setText(f"THD: {thd:.4f}%")

        total_power = np.sum(spectrum_linear ** 2)
        harmonic_power = v_fundamental ** 2 + v_harmonics_sq
        noise_power = max(total_power - harmonic_power, 0)
        thdn = (np.sqrt(v_harmonics_sq + noise_power) / v_fundamental * 100) if v_fundamental > 0 else 0.0
        self.thdn_label.setText(f"THD+N: {thdn:.4f}%")

        labels = ["Fund"] + harmonic_labels
        values_db = [20 * np.log10(max(v_fundamental, 1e-12))]
        colors = ["#4fc3f7"]
        for v in harmonic_values:
            values_db.append(20 * np.log10(max(v, 1e-12)))
            colors.append("#ff8a65")

        ax.bar(labels, values_db, color=colors, edgecolor="#555", linewidth=0.5)
        ax.set_xlabel("Harmonic")
        ax.set_ylabel("Magnitude (dB)")
        ax.set_title(f"THD Breakdown  (f₀ = {f0:.1f} Hz)")
        ax.grid(True, color="#333", linewidth=0.5, alpha=0.5, axis="y")
        self._style_axes()

    def _on_save_result(self):
        if self._last_result is None:
            return
        thd = self.thd_label.text().replace("THD: ", "")
        thdn = self.thdn_label.text().replace("THD+N: ", "")
        peaks = []
        for i in range(self.peak_table.rowCount()):
            freq_item = self.peak_table.item(i, 1)
            mag_item = self.peak_table.item(i, 2)
            if freq_item and mag_item:
                peaks.append({
                    "freq": float(freq_item.text()),
                    "mag": float(mag_item.text()),
                })
        data = {
            "freqs": self._last_result["freqs"],
            "spectrum_db": self._last_result["spectrum_db"],
            "peak_indices": self._last_result["peak_indices"],
            "channel": self._last_result["channel"],
            "peaks": peaks,
            "thd": thd,
            "thdn": thdn,
        }
        params = {
            "window": self.window_combo.currentText(),
            "height": self.height_spin.value(),
            "distance": self.distance_spin.value(),
            "prominence": self.prominence_spin.value(),
            "range": str(self._analysis_range) if self._analysis_range else "full",
        }
        ch = self._last_result["channel"]
        self.save_result_requested.emit(data, params, f"THD {ch}")

    def restore_result(self, result: AnalysisResult):
        d = result.data
        freqs = np.array(d.get("freqs", []))
        spectrum_db = np.array(d.get("spectrum_db", []))
        peak_indices = np.array(d.get("peak_indices", [])).astype(int)
        self._last_result = {
            "freqs": freqs,
            "spectrum_db": spectrum_db,
            "peak_indices": peak_indices,
            "channel": d.get("channel", ""),
        }

        if d.get("channel"):
            idx = self.channel_combo.findText(d["channel"])
            if idx >= 0:
                self.channel_combo.blockSignals(True)
                self.channel_combo.setCurrentIndex(idx)
                self.channel_combo.blockSignals(False)

        rp = result.params if result.params else {}
        if "window" in rp:
            idx = self.window_combo.findText(rp["window"])
            if idx >= 0:
                self.window_combo.blockSignals(True)
                self.window_combo.setCurrentIndex(idx)
                self.window_combo.blockSignals(False)
        if "height" in rp:
            self.height_spin.blockSignals(True)
            self.height_spin.setValue(float(rp["height"]))
            self.height_spin.blockSignals(False)
        if "distance" in rp:
            self.distance_spin.blockSignals(True)
            self.distance_spin.setValue(int(rp["distance"]))
            self.distance_spin.blockSignals(False)
        if "prominence" in rp:
            self.prominence_spin.blockSignals(True)
            self.prominence_spin.setValue(float(rp["prominence"]))
            self.prominence_spin.blockSignals(False)

        self._plot_spectrum(freqs, spectrum_db, peak_indices)
        self._fill_peak_table(freqs, spectrum_db, peak_indices)
        thd = d.get("thd", "—")
        thdn = d.get("thdn", "—")
        self.thd_label.setText(f"THD: {thd}")
        self.thdn_label.setText(f"THD+N: {thdn}")
        self.figure.tight_layout(pad=2.0)
        self.canvas.draw()

    def export_csv_data(self):
        if self.audio_data is None:
            raise ValueError("No data loaded")
        headers = ["Peak#", "Frequency(Hz)", "Magnitude(dB)", "THD%", "THD+N%"]
        peak_count = self.peak_table.rowCount()
        peak_nums = np.arange(1, peak_count + 1, dtype=float)
        freqs_arr = np.zeros(peak_count)
        mags_arr = np.zeros(peak_count)
        for i in range(peak_count):
            freq_item = self.peak_table.item(i, 1)
            mag_item = self.peak_table.item(i, 2)
            freqs_arr[i] = float(freq_item.text()) if freq_item else 0.0
            mags_arr[i] = float(mag_item.text()) if mag_item else 0.0
        thd_text = self.thd_label.text().replace("THD: ", "").replace("%", "")
        thdn_text = self.thdn_label.text().replace("THD+N: ", "").replace("%", "")
        max_len = max(peak_count, 1)
        thd_val = float(thd_text) if thd_text not in ("—", "") else 0.0
        thdn_val = float(thdn_text) if thdn_text not in ("—", "") else 0.0
        thd_col = np.full(max_len, thd_val)
        thdn_col = np.full(max_len, thdn_val)
        peak_nums = np.pad(peak_nums, (0, max_len - peak_count), constant_values=np.nan)
        freqs_arr = np.pad(freqs_arr, (0, max_len - peak_count), constant_values=np.nan)
        mags_arr = np.pad(mags_arr, (0, max_len - peak_count), constant_values=np.nan)
        arrays = [peak_nums, freqs_arr, mags_arr, thd_col, thdn_col]
        return arrays, headers

    def get_figure(self):
        return self.figure
