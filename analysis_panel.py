import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks, windows
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QDoubleSpinBox, QSpinBox,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class AnalysisPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.audio_data = None
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
        controls.addWidget(self.analyze_btn, row, 2, 1, 2)

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
        self.channel_combo.clear()
        for i in range(audio_data.num_channels):
            name = audio_data.channel_names[i] if i < len(audio_data.channel_names) else f"Ch {i}"
            self.channel_combo.addItem(name)

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
        signal = self.audio_data.get_channel(ch).astype(np.float64)
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

    def get_figure(self):
        return self.figure
