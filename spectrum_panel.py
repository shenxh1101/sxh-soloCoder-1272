import numpy as np
from scipy.signal import welch
from scipy.signal.windows import kaiser
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox,
    QDoubleSpinBox, QLabel, QSizePolicy, QCheckBox, QPushButton,
    QRadioButton, QButtonGroup, QFrame,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from results_manager import AnalysisResult


WINDOW_FUNCTIONS = {
    "Hanning": np.hanning,
    "Hamming": np.hamming,
    "Blackman": np.blackman,
}

FFT_SIZES = ["Auto", "256", "512", "1024", "2048", "4096", "8192", "16384", "32768", "65536"]

MAX_FFT_SAMPLES = 262144


class SpectrumWorker(QThread):
    result_ready = pyqtSignal(str, np.ndarray, np.ndarray)

    def __init__(self, signal, sr, window_name, beta, nfft, view, label="A"):
        super().__init__()
        self.signal = signal
        self.sr = sr
        self.window_name = window_name
        self.beta = beta
        self.nfft = nfft
        self.view = view
        self.label = label

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
            self.result_ready.emit(self.label, freqs[::step], magnitude_db[::step])
        elif self.view == "Phase":
            windowed = self.signal * window
            fft_result = np.fft.rfft(windowed, n=nfft)
            phase = np.unwrap(np.angle(fft_result))
            freqs = np.fft.rfftfreq(nfft, d=1.0 / self.sr)
            step = max(1, len(freqs) // 8000)
            self.result_ready.emit(self.label, freqs[::step], phase[::step])
        elif self.view == "Power Spectral Density (PSD)":
            nperseg = min(n_samples, 8192)
            win = window[:nperseg] if len(window) >= nperseg else window
            freqs_w, psd = welch(
                self.signal, fs=self.sr, window=win,
                nperseg=nperseg, nfft=max(nfft, nperseg), scaling="density",
            )
            psd_db = 10 * np.log10(psd + 1e-12)
            self.result_ready.emit(self.label, freqs_w, psd_db)

    def _get_window(self, n):
        if self.window_name == "Kaiser":
            return kaiser(n, self.beta)
        return WINDOW_FUNCTIONS.get(self.window_name, np.hanning)(n)


class SpectrumPanel(QWidget):
    save_result_requested = pyqtSignal(dict, dict, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._audio_data = None
        self._worker = None
        self._worker_b = None
        self._pending_view = None
        self._last_result = None
        self._last_result_b = None
        self._result_a_meta = None
        self._result_b_meta = None
        self._analysis_range = None
        self._results_manager = None
        self._setup_ui()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        controls = QHBoxLayout()

        controls.addWidget(QLabel("View:"))
        self._view_combo = QComboBox()
        self._view_combo.addItems(["Magnitude", "Phase", "Power Spectral Density (PSD)"])
        self._view_combo.currentIndexChanged.connect(self._on_view_changed)
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

        controls.addStretch()

        self._save_btn = QPushButton("💾 Save")
        self._save_btn.setFixedHeight(24)
        self._save_btn.setStyleSheet("""
            QPushButton { background-color: #0e639c; color: white; border: none;
            border-radius: 3px; padding: 2px 10px; font-size: 11px; }
            QPushButton:hover { background-color: #1177bb; }
        """)
        self._save_btn.clicked.connect(self._on_save_result)
        controls.addWidget(self._save_btn)

        root.addLayout(controls)

        # Compare section
        compare_frame = QFrame()
        compare_frame.setStyleSheet("""
            QFrame { background-color: #2b2b2b; border: 1px solid #3c3c3c;
                    border-radius: 4px; }
            QLabel { color: #cccccc; }
            QCheckBox { color: #cccccc; }
            QComboBox { background-color: #2d2d2d; color: #ffffff;
                       border: 1px solid #555; border-radius: 3px; padding: 2px 4px; }
            QRadioButton { color: #cccccc; }
        """)
        compare_layout = QVBoxLayout(compare_frame)
        compare_layout.setContentsMargins(8, 6, 8, 6)
        compare_layout.setSpacing(4)

        row0 = QHBoxLayout()
        self._compare_cb = QCheckBox("A/B Compare (overlay two spectra)")
        self._compare_cb.stateChanged.connect(self._on_compare_toggled)
        row0.addWidget(self._compare_cb)
        row0.addStretch()
        compare_layout.addLayout(row0)

        row1 = QHBoxLayout()
        self._mode_group = QButtonGroup(self)
        self._mode_channel_rb = QRadioButton("Compare channels")
        self._mode_saved_rb = QRadioButton("Compare saved results")
        self._mode_channel_rb.setChecked(True)
        self._mode_group.addButton(self._mode_channel_rb, 0)
        self._mode_group.addButton(self._mode_saved_rb, 1)
        self._mode_channel_rb.setEnabled(False)
        self._mode_saved_rb.setEnabled(False)
        self._mode_group.buttonClicked.connect(self._on_compare_mode_changed)
        row1.addWidget(self._mode_channel_rb)
        row1.addWidget(self._mode_saved_rb)
        row1.addStretch()
        compare_layout.addLayout(row1)

        row2 = QHBoxLayout()
        self._ch_a_label = QLabel("A: Channel")
        row2.addWidget(self._ch_a_label)
        self._channel_combo = QComboBox()
        self._channel_combo.setMinimumWidth(140)
        self._channel_combo.currentIndexChanged.connect(self._refresh)
        row2.addWidget(self._channel_combo)

        self._ch_b_label = QLabel("B: Channel")
        self._ch_b_label.setVisible(False)
        row2.addWidget(self._ch_b_label)
        self._channel_b_combo = QComboBox()
        self._channel_b_combo.setMinimumWidth(140)
        self._channel_b_combo.setVisible(False)
        self._channel_b_combo.currentIndexChanged.connect(self._refresh)
        row2.addWidget(self._channel_b_combo)

        self._saved_a_label = QLabel("A: Saved Result")
        self._saved_a_label.setVisible(False)
        row2.addWidget(self._saved_a_label)
        self._saved_a_combo = QComboBox()
        self._saved_a_combo.setMinimumWidth(180)
        self._saved_a_combo.setVisible(False)
        self._saved_a_combo.currentIndexChanged.connect(self._on_saved_result_changed)
        row2.addWidget(self._saved_a_combo)

        self._saved_b_label = QLabel("B: Saved Result")
        self._saved_b_label.setVisible(False)
        row2.addWidget(self._saved_b_label)
        self._saved_b_combo = QComboBox()
        self._saved_b_combo.setMinimumWidth(180)
        self._saved_b_combo.setVisible(False)
        self._saved_b_combo.currentIndexChanged.connect(self._on_saved_result_changed)
        row2.addWidget(self._saved_b_combo)

        row2.addStretch()
        compare_layout.addLayout(row2)

        root.addWidget(compare_frame)

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

    def set_results_manager(self, manager):
        self._results_manager = manager
        self._refresh_saved_combos()

    def _refresh_saved_combos(self):
        if self._results_manager is None:
            return
        self._saved_a_combo.blockSignals(True)
        self._saved_b_combo.blockSignals(True)
        self._saved_a_combo.clear()
        self._saved_b_combo.clear()
        results = self._results_manager.get_results("spectrum")
        for r in results:
            label = f"{r.name}"
            self._saved_a_combo.addItem(label, r.result_id)
            self._saved_b_combo.addItem(label, r.result_id)
        self._saved_a_combo.blockSignals(False)
        self._saved_b_combo.blockSignals(False)

    def set_analysis_range(self, t0, t1):
        if self._audio_data is None:
            return
        self._analysis_range = (t0, t1)
        if self._mode_channel_rb.isChecked():
            self._refresh()

    def _get_range_signal(self, channel_idx):
        if self._audio_data is None or channel_idx < 0:
            return None
        sig = self._audio_data.get_channel(channel_idx)
        if self._analysis_range is None:
            return sig
        t0, t1 = self._analysis_range
        i0 = int(t0 * self._audio_data.sample_rate)
        i1 = int(t1 * self._audio_data.sample_rate)
        i0 = max(0, min(i0, len(sig)))
        i1 = max(i0 + 1, min(i1, len(sig)))
        return sig[i0:i1]

    def _on_view_changed(self, idx):
        if self._mode_channel_rb.isChecked():
            self._refresh()
        else:
            self._on_saved_result_changed()

    def _on_window_changed(self, index):
        is_kaiser = self._window_combo.currentText() == "Kaiser"
        self._beta_label.setVisible(is_kaiser)
        self._beta_spin.setVisible(is_kaiser)
        if self._mode_channel_rb.isChecked():
            self._refresh()

    def _on_compare_toggled(self, state):
        enabled = state == Qt.CheckState.Checked.value
        self._mode_channel_rb.setEnabled(enabled)
        self._mode_saved_rb.setEnabled(enabled)
        self._update_compare_widgets()
        self._last_result_b = None
        self._result_b_meta = None
        if self._mode_channel_rb.isChecked():
            self._refresh()
        else:
            self._on_saved_result_changed()

    def _on_compare_mode_changed(self, btn):
        self._update_compare_widgets()
        self._last_result_b = None
        self._result_b_meta = None
        if self._mode_channel_rb.isChecked():
            self._refresh()
        else:
            self._on_saved_result_changed()

    def _update_compare_widgets(self):
        compare = self._compare_cb.isChecked()
        mode_saved = self._mode_saved_rb.isChecked()

        self._ch_a_label.setVisible(not mode_saved)
        self._channel_combo.setVisible(not mode_saved)
        self._ch_b_label.setVisible(compare and not mode_saved)
        self._channel_b_combo.setVisible(compare and not mode_saved)

        self._saved_a_label.setVisible(mode_saved)
        self._saved_a_combo.setVisible(mode_saved)
        self._saved_b_label.setVisible(compare and mode_saved)
        self._saved_b_combo.setVisible(compare and mode_saved)

    def _on_saved_result_changed(self):
        self._last_result = None
        self._last_result_b = None
        self._result_a_meta = None
        self._result_b_meta = None

        rid_a = self._saved_a_combo.currentData()
        if rid_a and self._results_manager:
            r = self._results_manager.get_result(rid_a)
            if r:
                d = r.data
                view = d.get("view", "Magnitude")
                x = np.array(d.get("x", []))
                y = np.array(d.get("y", []))
                self._last_result = (view, x, y)
                self._result_a_meta = {"name": r.name, "source": r.source_file}

        compare = self._compare_cb.isChecked()
        if compare:
            rid_b = self._saved_b_combo.currentData()
            if rid_b and self._results_manager:
                r = self._results_manager.get_result(rid_b)
                if r:
                    d = r.data
                    view = d.get("view", "Magnitude")
                    x = np.array(d.get("x", []))
                    y = np.array(d.get("y", []))
                    self._last_result_b = (view, x, y)
                    self._result_b_meta = {"name": r.name, "source": r.source_file}

        if self._last_result is not None:
            view = self._last_result[0]
            idx = 0
            if view == "Magnitude":
                idx = 0
            elif view == "Phase":
                idx = 1
            else:
                idx = 2
            if self._view_combo.currentIndex() != idx:
                self._view_combo.blockSignals(True)
                self._view_combo.setCurrentIndex(idx)
                self._view_combo.blockSignals(False)

        self._plot_current()

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
        self._analysis_range = None
        self._channel_combo.blockSignals(True)
        self._channel_b_combo.blockSignals(True)
        self._channel_combo.clear()
        self._channel_b_combo.clear()
        for i in range(audio_data.num_channels):
            label = audio_data.channel_names[i] if i < len(audio_data.channel_names) else f"Channel {i}"
            self._channel_combo.addItem(label)
            self._channel_b_combo.addItem(label)
        if audio_data.num_channels >= 2:
            self._channel_b_combo.setCurrentIndex(1)
        self._channel_combo.blockSignals(False)
        self._channel_b_combo.blockSignals(False)
        self._last_result = None
        self._last_result_b = None
        self._refresh_saved_combos()
        self._refresh()

    def _refresh(self):
        if self._mode_saved_rb.isChecked():
            self._on_saved_result_changed()
            return
        if self._audio_data is None:
            return
        ch_a = self._channel_combo.currentIndex()
        ch_b = self._channel_b_combo.currentIndex()
        compare = self._compare_cb.isChecked()
        if ch_a < 0:
            return
        signal_a = self._get_range_signal(ch_a)
        if signal_a is None:
            return
        sr = self._audio_data.sample_rate
        view = self._view_combo.currentText()

        def prepare(sig):
            n = len(sig)
            if n > MAX_FFT_SAMPLES and view != "Power Spectral Density (PSD)":
                sig = sig[-MAX_FFT_SAMPLES:]
                n = len(sig)
            return sig, n

        sig_a, n_a = prepare(signal_a)
        nfft = self._get_nfft(n_a)
        window_name = self._window_combo.currentText()
        beta = self._beta_spin.value()

        self._show_computing(True)

        if self._worker is not None and self._worker.isRunning():
            self._worker.quit()
            self._worker.wait(100)
        if self._worker_b is not None and self._worker_b.isRunning():
            self._worker_b.quit()
            self._worker_b.wait(100)

        self._pending_count = 2 if compare else 1
        self._result_a_meta = {"name": f"Ch {self._channel_combo.currentText()}"}
        self._worker = SpectrumWorker(sig_a, sr, window_name, beta, nfft, view, label="A")
        self._worker.result_ready.connect(self._on_result_ready)
        self._worker.start()

        if compare and ch_b >= 0:
            signal_b = self._get_range_signal(ch_b)
            if signal_b is not None:
                sig_b, n_b = prepare(signal_b)
                self._result_b_meta = {"name": f"Ch {self._channel_b_combo.currentText()}"}
                self._worker_b = SpectrumWorker(sig_b, sr, window_name, beta, nfft, view, label="B")
                self._worker_b.result_ready.connect(self._on_result_ready)
                self._worker_b.start()
            else:
                self._pending_count = 1
        elif compare:
            self._pending_count = 1

    def _show_computing(self, show):
        if show:
            self._status_label.setText("Computing spectrum…")
            self._status_label.setVisible(True)
        else:
            self._status_label.setVisible(False)

    def _on_result_ready(self, label, x_data, y_data):
        if label == "A":
            self._last_result = (self._view_combo.currentText(), x_data.copy(), y_data.copy())
        elif label == "B":
            self._last_result_b = (self._view_combo.currentText(), x_data.copy(), y_data.copy())

        self._pending_count -= 1
        if self._pending_count > 0:
            return

        self._show_computing(False)
        self._plot_current()

    def _plot_current(self):
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        ax.set_facecolor("#2d2d2d")
        ax.tick_params(colors="white", which="both")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("#555555")

        view = self._view_combo.currentText()
        compare = self._compare_cb.isChecked()

        if self._last_result is not None:
            v_a, xa, ya = self._last_result
            if view == "PSD" and v_a == "Power Spectral Density (PSD)":
                v_display = v_a
            else:
                v_display = view
            label_a = self._result_a_meta["name"] if self._result_a_meta else "A"
            ax.plot(xa, ya, color="#00ccff", linewidth=0.8, label=label_a)

        if compare and self._last_result_b is not None:
            v_b, xb, yb = self._last_result_b
            label_b = self._result_b_meta["name"] if self._result_b_meta else "B"
            ax.plot(xb, yb, color="#ff6b6b", linewidth=0.8, label=label_b, alpha=0.85)
            ax.legend(facecolor="#2b2b2b", edgecolor="#555555", labelcolor="#cccccc", fontsize=9)

        if view == "Magnitude":
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Magnitude (dB)")
            ax.set_title("Magnitude Spectrum")
        elif view == "Phase":
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Phase (radians)")
            ax.set_title("Phase Spectrum")
        elif view == "PSD" or view == "Power Spectral Density (PSD)":
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("PSD (dB/Hz)")
            ax.set_title("Power Spectral Density")

        ax.grid(True, which="both", linestyle="--", linewidth=0.5, color="#555555", alpha=0.7)
        self._figure.tight_layout()
        self._canvas.draw()

    def _on_save_result(self):
        if self._last_result is None:
            return
        view, x, y = self._last_result
        ch = self._channel_combo.currentText() if self._mode_channel_rb.isChecked() else (
            self._result_a_meta["name"] if self._result_a_meta else "Saved")
        rng = self._analysis_range if self._analysis_range else "full"
        data = {
            "view": view,
            "x": x,
            "y": y,
            "channel": ch,
        }
        params = {
            "window": self._window_combo.currentText(),
            "fft_size": self._fft_combo.currentText(),
            "range": str(rng),
        }
        self.save_result_requested.emit(data, params, f"Spectrum {ch} {view}")

    def restore_result(self, result: AnalysisResult):
        d = result.data
        view = d.get("view", "Magnitude")
        x = np.array(d.get("x", []))
        y = np.array(d.get("y", []))

        idx = 0
        if view == "Magnitude":
            idx = 0
        elif view == "Phase":
            idx = 1
        else:
            idx = 2
        self._view_combo.blockSignals(True)
        self._view_combo.setCurrentIndex(idx)
        self._view_combo.blockSignals(False)

        if d.get("window"):
            widx = self._window_combo.findText(d["window"])
            if widx >= 0:
                self._window_combo.blockSignals(True)
                self._window_combo.setCurrentIndex(widx)
                self._window_combo.blockSignals(False)

        self._last_result = (view, x, y)
        self._last_result_b = None
        self._result_a_meta = {"name": result.name, "source": result.source_file}
        self._result_b_meta = None
        self._compare_cb.blockSignals(True)
        self._compare_cb.setChecked(False)
        self._compare_cb.blockSignals(False)
        self._update_compare_widgets()
        self._plot_current()

    def export_csv_data(self):
        if self._last_result is None:
            raise ValueError("No spectrum computed yet. Wait for the analysis to finish first.")
        compare = self._compare_cb.isChecked() and self._last_result_b is not None

        view_a, xa, ya = self._last_result
        view_b, xb, yb = None, None, None
        if compare:
            view_b, xb, yb = self._last_result_b

        def ylabel(view, tag):
            if view == "Magnitude":
                return f"Magnitude(dB)_{tag}"
            elif view == "Phase":
                return f"Phase(rad)_{tag}"
            else:
                return f"PSD(dB/Hz)_{tag}"

        def label_name(meta, tag):
            if meta and meta.get("name"):
                return meta["name"]
            return tag

        label_a = label_name(self._result_a_meta, "A")
        headers = [f"Frequency(Hz)_A_{label_a}", ylabel(view_a, f"A_{label_a}")]
        arrays = [xa, ya]

        if compare:
            label_b = label_name(self._result_b_meta, "B")
            headers.append(f"Frequency(Hz)_B_{label_b}")
            headers.append(ylabel(view_b if view_b else view_a, f"B_{label_b}"))
            max_len = max(len(xa), len(xb))
            xa_pad = np.pad(xa, (0, max_len - len(xa)), constant_values=np.nan)
            ya_pad = np.pad(ya, (0, max_len - len(ya)), constant_values=np.nan)
            xb_pad = np.pad(xb, (0, max_len - len(xb)), constant_values=np.nan)
            yb_pad = np.pad(yb, (0, max_len - len(yb)), constant_values=np.nan)
            arrays = [xa_pad, ya_pad, xb_pad, yb_pad]

        return arrays, headers

    def get_last_result(self):
        return self._last_result

    def get_figure(self):
        return self._figure
