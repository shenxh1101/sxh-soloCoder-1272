import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QDoubleSpinBox,
    QSpinBox, QPushButton, QLabel, QGroupBox, QFormLayout, QSizePolicy,
    QInputDialog, QListWidget, QListWidgetItem,
)
from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from scipy import signal as sig
from audio_engine import AudioData
from results_manager import AnalysisResult


class FilterPanel(QWidget):
    filter_applied = pyqtSignal(object)
    save_result_requested = pyqtSignal(dict, dict, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._audio_data = None
        self._filtered_signal = None
        self._last_result = None
        self._presets = {}
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(300)
        self._debounce_timer.timeout.connect(self._update_preview)
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)

        ctrl = QWidget()
        ctrl.setFixedWidth(280)
        ctrl_layout = QVBoxLayout(ctrl)

        form_group = QGroupBox("Filter Parameters")
        form = QFormLayout(form_group)

        self.filter_type_combo = QComboBox()
        self.filter_type_combo.addItems(["Lowpass", "Highpass", "Bandpass", "Bandstop"])
        self.filter_type_combo.currentIndexChanged.connect(self._on_param_changed)
        self.filter_type_combo.currentIndexChanged.connect(self._update_cutoff_visibility)
        form.addRow("Filter Type:", self.filter_type_combo)

        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "FIR (Window)", "IIR Butterworth", "IIR Chebyshev Type I",
            "IIR Chebyshev Type II", "IIR Elliptic",
        ])
        self.method_combo.currentIndexChanged.connect(self._on_param_changed)
        self.method_combo.currentIndexChanged.connect(self._update_method_visibility)
        form.addRow("Design Method:", self.method_combo)

        self.channel_combo = QComboBox()
        self.channel_combo.currentIndexChanged.connect(self._on_param_changed)
        form.addRow("Channel:", self.channel_combo)

        self.fc1_spin = QDoubleSpinBox()
        self.fc1_spin.setRange(1, 999999)
        self.fc1_spin.setValue(1000)
        self.fc1_spin.setSuffix(" Hz")
        self.fc1_spin.setDecimals(1)
        self.fc1_spin.valueChanged.connect(self._on_param_changed)
        form.addRow("Cutoff (fc1):", self.fc1_spin)

        self.fc2_spin = QDoubleSpinBox()
        self.fc2_spin.setRange(1, 999999)
        self.fc2_spin.setValue(3000)
        self.fc2_spin.setSuffix(" Hz")
        self.fc2_spin.setDecimals(1)
        self.fc2_spin.valueChanged.connect(self._on_param_changed)
        self.fc2_spin.setVisible(False)
        self.fc2_label = QLabel("Cutoff (fc2):")
        form.addRow(self.fc2_label, self.fc2_spin)
        self.fc2_label.setVisible(False)

        self.order_spin = QSpinBox()
        self.order_spin.setRange(1, 200)
        self.order_spin.setValue(8)
        self.order_spin.valueChanged.connect(self._on_param_changed)
        form.addRow("Order:", self.order_spin)

        self.ripple_spin = QDoubleSpinBox()
        self.ripple_spin.setRange(0.01, 50.0)
        self.ripple_spin.setValue(1.0)
        self.ripple_spin.setSuffix(" dB")
        self.ripple_spin.setDecimals(2)
        self.ripple_spin.valueChanged.connect(self._on_param_changed)
        self.ripple_spin.setVisible(False)
        self.ripple_label = QLabel("Passband Ripple (Rp):")
        form.addRow(self.ripple_label, self.ripple_spin)
        self.ripple_label.setVisible(False)

        self.rs_spin = QDoubleSpinBox()
        self.rs_spin.setRange(1.0, 200.0)
        self.rs_spin.setValue(40.0)
        self.rs_spin.setSuffix(" dB")
        self.rs_spin.setDecimals(1)
        self.rs_spin.valueChanged.connect(self._on_param_changed)
        self.rs_spin.setVisible(False)
        self.rs_label = QLabel("Stopband Atten (Rs):")
        form.addRow(self.rs_label, self.rs_spin)
        self.rs_label.setVisible(False)

        self.window_combo = QComboBox()
        self.window_combo.addItems(["Hamming", "Hanning", "Blackman", "Kaiser", "Boxcar"])
        self.window_combo.currentIndexChanged.connect(self._on_param_changed)
        self.window_combo.setVisible(False)
        self.window_label = QLabel("Window:")
        form.addRow(self.window_label, self.window_combo)
        self.window_label.setVisible(False)

        self.kaiser_beta_spin = QDoubleSpinBox()
        self.kaiser_beta_spin.setRange(0.0, 30.0)
        self.kaiser_beta_spin.setValue(5.0)
        self.kaiser_beta_spin.setDecimals(1)
        self.kaiser_beta_spin.valueChanged.connect(self._on_param_changed)
        self.kaiser_beta_spin.setVisible(False)
        self.kaiser_label = QLabel("Kaiser β:")
        form.addRow(self.kaiser_label, self.kaiser_beta_spin)
        self.kaiser_label.setVisible(False)

        ctrl_layout.addWidget(form_group)

        preset_group = QGroupBox("Filter Presets")
        preset_layout = QVBoxLayout(preset_group)
        self.preset_list = QListWidget()
        self.preset_list.itemDoubleClicked.connect(self._on_preset_activated)
        preset_layout.addWidget(self.preset_list)

        preset_btn_row = QHBoxLayout()
        self.save_preset_btn = QPushButton("💾 Save")
        self.save_preset_btn.clicked.connect(self._on_save_preset)
        preset_btn_row.addWidget(self.save_preset_btn)
        self.load_preset_btn = QPushButton("⬇️ Load")
        self.load_preset_btn.clicked.connect(self._on_load_preset)
        preset_btn_row.addWidget(self.load_preset_btn)
        self.del_preset_btn = QPushButton("❌ Delete")
        self.del_preset_btn.clicked.connect(self._on_del_preset)
        preset_btn_row.addWidget(self.del_preset_btn)
        preset_layout.addLayout(preset_btn_row)
        ctrl_layout.addWidget(preset_group)

        action_group = QGroupBox("Actions")
        action_layout = QVBoxLayout(action_group)
        self.apply_btn = QPushButton("🎛️ Apply Filter to Signal")
        self.apply_btn.clicked.connect(self._apply_filter)
        action_layout.addWidget(self.apply_btn)
        self.send_btn = QPushButton("📤 Send Filtered to All Panels")
        self.send_btn.clicked.connect(self._on_send_filtered)
        self.send_btn.setEnabled(False)
        action_layout.addWidget(self.send_btn)
        self.save_result_btn = QPushButton("💾 Save Response")
        self.save_result_btn.clicked.connect(self._on_save_result)
        action_layout.addWidget(self.save_result_btn)
        ctrl_layout.addWidget(action_group)

        ctrl_layout.addStretch()
        layout.addWidget(ctrl)

        self.figure = Figure(figsize=(8, 8), facecolor="#2b2b2b")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.canvas, stretch=1)

        self._create_subplots()

    def _create_subplots(self):
        self.figure.clear()
        self.ax_mag = self.figure.add_subplot(3, 1, 1)
        self.ax_phase = self.figure.add_subplot(3, 1, 2)
        self.ax_time = self.figure.add_subplot(3, 1, 3)
        for ax in (self.ax_mag, self.ax_phase, self.ax_time):
            ax.set_facecolor("#1e1e1e")
            ax.tick_params(colors="#cccccc")
            ax.xaxis.label.set_color("#cccccc")
            ax.yaxis.label.set_color("#cccccc")
            ax.title.set_color("#cccccc")
            for spine in ax.spines.values():
                spine.set_color("#555555")
        self.ax_mag.set_title("Frequency Response (Magnitude)")
        self.ax_mag.set_ylabel("Magnitude (dB)")
        self.ax_mag.grid(True, color="#444444", linewidth=0.5)
        self.ax_phase.set_title("Phase Response")
        self.ax_phase.set_ylabel("Phase (rad)")
        self.ax_phase.grid(True, color="#444444", linewidth=0.5)
        self.ax_time.set_title("Signal Comparison")
        self.ax_time.set_xlabel("Time (s)")
        self.ax_time.set_ylabel("Amplitude")
        self.ax_time.grid(True, color="#444444", linewidth=0.5)
        self.figure.tight_layout()
        self.canvas.draw()

    def _update_cutoff_visibility(self):
        is_band = self.filter_type_combo.currentIndex() >= 2
        self.fc2_spin.setVisible(is_band)
        self.fc2_label.setVisible(is_band)
        if not is_band:
            self.fc1_spin.setPrefix("")
        else:
            self.fc1_spin.setPrefix("")
        self._update_nyquist_limit()

    def _update_method_visibility(self):
        idx = self.method_combo.currentIndex()
        is_fir = idx == 0
        needs_ripple = idx in (2, 4)
        needs_rs = idx in (3, 4)

        self.window_combo.setVisible(is_fir)
        self.window_label.setVisible(is_fir)
        self.kaiser_beta_spin.setVisible(is_fir and self.window_combo.currentIndex() == 3)
        self.kaiser_label.setVisible(is_fir and self.window_combo.currentIndex() == 3)

        self.ripple_spin.setVisible(needs_ripple)
        self.ripple_label.setVisible(needs_ripple)

        self.rs_spin.setVisible(needs_rs)
        self.rs_label.setVisible(needs_rs)

    def _update_nyquist_limit(self):
        if self._audio_data is not None:
            nyq = self._audio_data.sample_rate / 2.0
            self.fc1_spin.setMaximum(nyq - 1)
            self.fc2_spin.setMaximum(nyq - 1)
            if self.fc1_spin.value() >= nyq:
                self.fc1_spin.setValue(nyq * 0.8)
            if self.fc2_spin.value() >= nyq:
                self.fc2_spin.setValue(nyq * 0.9)

    def _on_param_changed(self):
        self._update_method_visibility()
        self._debounce_timer.start()

    def set_data(self, audio_data):
        self._audio_data = audio_data
        self._filtered_signal = None
        self.channel_combo.blockSignals(True)
        self.channel_combo.clear()
        for i in range(audio_data.num_channels):
            name = audio_data.channel_names[i] if audio_data.channel_names else f"Ch {i}"
            self.channel_combo.addItem(name)
        self.channel_combo.blockSignals(False)
        self._update_nyquist_limit()
        self._update_preview()

    def _design_filter(self):
        fs = self._audio_data.sample_rate
        ftype = self.filter_type_combo.currentIndex()
        method = self.method_combo.currentIndex()
        order = self.order_spin.value()
        fc1 = self.fc1_spin.value()
        fc2 = self.fc2_spin.value()

        btype_map = {0: "lowpass", 1: "highpass", 2: "bandpass", 3: "bandstop"}
        btype = btype_map[ftype]
        is_band = ftype >= 2

        if method == 0:
            window_name = self.window_combo.currentText().lower()
            if window_name == "kaiser":
                beta = self.kaiser_beta_spin.value()
                window = ("kaiser", beta)
            else:
                window = window_name
            numtaps = order + 1
            if is_band:
                b = sig.firwin(numtaps, [fc1, fc2], pass_zero=(btype == "bandstop"), fs=fs, window=window)
            else:
                b = sig.firwin(numtaps, fc1, pass_zero=(btype == "highpass"), fs=fs, window=window)
            a = np.array([1.0])
            return b, a, True
        else:
            Wn = np.array([fc1, fc2]) if is_band else fc1
            if method == 1:
                b, a = sig.butter(order, Wn, btype=btype, fs=fs)
            elif method == 2:
                rp = self.ripple_spin.value()
                b, a = sig.cheby1(order, rp, Wn, btype=btype, fs=fs)
            elif method == 3:
                rs = self.rs_spin.value()
                b, a = sig.cheby2(order, rs, Wn, btype=btype, fs=fs)
            elif method == 4:
                rp = self.ripple_spin.value()
                rs = self.rs_spin.value()
                b, a = sig.ellip(order, rp, rs, Wn, btype=btype, fs=fs)
            return b, a, False

    def _update_preview(self):
        if self._audio_data is None:
            return
        try:
            b, a, is_fir = self._design_filter()
        except Exception:
            return

        fs = self._audio_data.sample_rate
        w, h = sig.freqz(b, a, worN=2048, fs=fs)
        mag_db = 20 * np.log10(np.maximum(np.abs(h), 1e-12))
        phase = np.unwrap(np.angle(h))

        self._last_result = {
            "freqs": w,
            "mag_db": mag_db,
            "phase": phase,
            "b": b,
            "a": a,
            "is_fir": is_fir,
        }

        self.ax_mag.clear()
        self.ax_mag.set_facecolor("#1e1e1e")
        self.ax_mag.plot(w, mag_db, color="#00ccff", linewidth=1.2)
        self.ax_mag.set_title("Frequency Response (Magnitude)", color="#cccccc")
        self.ax_mag.set_ylabel("Magnitude (dB)", color="#cccccc")
        self.ax_mag.set_xlim(0, fs / 2)
        self.ax_mag.grid(True, color="#444444", linewidth=0.5)
        self.ax_mag.tick_params(colors="#cccccc")
        for spine in self.ax_mag.spines.values():
            spine.set_color("#555555")

        self.ax_phase.clear()
        self.ax_phase.set_facecolor("#1e1e1e")
        self.ax_phase.plot(w, phase, color="#ff9933", linewidth=1.2)
        self.ax_phase.set_title("Phase Response", color="#cccccc")
        self.ax_phase.set_ylabel("Phase (rad)", color="#cccccc")
        self.ax_phase.set_xlim(0, fs / 2)
        self.ax_phase.grid(True, color="#444444", linewidth=0.5)
        self.ax_phase.tick_params(colors="#cccccc")
        for spine in self.ax_phase.spines.values():
            spine.set_color("#555555")

        ch_idx = self.channel_combo.currentIndex()
        x = self._audio_data.get_channel(ch_idx)

        max_display = int(fs * 5)
        if len(x) > max_display:
            x = x[:max_display]
        t = np.arange(len(x)) / fs

        if self._filtered_signal is not None:
            f_display = self._filtered_signal[:max_display] if len(self._filtered_signal) > max_display else self._filtered_signal
        else:
            f_display = None

        self.ax_time.clear()
        self.ax_time.set_facecolor("#1e1e1e")

        self.ax_time.plot(t, x, color="#888888", linewidth=0.5, alpha=0.6, label="Original")
        if f_display is not None:
            self.ax_time.plot(t, f_display, color="#00ff88", linewidth=0.7, label="Filtered")

        if f_display is not None:
            self.ax_time.legend(facecolor="#2b2b2b", edgecolor="#555555", labelcolor="#cccccc", fontsize=8)
        self.ax_time.set_title("Signal Comparison", color="#cccccc")
        self.ax_time.set_xlabel("Time (s)", color="#cccccc")
        self.ax_time.set_ylabel("Amplitude", color="#cccccc")
        self.ax_time.grid(True, color="#444444", linewidth=0.5)
        self.ax_time.tick_params(colors="#cccccc")
        for spine in self.ax_time.spines.values():
            spine.set_color("#555555")

        self.figure.tight_layout()
        self.canvas.draw()

    def _apply_filter(self):
        if self._audio_data is None:
            return
        try:
            b, a, is_fir = self._design_filter()
        except Exception:
            return

        ch_idx = self.channel_combo.currentIndex()
        x = self._audio_data.get_channel(ch_idx)

        if is_fir:
            self._filtered_signal = sig.lfilter(b, a, x)
        else:
            self._filtered_signal = sig.filtfilt(b, a, x)

        self.send_btn.setEnabled(True)
        self._update_preview()

    def _on_send_filtered(self):
        if self._audio_data is None or self._filtered_signal is None:
            return
        ch_idx = self.channel_combo.currentIndex()
        num_ch = self._audio_data.num_channels
        new_signal = np.zeros_like(self._audio_data.signal)
        for i in range(num_ch):
            if i == ch_idx:
                new_signal[:, i] = self._filtered_signal
            else:
                ch_data = self._audio_data.get_channel(i)
                try:
                    b, a, is_fir = self._design_filter()
                    if is_fir:
                        new_signal[:, i] = sig.lfilter(b, a, ch_data)
                    else:
                        new_signal[:, i] = sig.filtfilt(b, a, ch_data)
                except Exception:
                    new_signal[:, i] = ch_data

        src_name = self._audio_data.filename or "Unknown"
        ftype = self.filter_type_combo.currentText()
        method = self.method_combo.currentText().replace("IIR ", "").replace("FIR ", "")
        new_name = f"{src_name} [{ftype} {method}]"

        new_audio = AudioData(
            signal=new_signal,
            sample_rate=self._audio_data.sample_rate,
            channel_names=list(self._audio_data.channel_names) if self._audio_data.channel_names else [f"Ch {i}" for i in range(num_ch)],
            filename=new_name,
        )
        self.filter_applied.emit(new_audio)

    def _on_save_preset(self):
        name, ok = QInputDialog.getText(self, "Save Preset", "Preset name:", text="My Filter")
        if not ok or not name.strip():
            return
        name = name.strip()
        preset = {
            "filter_type": self.filter_type_combo.currentIndex(),
            "method": self.method_combo.currentIndex(),
            "order": self.order_spin.value(),
            "fc1": self.fc1_spin.value(),
            "fc2": self.fc2_spin.value(),
            "ripple": self.ripple_spin.value(),
            "rs": self.rs_spin.value(),
            "window": self.window_combo.currentIndex(),
            "kaiser_beta": self.kaiser_beta_spin.value(),
        }
        self._presets[name] = preset
        self._refresh_preset_list()

    def _refresh_preset_list(self):
        self.preset_list.clear()
        for name in sorted(self._presets.keys()):
            item = QListWidgetItem(name)
            self.preset_list.addItem(item)

    def _on_load_preset(self):
        item = self.preset_list.currentItem()
        if item is None:
            return
        self._apply_preset(item.text())

    def _on_preset_activated(self, item):
        self._apply_preset(item.text())

    def _apply_preset(self, name):
        if name not in self._presets:
            return
        p = self._presets[name]
        self.filter_type_combo.blockSignals(True)
        self.method_combo.blockSignals(True)
        self.order_spin.blockSignals(True)
        self.fc1_spin.blockSignals(True)
        self.fc2_spin.blockSignals(True)
        self.ripple_spin.blockSignals(True)
        self.rs_spin.blockSignals(True)
        self.window_combo.blockSignals(True)
        self.kaiser_beta_spin.blockSignals(True)

        self.filter_type_combo.setCurrentIndex(p["filter_type"])
        self.method_combo.setCurrentIndex(p["method"])
        self.order_spin.setValue(p["order"])
        self.fc1_spin.setValue(p["fc1"])
        self.fc2_spin.setValue(p["fc2"])
        self.ripple_spin.setValue(p["ripple"])
        self.rs_spin.setValue(p["rs"])
        self.window_combo.setCurrentIndex(p["window"])
        self.kaiser_beta_spin.setValue(p["kaiser_beta"])

        self.filter_type_combo.blockSignals(False)
        self.method_combo.blockSignals(False)
        self.order_spin.blockSignals(False)
        self.fc1_spin.blockSignals(False)
        self.fc2_spin.blockSignals(False)
        self.ripple_spin.blockSignals(False)
        self.rs_spin.blockSignals(False)
        self.window_combo.blockSignals(False)
        self.kaiser_beta_spin.blockSignals(False)

        self._update_method_visibility()
        self._update_cutoff_visibility()
        self._on_param_changed()

    def _on_del_preset(self):
        item = self.preset_list.currentItem()
        if item is None:
            return
        name = item.text()
        if name in self._presets:
            del self._presets[name]
            self._refresh_preset_list()

    def _on_save_result(self):
        if self._last_result is None:
            return
        r = self._last_result
        ftype = self.filter_type_combo.currentText()
        method = self.method_combo.currentText()
        data = {
            "freqs": r["freqs"],
            "mag_db": r["mag_db"],
            "phase": r["phase"],
            "b": r["b"],
            "a": r["a"],
            "is_fir": r["is_fir"],
        }
        params = {
            "filter_type": ftype,
            "method": method,
            "order": self.order_spin.value(),
            "fc1": self.fc1_spin.value(),
            "fc2": self.fc2_spin.value(),
        }
        self.save_result_requested.emit(data, params, f"Filter {ftype}")

    def restore_result(self, result: AnalysisResult):
        d = result.data
        freqs = np.array(d.get("freqs", []))
        mag_db = np.array(d.get("mag_db", []))
        phase = np.array(d.get("phase", []))
        self._last_result = {
            "freqs": freqs,
            "mag_db": mag_db,
            "phase": phase,
            "b": np.array(d.get("b", [1])),
            "a": np.array(d.get("a", [1])),
            "is_fir": d.get("is_fir", True),
        }
        self._create_subplots()
        self.ax_mag.plot(freqs, mag_db, color="#00ccff", linewidth=1.2)
        self.ax_mag.set_title("Frequency Response (Magnitude)", color="#cccccc")
        self.ax_phase.plot(freqs, phase, color="#ff9933", linewidth=1.2)
        self.ax_phase.set_title("Phase Response", color="#cccccc")
        self.figure.tight_layout()
        self.canvas.draw()

    def get_figure(self):
        return self.figure
