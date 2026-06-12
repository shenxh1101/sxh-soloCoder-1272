import numpy as np
from scipy import signal as sp_signal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QComboBox, QDoubleSpinBox, QPushButton, QHeaderView, QGroupBox,
    QFormLayout, QLabel, QAbstractItemView,
)
from PyQt6.QtCore import pyqtSignal, Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from audio_engine import AudioData


class SynthesisPanel(QWidget):
    signal_synthesized = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)

        top = QVBoxLayout()
        top.addWidget(self._build_component_group())
        top.addWidget(self._build_modulation_group())
        top.addWidget(self._build_global_group())

        btn_row = QHBoxLayout()
        self.btn_generate = QPushButton("Generate")
        self.btn_generate.clicked.connect(self._on_generate)
        self.btn_send = QPushButton("Send to Main")
        self.btn_send.clicked.connect(self._on_send)
        btn_row.addWidget(self.btn_generate)
        btn_row.addWidget(self.btn_send)
        top.addLayout(btn_row)

        self.canvas = FigureCanvasQTAgg(Figure(figsize=(8, 5), tight_layout=True))
        self.ax_time = self.canvas.figure.add_subplot(2, 1, 1)
        self.ax_freq = self.canvas.figure.add_subplot(2, 1, 2)

        root.addLayout(top, 1)
        root.addWidget(self.canvas, 3)

    def _build_component_group(self):
        grp = QGroupBox("Signal Components")
        lay = QVBoxLayout(grp)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Type", "Frequency(Hz)", "Amplitude", "Phase(°)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        lay.addWidget(self.table)

        btn_row = QHBoxLayout()
        btn_add = QPushButton("Add Component")
        btn_add.clicked.connect(self._add_component_row)
        btn_remove = QPushButton("Remove Component")
        btn_remove.clicked.connect(self._remove_component_row)
        btn_row.addWidget(btn_add)
        btn_row.addWidget(btn_remove)
        lay.addLayout(btn_row)

        self._add_component_row()
        return grp

    def _add_component_row(self):
        row = self.table.rowCount()
        self.table.insertRow(row)

        combo_type = QComboBox()
        combo_type.addItems(["Sine", "Square", "Sawtooth", "Noise"])
        self.table.setCellWidget(row, 0, combo_type)

        spin_freq = QDoubleSpinBox()
        spin_freq.setRange(1, 20000)
        spin_freq.setValue(440)
        spin_freq.setSuffix(" Hz")
        spin_freq.setDecimals(1)
        self.table.setCellWidget(row, 1, spin_freq)

        spin_amp = QDoubleSpinBox()
        spin_amp.setRange(0, 1)
        spin_amp.setValue(0.5)
        spin_amp.setSingleStep(0.05)
        spin_amp.setDecimals(2)
        self.table.setCellWidget(row, 2, spin_amp)

        spin_phase = QDoubleSpinBox()
        spin_phase.setRange(0, 360)
        spin_phase.setValue(0)
        spin_phase.setSuffix("°")
        spin_phase.setDecimals(1)
        self.table.setCellWidget(row, 3, spin_phase)

    def _remove_component_row(self):
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)

    def _build_modulation_group(self):
        grp = QGroupBox("Modulation")
        form = QFormLayout(grp)

        self.combo_mod = QComboBox()
        self.combo_mod.addItems(["None", "AM", "FM"])
        self.combo_mod.currentIndexChanged.connect(self._on_mod_type_changed)
        form.addRow("Type:", self.combo_mod)

        self.spin_mod_freq = QDoubleSpinBox()
        self.spin_mod_freq.setRange(1, 5000)
        self.spin_mod_freq.setValue(10)
        self.spin_mod_freq.setSuffix(" Hz")
        self.spin_mod_freq.setDecimals(1)
        self.spin_mod_freq.setEnabled(False)
        form.addRow("Mod Frequency:", self.spin_mod_freq)

        self.spin_mod_depth = QDoubleSpinBox()
        self.spin_mod_depth.setRange(0, 10)
        self.spin_mod_depth.setValue(1.0)
        self.spin_mod_depth.setSingleStep(0.1)
        self.spin_mod_depth.setDecimals(2)
        self.spin_mod_depth.setEnabled(False)
        form.addRow("Depth / Index:", self.spin_mod_depth)

        return grp

    def _on_mod_type_changed(self, idx):
        enabled = idx != 0
        self.spin_mod_freq.setEnabled(enabled)
        self.spin_mod_depth.setEnabled(enabled)

    def _build_global_group(self):
        grp = QGroupBox("Global Settings")
        form = QFormLayout(grp)

        self.spin_duration = QDoubleSpinBox()
        self.spin_duration.setRange(0.1, 60)
        self.spin_duration.setValue(1.0)
        self.spin_duration.setSuffix(" s")
        self.spin_duration.setDecimals(2)
        form.addRow("Duration:", self.spin_duration)

        self.combo_sr = QComboBox()
        self.combo_sr.addItems(["8000", "16000", "22050", "44100", "48000", "96000"])
        self.combo_sr.setCurrentText("44100")
        form.addRow("Sample Rate:", self.combo_sr)

        return grp

    def _synthesize(self):
        duration = self.spin_duration.value()
        sr = int(self.combo_sr.currentText())
        t = np.arange(0, duration, 1.0 / sr)

        sig = np.zeros_like(t)
        for row in range(self.table.rowCount()):
            wave_type = self.table.cellWidget(row, 0).currentText()
            freq = self.table.cellWidget(row, 1).value()
            amp = self.table.cellWidget(row, 2).value()
            phase_deg = self.table.cellWidget(row, 3).value()
            phase_rad = np.deg2rad(phase_deg)

            if wave_type == "Sine":
                comp = amp * np.sin(2 * np.pi * freq * t + phase_rad)
            elif wave_type == "Square":
                comp = amp * sp_signal.square(2 * np.pi * freq * t + phase_rad)
            elif wave_type == "Sawtooth":
                comp = amp * sp_signal.sawtooth(2 * np.pi * freq * t + phase_rad)
            elif wave_type == "Noise":
                comp = np.random.normal(0, amp, len(t))
            else:
                comp = np.zeros_like(t)

            sig = sig + comp

        mod_type = self.combo_mod.currentText()
        if mod_type == "AM":
            fm = self.spin_mod_freq.value()
            depth = self.spin_mod_depth.value()
            sig = sig * (1 + depth * np.cos(2 * np.pi * fm * t))
        elif mod_type == "FM":
            fm = self.spin_mod_freq.value()
            depth = self.spin_mod_depth.value()
            carrier_freq = 440.0
            for row in range(self.table.rowCount()):
                wave_type = self.table.cellWidget(row, 0).currentText()
                if wave_type != "Noise":
                    carrier_freq = self.table.cellWidget(row, 1).value()
                    break
            instantaneous_phase = 2 * np.pi * carrier_freq * t + depth * np.sin(2 * np.pi * fm * t)
            sig = np.sin(instantaneous_phase)

        return t, sig, sr

    def _on_generate(self):
        t, sig, sr = self._synthesize()

        self.ax_time.clear()
        self.ax_time.plot(t, sig, linewidth=0.6)
        self.ax_time.set_xlabel("Time (s)")
        self.ax_time.set_ylabel("Amplitude")
        self.ax_time.set_title("Time Domain")

        self.ax_freq.clear()
        n = len(sig)
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        spectrum = np.abs(np.fft.rfft(sig)) / n
        self.ax_freq.plot(freqs, spectrum, linewidth=0.6)
        self.ax_freq.set_xlabel("Frequency (Hz)")
        self.ax_freq.set_ylabel("Magnitude")
        self.ax_freq.set_title("Frequency Spectrum")

        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def _on_send(self):
        _, sig, sr = self._synthesize()
        signal_2d = sig.reshape(-1, 1)
        audio_data = AudioData(
            signal=signal_2d, sample_rate=sr,
            channel_names=["Synthesized"], filename="Synthesized Signal"
        )
        self.signal_synthesized.emit(audio_data)

    def get_figure(self):
        return self.canvas.figure
