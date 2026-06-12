import os
import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QFileDialog, QMessageBox,
    QStatusBar, QToolBar, QLabel, QWidget, QVBoxLayout,
)
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtCore import Qt, QTimer

from audio_engine import AudioData, AudioFileLoader, MicrophoneRecorder
from waveform_panel import WaveformPanel
from spectrum_panel import SpectrumPanel
from filter_panel import FilterPanel
from timefreq_panel import TimeFreqPanel
from synthesis_panel import SynthesisPanel
from analysis_panel import AnalysisPanel
from export_utils import export_figure_to_png, export_data_to_csv


DARK_STYLE = """
QMainWindow {
    background-color: #1e1e1e;
}
QTabWidget::pane {
    border: 1px solid #3c3c3c;
    background-color: #1e1e1e;
}
QTabBar::tab {
    background-color: #2d2d2d;
    color: #cccccc;
    padding: 8px 18px;
    border: 1px solid #3c3c3c;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background-color: #1e1e1e;
    color: #ffffff;
    border-bottom: 2px solid #007acc;
}
QTabBar::tab:hover:!selected {
    background-color: #3c3c3c;
}
QToolBar {
    background-color: #2d2d2d;
    border: none;
    spacing: 6px;
    padding: 4px;
}
QToolBar QToolButton {
    background-color: #3c3c3c;
    color: #cccccc;
    border: 1px solid #4c4c4c;
    border-radius: 3px;
    padding: 5px 12px;
    min-width: 60px;
}
QToolBar QToolButton:hover {
    background-color: #505050;
}
QToolBar QToolButton:pressed {
    background-color: #007acc;
}
QToolBar QLabel {
    color: #cccccc;
    padding: 0 4px;
}
QStatusBar {
    background-color: #007acc;
    color: #ffffff;
    font-size: 12px;
}
QMenuBar {
    background-color: #2d2d2d;
    color: #cccccc;
    border-bottom: 1px solid #3c3c3c;
}
QMenuBar::item:selected {
    background-color: #3c3c3c;
}
QMenu {
    background-color: #2d2d2d;
    color: #cccccc;
    border: 1px solid #3c3c3c;
}
QMenu::item:selected {
    background-color: #094771;
}
QMessageBox {
    background-color: #1e1e1e;
    color: #cccccc;
}
QMessageBox QLabel {
    color: #cccccc;
}
QMessageBox QPushButton {
    background-color: #007acc;
    color: white;
    border: none;
    border-radius: 3px;
    padding: 5px 16px;
    min-width: 80px;
}
QFileDialog {
    background-color: #1e1e1e;
    color: #cccccc;
}
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._audio_data = None
        self._file_loader = None
        self._mic_recorder = None
        self._live_preview_timer = QTimer(self)
        self._live_preview_timer.setInterval(200)
        self._live_preview_timer.timeout.connect(self._on_live_preview_tick)
        self._setup_ui()
        self._setup_menu()
        self._setup_toolbar()
        self._setup_statusbar()
        self.setStyleSheet(DARK_STYLE)
        self.setWindowTitle("Signal Processing & Spectrum Analysis Workbench")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

    def _setup_ui(self):
        self._tabs = QTabWidget()
        self._tabs.setTabPosition(QTabWidget.TabPosition.North)
        self._tabs.setDocumentMode(True)

        self._waveform_panel = WaveformPanel()
        self._spectrum_panel = SpectrumPanel()
        self._filter_panel = FilterPanel()
        self._timefreq_panel = TimeFreqPanel()
        self._synthesis_panel = SynthesisPanel()
        self._analysis_panel = AnalysisPanel()

        self._tabs.addTab(self._waveform_panel, "📈 Waveform")
        self._tabs.addTab(self._spectrum_panel, "📊 Spectrum")
        self._tabs.addTab(self._filter_panel, "🔧 Filter Design")
        self._tabs.addTab(self._timefreq_panel, "🌊 Time-Frequency")
        self._tabs.addTab(self._synthesis_panel, "🎹 Synthesis")
        self._tabs.addTab(self._analysis_panel, "🔍 Peak / THD")

        self._synthesis_panel.signal_synthesized.connect(self._on_synthesized)

        self.setCentralWidget(self._tabs)

    def _setup_menu(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")
        open_action = QAction("&Open Audio File…", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._on_open_file)
        file_menu.addAction(open_action)

        self._record_action = QAction("⏺ Start &Recording", self)
        self._record_action.triggered.connect(self._on_toggle_recording)
        file_menu.addAction(self._record_action)

        file_menu.addSeparator()

        export_action = QAction("&Export Current View…", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self._on_export)
        file_menu.addAction(export_action)

        export_csv_action = QAction("Export &CSV Data…", self)
        export_csv_action.triggered.connect(self._on_export_csv)
        file_menu.addAction(export_csv_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        view_menu = menubar.addMenu("&View")
        self._tab_actions = {}
        for i in range(self._tabs.count()):
            tab_text = self._tabs.tabText(i)
            action = QAction(tab_text, self)
            action.triggered.connect(lambda checked, idx=i: self._tabs.setCurrentIndex(idx))
            view_menu.addAction(action)
            self._tab_actions[i] = action

        help_menu = menubar.addMenu("&Help")
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    def _setup_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        open_btn = QAction("📂 Open", self)
        open_btn.triggered.connect(self._on_open_file)
        toolbar.addAction(open_btn)

        self._rec_btn = QAction("⏺ Record", self)
        self._rec_btn.triggered.connect(self._on_toggle_recording)
        toolbar.addAction(self._rec_btn)

        toolbar.addSeparator()

        export_btn = QAction("💾 Export PNG", self)
        export_btn.triggered.connect(self._on_export)
        toolbar.addAction(export_btn)

        csv_btn = QAction("📄 Export CSV", self)
        csv_btn.triggered.connect(self._on_export_csv)
        toolbar.addAction(csv_btn)

        toolbar.addSeparator()

        self._file_label = QLabel("No file loaded")
        self._file_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        toolbar.addWidget(self._file_label)

    def _setup_statusbar(self):
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)
        self._statusbar.showMessage("Ready")

    def _set_audio_data(self, audio_data):
        self._audio_data = audio_data
        self._waveform_panel.set_data(audio_data)
        self._spectrum_panel.set_data(audio_data)
        self._filter_panel.set_data(audio_data)
        self._timefreq_panel.set_data(audio_data)
        self._analysis_panel.set_data(audio_data)
        if audio_data is not None:
            name = os.path.basename(audio_data.filename) if audio_data.filename else "Unknown"
            self._file_label.setText(
                f"{name} | {audio_data.sample_rate} Hz | "
                f"{audio_data.num_channels} ch | {audio_data.duration:.2f} s"
            )
            self._statusbar.showMessage(
                f"Loaded: {name} — {audio_data.num_samples} samples, "
                f"{audio_data.duration:.3f} s, {audio_data.sample_rate} Hz"
            )

    def _on_open_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Audio File", "",
            "Audio Files (*.wav *.mp3 *.flac *.ogg *.aiff);;All Files (*)"
        )
        if not filepath:
            return
        self._statusbar.showMessage(f"Loading: {os.path.basename(filepath)}…")
        self._file_loader = AudioFileLoader(filepath)
        self._file_loader.loaded.connect(self._on_file_loaded)
        self._file_loader.error.connect(self._on_file_error)
        self._file_loader.start()

    def _on_file_loaded(self, audio_data):
        self._set_audio_data(audio_data)
        self._file_loader = None

    def _on_file_error(self, err_msg):
        QMessageBox.critical(self, "Load Error", f"Failed to load audio file:\n{err_msg}")
        self._statusbar.showMessage("Load failed")
        self._file_loader = None

    def _on_toggle_recording(self):
        if self._mic_recorder is not None and self._mic_recorder.is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        self._mic_recorder = MicrophoneRecorder(sample_rate=44100, channels=2, blocksize=4096)
        try:
            self._mic_recorder.start()
            self._live_preview_timer.start()
            self._rec_btn.setText("⏹ Stop")
            self._record_action.setText("⏹ Stop &Recording")
            self._statusbar.showMessage("Recording… Click Stop to finish.")
        except Exception as e:
            QMessageBox.critical(self, "Recording Error", f"Could not start recording:\n{e}")
            self._mic_recorder = None

    def _on_live_preview_tick(self):
        if self._mic_recorder is None or not self._mic_recorder.is_recording:
            return
        chunk_data = self._mic_recorder.get_accumulated_chunks()
        if chunk_data is None:
            return
        if chunk_data.ndim == 1:
            chunk_data = chunk_data.reshape(-1, 1)
        num_ch = chunk_data.shape[1]
        ch_names = [f"Channel {i+1}" for i in range(num_ch)]
        if num_ch >= 2:
            ch_names[0] = "Left"
            ch_names[1] = "Right"

        if self._audio_data is not None and self._audio_data.filename == "Microphone Recording":
            new_signal = np.concatenate([self._audio_data.signal, chunk_data], axis=0)
        else:
            new_signal = chunk_data

        live_audio = AudioData(
            signal=new_signal, sample_rate=self._mic_recorder.sample_rate,
            channel_names=ch_names, filename="Microphone Recording"
        )
        self._audio_data = live_audio
        self._waveform_panel.set_data(live_audio)
        self._spectrum_panel.set_data(live_audio)
        self._statusbar.showMessage(f"Recording… {live_audio.duration:.1f} s")

    def _stop_recording(self):
        self._live_preview_timer.stop()
        if self._mic_recorder is None:
            return
        audio_data = self._mic_recorder.stop()
        self._mic_recorder = None
        self._rec_btn.setText("⏺ Record")
        self._record_action.setText("⏺ Start &Recording")
        if audio_data is not None:
            self._set_audio_data(audio_data)

    def _on_synthesized(self, audio_data):
        self._set_audio_data(audio_data)
        self._tabs.setCurrentIndex(0)
        self._statusbar.showMessage("Synthesized signal loaded into all panels")

    def _get_current_panel(self):
        idx = self._tabs.currentIndex()
        return self._tabs.widget(idx)

    def _on_export(self):
        panel = self._get_current_panel()
        if not hasattr(panel, 'get_figure'):
            QMessageBox.warning(self, "Export", "Current tab does not support export.")
            return
        figure = panel.get_figure()
        if figure is None:
            QMessageBox.warning(self, "Export", "No figure to export.")
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export PNG", "", "PNG Files (*.png);;All Files (*)"
        )
        if not filepath:
            return
        if not filepath.lower().endswith(".png"):
            filepath += ".png"
        try:
            export_figure_to_png(figure, filepath, dpi=300)
            self._statusbar.showMessage(f"Exported: {filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export:\n{e}")

    def _on_export_csv(self):
        panel = self._get_current_panel()
        if hasattr(panel, 'export_csv_data'):
            try:
                arrays, headers = panel.export_csv_data()
            except Exception as e:
                QMessageBox.warning(self, "Export CSV", f"Could not extract data:\n{e}")
                return
        elif self._audio_data is not None:
            headers = ["Time(s)"]
            arrays = [self._audio_data.time_array]
            for ch in range(self._audio_data.num_channels):
                name = (
                    self._audio_data.channel_names[ch]
                    if ch < len(self._audio_data.channel_names)
                    else f"Ch{ch}"
                )
                headers.append(name)
                arrays.append(self._audio_data.get_channel(ch))
        else:
            QMessageBox.warning(self, "Export CSV", "No data to export.")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not filepath:
            return
        if not filepath.lower().endswith(".csv"):
            filepath += ".csv"
        try:
            export_data_to_csv(filepath, *arrays, headers=headers)
            self._statusbar.showMessage(f"CSV exported: {filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export CSV:\n{e}")

    def _on_about(self):
        QMessageBox.about(
            self,
            "About",
            "<h3>Signal Processing & Spectrum Analysis Workbench</h3>"
            "<p>Interactive signal processing and frequency analysis tool.</p>"
            "<p>Built with PyQt6, NumPy, SciPy, Matplotlib, PyWavelets.</p>"
        )

    def closeEvent(self, event):
        if self._mic_recorder is not None and self._mic_recorder.is_recording:
            self._mic_recorder.stop()
        self._live_preview_timer.stop()
        event.accept()
