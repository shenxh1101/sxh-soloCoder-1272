import os
import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QFileDialog, QMessageBox, QInputDialog,
    QStatusBar, QToolBar, QLabel, QWidget, QVBoxLayout, QHBoxLayout,
    QListWidget, QListWidgetItem, QPushButton, QSplitter, QDoubleSpinBox,
    QFrame, QDialog, QAbstractItemView, QDialogButtonBox,
)
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtCore import Qt, QTimer, pyqtSignal

from audio_engine import AudioData, AudioFileLoader, MicrophoneRecorder
from waveform_panel import WaveformPanel
from spectrum_panel import SpectrumPanel
from filter_panel import FilterPanel
from timefreq_panel import TimeFreqPanel
from synthesis_panel import SynthesisPanel
from analysis_panel import AnalysisPanel
from results_manager import ResultsManager, AnalysisResult
from export_utils import export_figure_to_png, export_data_to_csv
from batch_dialog import BatchDialog
from report_generator import save_html_report


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
    analysis_range_changed = pyqtSignal(float, float)

    def __init__(self):
        super().__init__()
        self._audio_data = None
        self._file_loader = None
        self._mic_recorder = None
        self._results_manager = ResultsManager()
        self._analysis_range = (0.0, 0.0)
        self._live_preview_timer = QTimer(self)
        self._live_preview_timer.setInterval(200)
        self._live_preview_timer.timeout.connect(self._on_live_preview_tick)
        self._setup_ui()
        self._setup_menu()
        self._setup_toolbar()
        self._setup_range_toolbar()
        self._setup_statusbar()
        self.setStyleSheet(DARK_STYLE)
        self.setWindowTitle("Signal Processing & Spectrum Analysis Workbench")
        self.setMinimumSize(1400, 900)
        self.resize(1700, 1000)

    def _setup_ui(self):
        self._splitter = QSplitter(Qt.Orientation.Horizontal)

        self._sidebar = self._build_results_sidebar()
        self._sidebar.setMinimumWidth(220)
        self._sidebar.setMaximumWidth(320)

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
        self._filter_panel.filter_applied.connect(self._on_filter_applied)
        self._spectrum_panel.save_result_requested.connect(self._on_save_spectrum_result)
        self._analysis_panel.save_result_requested.connect(self._on_save_analysis_result)
        self._timefreq_panel.save_result_requested.connect(self._on_save_timefreq_result)
        self._filter_panel.save_result_requested.connect(self._on_save_filter_result)

        self._spectrum_panel.set_results_manager(self._results_manager)
        self._results_manager.add_callback(self._on_results_updated)

        self.analysis_range_changed.connect(self._spectrum_panel.set_analysis_range)
        self.analysis_range_changed.connect(self._timefreq_panel.set_analysis_range)
        self.analysis_range_changed.connect(self._analysis_panel.set_analysis_range)

        self._splitter.addWidget(self._sidebar)
        self._splitter.addWidget(self._tabs)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setSizes([260, 1400])

        self.setCentralWidget(self._splitter)

    def _build_results_sidebar(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)
        w.setStyleSheet("""
            QWidget { background-color: #252526; color: #cccccc; }
            QPushButton {
                background-color: #3c3c3c; color: #fff; border: 1px solid #555;
                border-radius: 3px; padding: 4px 8px; font-size: 11px;
            }
            QPushButton:hover { background-color: #505050; }
            QPushButton:pressed { background-color: #0e639c; }
            QListWidget {
                background-color: #1e1e1e; color: #cccccc;
                border: 1px solid #3c3c3c; font-size: 11px;
            }
            QListWidget::item { padding: 3px 6px; }
            QListWidget::item:selected { background-color: #094771; }
            QLabel { color: #cccccc; font-weight: bold; font-size: 12px; padding: 4px 0; }
            QFrame[class="h_line"] {
                background-color: #3c3c3c; max-height: 1px; min-height: 1px;
            }
        """)

        title = QLabel("📚 Analysis Results")
        lay.addWidget(title)

        btn_row = QHBoxLayout()
        self._res_save_btn = QPushButton("💾 Save Current")
        self._res_save_btn.clicked.connect(self._on_save_current_result)
        btn_row.addWidget(self._res_save_btn)

        self._res_del_btn = QPushButton("❌ Delete")
        self._res_del_btn.clicked.connect(self._on_delete_result)
        btn_row.addWidget(self._res_del_btn)
        lay.addLayout(btn_row)

        btn_row2 = QHBoxLayout()
        self._res_load_btn = QPushButton("⬇️ Load Set")
        self._res_load_btn.clicked.connect(self._on_load_results)
        btn_row2.addWidget(self._res_load_btn)

        self._res_store_btn = QPushButton("⬆️ Save Set")
        self._res_store_btn.clicked.connect(self._on_store_results)
        btn_row2.addWidget(self._res_store_btn)
        lay.addLayout(btn_row2)

        hline = QFrame()
        hline.setProperty("class", "h_line")
        hline.setFrameShape(QFrame.Shape.HLine)
        hline.setFrameShadow(QFrame.Shadow.Sunken)
        lay.addWidget(hline)

        self._results_list = QListWidget()
        self._results_list.itemDoubleClicked.connect(self._on_result_double_clicked)
        lay.addWidget(self._results_list, stretch=1)

        lay.addStretch()
        return w

    def _setup_range_toolbar(self):
        toolbar = QToolBar("Analysis Range")
        toolbar.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

        toolbar.addWidget(QLabel(" 🔍 Analysis Range:"))
        toolbar.addWidget(QLabel("Start:"))
        self._range_start_spin = QDoubleSpinBox()
        self._range_start_spin.setRange(0.0, 99999.0)
        self._range_start_spin.setValue(0.0)
        self._range_start_spin.setSuffix(" s")
        self._range_start_spin.setDecimals(3)
        self._range_start_spin.setStyleSheet("""
            QDoubleSpinBox { background-color: #2d2d2d; color: #fff;
            border: 1px solid #555; border-radius: 3px; padding: 3px 6px; }
        """)
        toolbar.addWidget(self._range_start_spin)

        toolbar.addWidget(QLabel("  End:"))
        self._range_end_spin = QDoubleSpinBox()
        self._range_end_spin.setRange(0.0, 99999.0)
        self._range_end_spin.setValue(0.0)
        self._range_end_spin.setSuffix(" s")
        self._range_end_spin.setDecimals(3)
        self._range_end_spin.setStyleSheet("""
            QDoubleSpinBox { background-color: #2d2d2d; color: #fff;
            border: 1px solid #555; border-radius: 3px; padding: 3px 6px; }
        """)
        toolbar.addWidget(self._range_end_spin)

        self._apply_range_btn = QPushButton("✅ Apply Range")
        self._apply_range_btn.setStyleSheet("""
            QPushButton { background-color: #0e639c; color: #fff;
            border: none; border-radius: 3px; padding: 4px 12px; margin: 0 8px; }
            QPushButton:hover { background-color: #1177bb; }
        """)
        self._apply_range_btn.clicked.connect(self._on_apply_range)
        toolbar.addWidget(self._apply_range_btn)

        self._reset_range_btn = QPushButton("🔄 Full Signal")
        self._reset_range_btn.setStyleSheet("""
            QPushButton { background-color: #555; color: #fff;
            border: none; border-radius: 3px; padding: 4px 12px; }
            QPushButton:hover { background-color: #666; }
        """)
        self._reset_range_btn.clicked.connect(self._on_reset_range)
        toolbar.addWidget(self._reset_range_btn)

        toolbar.addSeparator()
        self._range_info_label = QLabel("No signal loaded")
        self._range_info_label.setStyleSheet("color: #aaaaaa; font-size: 11px; margin-left: 10px;")
        toolbar.addWidget(self._range_info_label)

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

        batch_action = QAction("&Batch Analysis…", self)
        batch_action.triggered.connect(self._on_batch_analysis)
        file_menu.addAction(batch_action)

        report_action = QAction("&Export HTML Report…", self)
        report_action.triggered.connect(self._on_export_report)
        file_menu.addAction(report_action)

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

        batch_btn = QAction("📋 Batch", self)
        batch_btn.triggered.connect(self._on_batch_analysis)
        toolbar.addAction(batch_btn)

        report_btn = QAction("📑 Report", self)
        report_btn.triggered.connect(self._on_export_report)
        toolbar.addAction(report_btn)

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
            self._range_start_spin.setMaximum(audio_data.duration)
            self._range_end_spin.setMaximum(audio_data.duration)
            self._range_start_spin.setValue(0.0)
            self._range_end_spin.setValue(audio_data.duration)
            self._analysis_range = (0.0, audio_data.duration)
            self._update_range_info()
        else:
            self._range_info_label.setText("No signal loaded")

    def _update_range_info(self):
        if self._audio_data is None:
            return
        t0, t1 = self._analysis_range
        sr = self._audio_data.sample_rate
        samples = int((t1 - t0) * sr)
        self._range_info_label.setText(
            f"Range: {t0:.3f}s - {t1:.3f}s  |  {samples:,} samples  |  {(t1-t0):.2f}s duration"
        )

    def _on_apply_range(self):
        if self._audio_data is None:
            QMessageBox.warning(self, "Range", "Load an audio file first.")
            return
        t0 = self._range_start_spin.value()
        t1 = self._range_end_spin.value()
        if t1 <= t0:
            QMessageBox.warning(self, "Range", "End must be greater than Start.")
            return
        if t1 > self._audio_data.duration:
            t1 = self._audio_data.duration
            self._range_end_spin.setValue(t1)
        self._analysis_range = (t0, t1)
        self._update_range_info()
        self.analysis_range_changed.emit(t0, t1)
        self._statusbar.showMessage(f"Analysis range set to {t0:.3f}s - {t1:.3f}s")

    def _on_reset_range(self):
        if self._audio_data is None:
            return
        self._range_start_spin.setValue(0.0)
        self._range_end_spin.setValue(self._audio_data.duration)
        self._analysis_range = (0.0, self._audio_data.duration)
        self._update_range_info()
        self.analysis_range_changed.emit(0.0, self._audio_data.duration)
        self._statusbar.showMessage("Analysis range reset to full signal")

    def _on_filter_applied(self, audio_data):
        self._set_audio_data(audio_data)
        self._tabs.setCurrentIndex(0)
        self._statusbar.showMessage("Filtered signal loaded into all panels")

    def _on_results_updated(self, results):
        self._refresh_results_list()
        self._spectrum_panel._refresh_saved_combos()

    def _refresh_results_list(self):
        self._results_list.clear()
        for r in self._results_manager.get_results():
            label = f"[{r.result_type}] {r.name}"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, r.result_id)
            self._results_list.addItem(item)

    def _on_save_current_result(self):
        idx = self._tabs.currentIndex()
        if idx == 1:
            self._spectrum_panel._on_save_result()
        elif idx == 3:
            self._timefreq_panel._on_save_result()
        elif idx == 5:
            self._analysis_panel._on_save_result()
        elif idx == 2:
            self._filter_panel._on_save_result()
        else:
            QMessageBox.information(self, "Save Result", "Current tab does not support saving results.")

    def _on_save_spectrum_result(self, data, params, name_suggestion):
        self._save_result("spectrum", data, params, name_suggestion)

    def _on_save_analysis_result(self, data, params, name_suggestion):
        self._save_result("analysis", data, params, name_suggestion)

    def _on_save_timefreq_result(self, data, params, name_suggestion):
        self._save_result("timefreq", data, params, name_suggestion)

    def _on_save_filter_result(self, data, params, name_suggestion):
        self._save_result("filter", data, params, name_suggestion)

    def _save_result(self, rtype, data, params, name_suggestion):
        name, ok = QInputDialog.getText(self, "Save Result", "Result name:", text=name_suggestion)
        if not ok or not name.strip():
            return
        src = self._audio_data.filename if self._audio_data else ""
        r = self._results_manager.add_result(rtype, name.strip(), data, params, src)
        self._refresh_results_list()
        self._statusbar.showMessage(f"Saved result: {r.name}")

    def _on_delete_result(self):
        item = self._results_list.currentItem()
        if item is None:
            return
        rid = item.data(Qt.ItemDataRole.UserRole)
        self._results_manager.remove_result(rid)
        self._refresh_results_list()

    def _on_store_results(self):
        if len(self._results_manager.get_results()) == 0:
            QMessageBox.information(self, "Save Results", "No results to save.")
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Results Set", "", "JSON Files (*.json);;All Files (*)"
        )
        if not filepath:
            return
        if not filepath.lower().endswith(".json"):
            filepath += ".json"
        try:
            self._results_manager.save_to_json(filepath)
            self._statusbar.showMessage(f"Saved results to {filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save results:\n{e}")

    def _on_load_results(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Results Set", "", "JSON Files (*.json);;All Files (*)"
        )
        if not filepath:
            return
        try:
            self._results_manager.load_from_json(filepath)
            self._refresh_results_list()
            self._statusbar.showMessage(f"Loaded results from {filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load results:\n{e}")

    def _on_result_double_clicked(self, item):
        rid = item.data(Qt.ItemDataRole.UserRole)
        r = self._results_manager.get_result(rid)
        if r is None:
            return
        if r.result_type == "spectrum":
            self._tabs.setCurrentIndex(1)
            self._spectrum_panel.restore_result(r)
        elif r.result_type == "timefreq":
            self._tabs.setCurrentIndex(3)
            self._timefreq_panel.restore_result(r)
        elif r.result_type == "analysis":
            self._tabs.setCurrentIndex(5)
            self._analysis_panel.restore_result(r)
        elif r.result_type == "filter":
            self._tabs.setCurrentIndex(2)
            self._filter_panel.restore_result(r)

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

    def _on_batch_analysis(self):
        dlg = BatchDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        results = dlg.get_completed_results()
        if not results:
            return
        count = 0
        for res in results:
            src = res.get("source_file", "")
            ch_name = res.get("channel_name", "Ch 0")
            if "spectrum" in res:
                s = res["spectrum"]
                data = {
                    "view": "Magnitude",
                    "x": s["freqs"],
                    "y": s["magnitude_db"],
                    "channel": ch_name,
                }
                self._results_manager.add_result(
                    "spectrum", f"{os.path.basename(src)} {ch_name} Mag", data, {"source": "batch"}, src
                )
                count += 1
            if "stft" in res:
                s = res["stft"]
                data = {
                    "type": "stft",
                    "times": s["times"],
                    "freqs": s["freqs"],
                    "power_db": s["power_db"],
                    "channel": ch_name,
                }
                self._results_manager.add_result(
                    "timefreq", f"{os.path.basename(src)} {ch_name} STFT",
                    data, {"params": s["params"]}, src
                )
                count += 1
            if "wavelet" in res:
                s = res["wavelet"]
                data = {
                    "type": "wavelet",
                    "times": s["times"],
                    "freqs": s["freqs"],
                    "power_db": s["power_db"],
                    "channel": ch_name,
                }
                self._results_manager.add_result(
                    "timefreq", f"{os.path.basename(src)} {ch_name} Wavelet",
                    data, {"params": s["params"]}, src
                )
                count += 1
            if "thd" in res:
                s = res["thd"]
                peaks = []
                for i in range(min(len(s["peak_freqs"]), len(s["peak_mags"]))):
                    peaks.append({"freq": float(s["peak_freqs"][i]), "mag": float(s["peak_mags"][i])})
                data = {
                    "freqs": s["freqs"],
                    "spectrum_db": s["spectrum_db"],
                    "peak_indices": s["peak_indices"],
                    "channel": ch_name,
                    "peaks": peaks,
                    "thd": f"{s['thd']:.4f}%",
                    "thdn": f"{s['thdn']:.4f}%",
                }
                self._results_manager.add_result(
                    "analysis", f"{os.path.basename(src)} {ch_name} THD",
                    data, {"source": "batch"}, src
                )
                count += 1
        self._refresh_results_list()
        self._statusbar.showMessage(f"Batch analysis imported {count} results")

    def _on_export_report(self):
        all_results = self._results_manager.get_results()
        if not all_results:
            QMessageBox.information(self, "Report", "No saved results to include in the report.")
            return

        from PyQt6.QtWidgets import QDialog, QDialogButtonBox
        sel_dlg = QDialog(self)
        sel_dlg.setWindowTitle("Select Results for Report")
        sel_dlg.setMinimumWidth(400)
        sel_dlg.setMinimumHeight(400)
        sel_dlg.setStyleSheet("""
            QDialog { background-color: #252526; color: #cccccc; }
            QListWidget { background-color: #1e1e1e; color: #cccccc;
                         border: 1px solid #3c3c3c; }
            QListWidget::item { padding: 3px 6px; }
            QListWidget::item:selected { background-color: #094771; }
            QLabel { color: #cccccc; }
            QPushButton { background-color: #3c3c3c; color: white;
                          border: 1px solid #555; border-radius: 3px; padding: 6px 14px; }
            QPushButton:hover { background-color: #505050; }
        """)
        lay = QVBoxLayout(sel_dlg)
        lay.addWidget(QLabel("Select results to include (Ctrl/Shift for multi-select):"))
        lw = QListWidget()
        lw.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        for r in all_results:
            it = QListWidgetItem(f"[{r.result_type}] {r.name}")
            it.setData(Qt.ItemDataRole.UserRole, r.result_id)
            lw.addItem(it)
            it.setSelected(True)
        lay.addWidget(lw, stretch=1)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(sel_dlg.accept)
        btns.rejected.connect(sel_dlg.reject)
        lay.addWidget(btns)
        if sel_dlg.exec() != QDialog.DialogCode.Accepted:
            return

        selected_ids = [it.data(Qt.ItemDataRole.UserRole) for it in lw.selectedItems()]
        selected_results = [self._results_manager.get_result(rid) for rid in selected_ids]
        selected_results = [r for r in selected_results if r is not None]
        if not selected_results:
            QMessageBox.information(self, "Report", "No results selected.")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export HTML Report", "", "HTML Files (*.html);;All Files (*)"
        )
        if not filepath:
            return
        try:
            save_html_report(selected_results, filepath,
                              title="Signal Analysis Report",
                              author="Signal Analysis Workbench")
            self._statusbar.showMessage(f"Report exported: {filepath}")
            QMessageBox.information(self, "Report", f"Report saved:\n{filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save report:\n{e}")

    def closeEvent(self, event):
        if self._mic_recorder is not None and self._mic_recorder.is_recording:
            self._mic_recorder.stop()
        self._live_preview_timer.stop()
        event.accept()
