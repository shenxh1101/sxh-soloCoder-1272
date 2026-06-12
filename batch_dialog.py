import os
from typing import List
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QCheckBox, QSpinBox, QDoubleSpinBox,
    QFileDialog, QMessageBox, QProgressBar, QPlainTextEdit, QGroupBox,
    QFormLayout, QWidget, QAbstractItemView,
)
from PyQt6.QtCore import Qt
from batch_analysis import BatchTask, BatchWorker


class BatchDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Analysis")
        self.setMinimumWidth(780)
        self.setMinimumHeight(540)
        self.tasks: List[BatchTask] = []
        self._worker = None
        self.setStyleSheet("""
            QDialog { background-color: #252526; color: #cccccc; }
            QGroupBox { border: 1px solid #3c3c3c; border-radius: 4px;
                        margin-top: 8px; padding-top: 14px; font-weight: bold; }
            QGroupBox::title { left: 10px; padding: 0 4px; }
            QPushButton { background-color: #3c3c3c; color: white;
                          border: 1px solid #555; border-radius: 3px; padding: 6px 14px; }
            QPushButton:hover { background-color: #505050; }
            QPushButton#primary { background-color: #0e639c; }
            QPushButton#primary:hover { background-color: #1177bb; }
            QPushButton:disabled { color: #777; background-color: #2a2a2a; }
            QListWidget { background-color: #1e1e1e; color: #cccccc;
                         border: 1px solid #3c3c3c; font-size: 11px; }
            QListWidget::item { padding: 3px 6px; }
            QListWidget::item:selected { background-color: #094771; }
            QLabel { color: #cccccc; }
            QCheckBox { color: #cccccc; }
            QSpinBox, QDoubleSpinBox { background-color: #2d2d2d; color: #fff;
                border: 1px solid #555; border-radius: 3px; padding: 2px 4px; }
            QPlainTextEdit { background-color: #1e1e1e; color: #ccc; border: 1px solid #3c3c3c; }
            QProgressBar { background-color: #2d2d2d; color: white; border: 1px solid #555;
                          border-radius: 3px; text-align: center; height: 18px; }
            QProgressBar::chunk { background-color: #0e639c; }
        """)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)

        files_box = QGroupBox("Files to process")
        f_lay = QVBoxLayout(files_box)
        btn_row = QHBoxLayout()
        self.add_btn = QPushButton("➕ Add Files…")
        self.add_btn.clicked.connect(self._on_add_files)
        btn_row.addWidget(self.add_btn)
        self.remove_btn = QPushButton("➖ Remove")
        self.remove_btn.clicked.connect(self._on_remove)
        btn_row.addWidget(self.remove_btn)
        self.clear_btn = QPushButton("🗑️ Clear")
        self.clear_btn.clicked.connect(self._on_clear)
        btn_row.addWidget(self.clear_btn)
        btn_row.addStretch()
        f_lay.addLayout(btn_row)
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.file_list.currentItemChanged.connect(self._on_selection_changed)
        f_lay.addWidget(self.file_list, stretch=1)
        root.addWidget(files_box, stretch=2)

        params_box = QGroupBox("Per-file parameters (applies to selected file)")
        p_lay = QFormLayout(params_box)
        self.channel_spin = QSpinBox()
        self.channel_spin.setRange(0, 32)
        self.channel_spin.valueChanged.connect(self._apply_params_to_selected)
        p_lay.addRow("Channel index:", self.channel_spin)

        self.t_start_spin = QDoubleSpinBox()
        self.t_start_spin.setRange(0.0, 99999.0)
        self.t_start_spin.setDecimals(3)
        self.t_start_spin.setSuffix(" s")
        self.t_start_spin.valueChanged.connect(self._apply_params_to_selected)
        p_lay.addRow("Start time:", self.t_start_spin)

        self.t_end_spin = QDoubleSpinBox()
        self.t_end_spin.setRange(0.0, 99999.0)
        self.t_end_spin.setDecimals(3)
        self.t_end_spin.setSuffix(" s")
        self.t_end_spin.setValue(0.0)
        self.t_end_spin.valueChanged.connect(self._apply_params_to_selected)
        p_lay.addRow("End time (0 = full):", self.t_end_spin)

        checks_row = QHBoxLayout()
        self.c_spectrum = QCheckBox("Spectrum")
        self.c_spectrum.setChecked(True)
        self.c_spectrum.stateChanged.connect(self._apply_params_to_selected)
        self.c_stft = QCheckBox("STFT")
        self.c_stft.setChecked(True)
        self.c_stft.stateChanged.connect(self._apply_params_to_selected)
        self.c_wavelet = QCheckBox("Wavelet")
        self.c_wavelet.stateChanged.connect(self._apply_params_to_selected)
        self.c_thd = QCheckBox("Peak / THD")
        self.c_thd.setChecked(True)
        self.c_thd.stateChanged.connect(self._apply_params_to_selected)
        checks_row.addWidget(self.c_spectrum)
        checks_row.addWidget(self.c_stft)
        checks_row.addWidget(self.c_wavelet)
        checks_row.addWidget(self.c_thd)
        checks_row.addStretch()
        checks_wrap = QWidget()
        checks_wrap.setLayout(checks_row)
        p_lay.addRow("Analyses:", checks_wrap)
        root.addWidget(params_box)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Progress and status messages will appear here…")
        self.log.setFixedHeight(120)
        root.addWidget(self.log)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        root.addWidget(self.progress)

        action_row = QHBoxLayout()
        action_row.addStretch()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel)
        action_row.addWidget(self.cancel_btn)
        self.run_btn = QPushButton("▶ Run Analysis")
        self.run_btn.setObjectName("primary")
        self.run_btn.clicked.connect(self._on_run)
        action_row.addWidget(self.run_btn)
        root.addLayout(action_row)

    def _on_add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Audio Files", "",
            "Audio Files (*.wav *.mp3 *.flac *.ogg *.aiff);;All Files (*)"
        )
        for f in files:
            if any(t.filepath == f for t in self.tasks):
                continue
            t = BatchTask(filepath=f)
            self.tasks.append(t)
            self._append_item(t)

    def _append_item(self, t: BatchTask):
        item = QListWidgetItem(f"[{t.status}] {t.name()} — channel {t.channel}")
        item.setData(Qt.ItemDataRole.UserRole, t)
        self.file_list.addItem(item)

    def _on_remove(self):
        rows = sorted({self.file_list.row(i) for i in self.file_list.selectedItems()}, reverse=True)
        for r in rows:
            self.file_list.takeItem(r)
            del self.tasks[r]

    def _on_clear(self):
        self.tasks.clear()
        self.file_list.clear()

    def _on_selection_changed(self, cur, prev):
        if cur is None:
            return
        t: BatchTask = cur.data(Qt.ItemDataRole.UserRole)
        if t is None:
            return
        self.channel_spin.blockSignals(True)
        self.t_start_spin.blockSignals(True)
        self.t_end_spin.blockSignals(True)
        self.c_spectrum.blockSignals(True)
        self.c_stft.blockSignals(True)
        self.c_wavelet.blockSignals(True)
        self.c_thd.blockSignals(True)

        self.channel_spin.setValue(t.channel)
        self.t_start_spin.setValue(t.t_start)
        self.t_end_spin.setValue(t.t_end)
        self.c_spectrum.setChecked(t.do_spectrum)
        self.c_stft.setChecked(t.do_stft)
        self.c_wavelet.setChecked(t.do_wavelet)
        self.c_thd.setChecked(t.do_thd)

        self.channel_spin.blockSignals(False)
        self.t_start_spin.blockSignals(False)
        self.t_end_spin.blockSignals(False)
        self.c_spectrum.blockSignals(False)
        self.c_stft.blockSignals(False)
        self.c_wavelet.blockSignals(False)
        self.c_thd.blockSignals(False)

    def _apply_params_to_selected(self):
        items = self.file_list.selectedItems() or [self.file_list.item(i) for i in range(self.file_list.count())]
        for it in items:
            t: BatchTask = it.data(Qt.ItemDataRole.UserRole)
            if t is None:
                continue
            t.channel = self.channel_spin.value()
            t.t_start = self.t_start_spin.value()
            t.t_end = self.t_end_spin.value()
            t.do_spectrum = self.c_spectrum.isChecked()
            t.do_stft = self.c_stft.isChecked()
            t.do_wavelet = self.c_wavelet.isChecked()
            t.do_thd = self.c_thd.isChecked()
            it.setText(f"[{t.status}] {t.name()} — channel {t.channel}")

    def _on_cancel(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._log("Stopping…")
        else:
            self.reject()

    def _on_run(self):
        if not self.tasks:
            QMessageBox.information(self, "Batch", "No files in the queue.")
            return
        self.run_btn.setEnabled(False)
        self.add_btn.setEnabled(False)
        self.progress.setMaximum(len(self.tasks))
        self.progress.setValue(0)
        self._log(f"Starting batch of {len(self.tasks)} files…")

        self._worker = BatchWorker(self.tasks)
        self._worker.task_started.connect(self._on_task_started)
        self._worker.task_finished.connect(self._on_task_finished)
        self._worker.task_error.connect(self._on_task_error)
        self._worker.log.connect(self._log)
        self._worker.all_finished.connect(self._on_all_finished)
        self._worker.start()

    def _log(self, msg: str):
        self.log.appendPlainText(msg)

    def _on_task_started(self, idx: int):
        self.tasks[idx].status = "running"
        self._refresh_item(idx)

    def _on_task_finished(self, idx: int, results: dict):
        self.tasks[idx].status = "done"
        self.tasks[idx]._results = results
        self._refresh_item(idx)
        self.progress.setValue(self.progress.value() + 1)

    def _on_task_error(self, idx: int, err: str):
        self.tasks[idx].status = f"error: {err[:40]}"
        self._refresh_item(idx)
        self.progress.setValue(self.progress.value() + 1)

    def _on_all_finished(self):
        self.run_btn.setEnabled(True)
        self.add_btn.setEnabled(True)
        self._log("Batch complete.")
        QMessageBox.information(self, "Batch", "Batch analysis completed. Results are available for import.")
        self.accept()

    def _refresh_item(self, idx: int):
        if idx < 0 or idx >= self.file_list.count():
            return
        it = self.file_list.item(idx)
        t = self.tasks[idx]
        it.setText(f"[{t.status}] {t.name()} — channel {t.channel}")

    def get_completed_results(self):
        out = []
        for t in self.tasks:
            if hasattr(t, "_results"):
                out.append(t._results)
        return out
