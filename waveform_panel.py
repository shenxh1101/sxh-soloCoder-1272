import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox,
    QPushButton, QCheckBox, QLabel, QScrollArea, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.widgets import SpanSelector

_CHANNEL_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
]

_DECIMATION_THRESHOLD = 50000


def _decimate_minmax(data, max_points):
    if len(data) <= max_points:
        return np.arange(len(data)), data
    block_size = len(data) // max_points
    n_blocks = len(data) // block_size
    trimmed = data[:n_blocks * block_size].reshape(n_blocks, block_size)
    mins = trimmed.min(axis=1)
    maxs = trimmed.max(axis=1)
    indices = np.arange(n_blocks) * block_size
    out_idx = np.empty(2 * n_blocks, dtype=int)
    out_idx[0::2] = indices
    out_idx[1::2] = indices + block_size - 1
    out_vals = np.empty(2 * n_blocks, dtype=data.dtype)
    out_vals[0::2] = mins
    out_vals[1::2] = maxs
    return out_idx, out_vals


class WaveformPanel(QWidget):
    time_range_selected = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._audio_data = None
        self._display_mode = "Separate"
        self._selected_channels = set()
        self._zoom_ranges = {}
        self._span_selector = None
        self._cursor_line = None
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)

        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Separate", "Overlay"])
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
        toolbar.addWidget(QLabel("Display:"))
        toolbar.addWidget(self._mode_combo)

        toolbar.addSpacing(12)

        ch_label = QLabel("Channels:")
        toolbar.addWidget(ch_label)

        self._ch_scroll = QScrollArea()
        self._ch_scroll.setWidgetResizable(True)
        self._ch_scroll.setMaximumHeight(36)
        self._ch_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._ch_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._ch_container = QWidget()
        self._ch_layout = QHBoxLayout(self._ch_container)
        self._ch_layout.setContentsMargins(0, 0, 0, 0)
        self._ch_layout.setSpacing(4)
        self._ch_scroll.setWidget(self._ch_container)
        self._ch_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        toolbar.addWidget(self._ch_scroll, stretch=1)

        toolbar.addSpacing(12)

        self._reset_btn = QPushButton("Reset Zoom")
        self._reset_btn.clicked.connect(self._reset_zoom)
        toolbar.addWidget(self._reset_btn)

        main_layout.addLayout(toolbar)

        self._figure = Figure(facecolor="#1e1e1e", dpi=100)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._mpl_toolbar = NavigationToolbar2QT(self._canvas, self)
        self._mpl_toolbar.setStyleSheet("background-color: #2d2d2d; color: #cccccc;")
        main_layout.addWidget(self._mpl_toolbar)
        main_layout.addWidget(self._canvas, stretch=1)

        self._status_label = QLabel("No data loaded")
        self._status_label.setStyleSheet(
            "color: #aaaaaa; background-color: #1e1e1e; "
            "padding: 2px 6px; font-family: monospace; font-size: 11px;"
        )
        main_layout.addWidget(self._status_label)

        self._canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.setStyleSheet("background-color: #1e1e1e;")

    def set_data(self, audio_data):
        self._audio_data = audio_data
        self._zoom_ranges = {}
        self._selected_channels.clear()
        self._rebuild_channel_checkboxes()
        if audio_data is not None:
            for i in range(audio_data.num_channels):
                self._selected_channels.add(i)
        self._plot_later()

    def _plot_later(self):
        if not hasattr(self, '_plot_timer'):
            self._plot_timer = None
        if self._plot_timer is not None:
            self._plot_timer.stop()
        from PyQt6.QtCore import QTimer
        self._plot_timer = QTimer(self)
        self._plot_timer.setSingleShot(True)
        self._plot_timer.setInterval(50)
        self._plot_timer.timeout.connect(self._do_plot)
        self._plot_timer.start()

    def _do_plot(self):
        self._plot_timer = None
        self._plot()

    def _rebuild_channel_checkboxes(self):
        while self._ch_layout.count():
            item = self._ch_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        if self._audio_data is None:
            return

        for i in range(self._audio_data.num_channels):
            name = (
                self._audio_data.channel_names[i]
                if i < len(self._audio_data.channel_names)
                else f"Ch {i + 1}"
            )
            cb = QCheckBox(name)
            color = _CHANNEL_COLORS[i % len(_CHANNEL_COLORS)]
            cb.setStyleSheet(
                f"QCheckBox {{ color: {color}; spacing: 4px; }}"
                f"QCheckBox::indicator {{ width: 12px; height: 12px; }}"
            )
            cb.setChecked(i in self._selected_channels)
            cb.stateChanged.connect(lambda state, ch=i: self._on_channel_toggle(ch, state))
            self._ch_layout.addWidget(cb)

    def _on_channel_toggle(self, ch, state):
        if state == Qt.CheckState.Checked.value:
            self._selected_channels.add(ch)
        else:
            self._selected_channels.discard(ch)
        self._plot()

    def _on_mode_changed(self, text):
        self._display_mode = text
        self._zoom_ranges = {}
        self._plot()

    def _reset_zoom(self):
        self._zoom_ranges = {}
        self._plot()

    def _plot(self):
        self._figure.clear()
        if self._span_selector is not None:
            self._span_selector = None

        if self._audio_data is None or len(self._selected_channels) == 0:
            ax = self._figure.add_subplot(111)
            ax.set_facecolor("#2d2d2d")
            ax.tick_params(colors="#888888")
            ax.set_xlabel("Time (s)", color="#aaaaaa")
            ax.set_ylabel("Amplitude", color="#aaaaaa")
            ax.text(
                0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color="#666666", fontsize=14,
            )
            for spine in ax.spines.values():
                spine.set_color("#444444")
            self._canvas.draw_idle()
            self._update_status(None, None)
            return

        sorted_chs = sorted(self._selected_channels)

        if self._display_mode == "Overlay":
            self._plot_overlay(sorted_chs)
        else:
            self._plot_separate(sorted_chs)

        self._canvas.draw_idle()
        self._update_status_after_plot()

    def _plot_overlay(self, channels):
        ax = self._figure.add_subplot(111)
        ax.set_facecolor("#2d2d2d")
        ax.tick_params(colors="#888888")
        ax.set_xlabel("Time (s)", color="#aaaaaa")
        ax.set_ylabel("Amplitude", color="#aaaaaa")
        for spine in ax.spines.values():
            spine.set_color("#444444")

        for ch in channels:
            self._draw_channel_on_ax(ax, ch)

        zoom = self._zoom_ranges.get("overlay")
        if zoom is not None:
            ax.set_xlim(zoom[0], zoom[1])

        self._setup_span_selector(ax, "overlay")

    def _plot_separate(self, channels):
        n = len(channels)
        axes = self._figure.subplots(n, 1, sharex=True, squeeze=False)[:, 0]

        for idx, ch in enumerate(channels):
            ax = axes[idx]
            ax.set_facecolor("#2d2d2d")
            ax.tick_params(colors="#888888")
            for spine in ax.spines.values():
                spine.set_color("#444444")

            name = (
                self._audio_data.channel_names[ch]
                if ch < len(self._audio_data.channel_names)
                else f"Ch {ch + 1}"
            )
            ax.set_ylabel(name, color="#aaaaaa", fontsize=9)

            self._draw_channel_on_ax(ax, ch)

            zoom = self._zoom_ranges.get(ch)
            if zoom is not None:
                ax.set_xlim(zoom[0], zoom[1])

            self._setup_span_selector(ax, ch)

        axes[-1].set_xlabel("Time (s)", color="#aaaaaa")
        self._figure.subplots_adjust(hspace=0.3, left=0.1, right=0.98, top=0.96, bottom=0.08)

    def _draw_channel_on_ax(self, ax, ch):
        data = self._audio_data.get_channel(ch)
        color = _CHANNEL_COLORS[ch % len(_CHANNEL_COLORS)]
        n_samples = len(data)
        sr = self._audio_data.sample_rate

        if n_samples > _DECIMATION_THRESHOLD:
            display_pts = _DECIMATION_THRESHOLD // 2
            block_size = max(1, n_samples // display_pts)
            n_blocks = n_samples // block_size
            trimmed = data[:n_blocks * block_size].reshape(n_blocks, block_size)
            mins = trimmed.min(axis=1)
            maxs = trimmed.max(axis=1)
            t = (np.arange(n_blocks) * block_size + block_size // 2) / sr
            ax.fill_between(t, mins, maxs, color=color, alpha=0.35, linewidth=0)
            ax.plot(t, mins, color=color, linewidth=0.5, alpha=0.6)
            ax.plot(t, maxs, color=color, linewidth=0.5, alpha=0.6)
        else:
            t = np.arange(n_samples) / sr
            ax.plot(t, data, color=color, linewidth=0.8)

    def _setup_span_selector(self, ax, key):
        span = SpanSelector(
            ax,
            lambda xmin, xmax, k=key: self._on_span_select(k, xmin, xmax),
            "horizontal",
            useblit=True,
            props=dict(facecolor="#ffffff", alpha=0.15, edgecolor="#ffffff", linewidth=0.5),
            handle_props=dict(color="#cccccc", linewidth=1),
            minspan=0.001,
            interactive=False,
        )
        if self._span_selector is None or not isinstance(self._span_selector, list):
            self._span_selector = []
        self._span_selector.append(span)

    def _on_span_select(self, key, xmin, xmax):
        self._zoom_ranges[key] = (xmin, xmax)
        self.time_range_selected.emit(xmin, xmax)
        self._plot()

    def _on_mouse_move(self, event):
        if event.inaxes is None:
            return
        x = event.xdata
        if x is None:
            return
        xlim = event.inaxes.get_xlim()
        self._update_status(xlim[0], xlim[1], cursor=x)

    def _update_status_after_plot(self):
        if self._audio_data is None:
            self._update_status(None, None)
            return
        all_axes = self._figure.get_axes()
        if not all_axes:
            self._update_status(None, None)
            return
        xlim = all_axes[0].get_xlim()
        self._update_status(xlim[0], xlim[1])

    def _update_status(self, t_start, t_end, cursor=None):
        parts = []
        if t_start is not None and t_end is not None:
            parts.append(f"Visible: {t_start:.4f} s — {t_end:.4f} s")
            duration = self._audio_data.duration if self._audio_data else 0
            parts.append(f"of {duration:.4f} s")
        if cursor is not None:
            parts.append(f"Cursor: {cursor:.6f} s")
        if not parts:
            self._status_label.setText("No data loaded")
        else:
            self._status_label.setText("  |  ".join(parts))

    def get_figure(self):
        return self._figure
