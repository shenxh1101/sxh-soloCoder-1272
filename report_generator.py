from __future__ import annotations
import os
import io
import base64
from datetime import datetime
from typing import List
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from results_manager import AnalysisResult


def _fig_to_png_base64(fig: Figure, dpi: int = 110) -> str:
    canvas = FigureCanvasAgg(fig)
    buf = io.BytesIO()
    fig.patch.set_facecolor("white")
    for ax in fig.axes:
        ax.set_facecolor("white")
        ax.tick_params(colors="black")
        ax.xaxis.label.set_color("black")
        ax.yaxis.label.set_color("black")
        ax.title.set_color("black")
        for spine in ax.spines.values():
            spine.set_color("#333333")
    canvas.print_png(buf, dpi=dpi)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _build_spectrum_fig(r: AnalysisResult) -> Figure:
    fig = Figure(figsize=(9, 4), dpi=100, facecolor="white")
    ax = fig.add_subplot(111)
    d = r.data
    x = np.array(d.get("x", []))
    y = np.array(d.get("y", []))
    view = d.get("view", "Magnitude")
    ax.plot(x, y, color="#1e6fba", linewidth=0.9)
    ax.set_xlabel("Frequency (Hz)")
    if view == "Magnitude":
        ax.set_ylabel("Magnitude (dB)")
        ax.set_title("Magnitude Spectrum")
    elif view == "Phase":
        ax.set_ylabel("Phase (radians)")
        ax.set_title("Phase Spectrum")
    else:
        ax.set_ylabel("PSD (dB/Hz)")
        ax.set_title("Power Spectral Density")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    return fig


def _build_timefreq_fig(r: AnalysisResult) -> Figure:
    fig = Figure(figsize=(9, 4), dpi=100, facecolor="white")
    ax = fig.add_subplot(111)
    d = r.data
    rtype = d.get("type", "stft")
    times = np.array(d.get("times", []))
    freqs = np.array(d.get("freqs", []))
    power_db = np.array(d.get("power_db", []))
    cmap = "viridis" if rtype == "stft" else "magma"
    pcm = ax.pcolormesh(times, freqs, power_db, shading="gouraud", cmap=cmap)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("STFT Spectrogram" if rtype == "stft" else "Wavelet Scalogram")
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label("Power (dB)")
    fig.tight_layout()
    return fig


def _build_filter_fig(r: AnalysisResult) -> Figure:
    fig = Figure(figsize=(9, 5), dpi=100, facecolor="white")
    ax_mag = fig.add_subplot(211)
    ax_phase = fig.add_subplot(212)
    d = r.data
    freqs = np.array(d.get("freqs", []))
    mag_db = np.array(d.get("mag_db", []))
    phase = np.array(d.get("phase", []))
    ax_mag.plot(freqs, mag_db, color="#1e6fba", linewidth=1.0)
    ax_mag.set_title("Frequency Response (Magnitude)")
    ax_mag.set_ylabel("Magnitude (dB)")
    ax_mag.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax_phase.plot(freqs, phase, color="#c44e00", linewidth=1.0)
    ax_phase.set_title("Phase Response")
    ax_phase.set_xlabel("Frequency (Hz)")
    ax_phase.set_ylabel("Phase (rad)")
    ax_phase.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    return fig


def _build_analysis_fig(r: AnalysisResult) -> Figure:
    fig = Figure(figsize=(9, 4), dpi=100, facecolor="white")
    ax = fig.add_subplot(111)
    d = r.data
    freqs = np.array(d.get("freqs", []))
    spectrum_db = np.array(d.get("spectrum_db", []))
    peak_indices = np.array(d.get("peak_indices", [])).astype(int)
    ax.plot(freqs, spectrum_db, color="#1e6fba", linewidth=0.7, alpha=0.8)
    if len(peak_indices) > 0:
        ax.plot(freqs[peak_indices], spectrum_db[peak_indices], "rx", markersize=5)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("Peak Detection")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    return fig


def _params_table(params: dict) -> str:
    if not params:
        return "<p><em>No parameters recorded.</em></p>"
    rows = ""
    for k, v in params.items():
        if isinstance(v, float):
            v_str = f"{v:.6g}"
        elif isinstance(v, dict):
            v_str = ", ".join(f"{kk}={vv}" for kk, vv in v.items())
        elif isinstance(v, (list, np.ndarray)):
            v_str = f"[{len(v)} items]"
        else:
            v_str = str(v)
        rows += f"<tr><td><code>{k}</code></td><td>{v_str}</td></tr>"
    return f"""<table class="params">
        <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
        <tbody>{rows}</tbody></table>"""


def _thd_table(r: AnalysisResult) -> str:
    d = r.data
    peaks = d.get("peaks", [])
    thd = d.get("thd", "—")
    thdn = d.get("thdn", "—")
    if not peaks:
        return f"<p><strong>THD:</strong> {thd} &nbsp;&nbsp; <strong>THD+N:</strong> {thdn}</p>"
    rows = ""
    for i, p in enumerate(peaks):
        rows += f"<tr><td>{i+1}</td><td>{p.get('freq', 0):.3f}</td><td>{p.get('mag', 0):.2f}</td></tr>"
    return f"""<p><strong>THD:</strong> {thd} &nbsp;&nbsp; <strong>THD+N:</strong> {thdn}</p>
    <table class="peaks">
        <thead><tr><th>#</th><th>Frequency (Hz)</th><th>Magnitude (dB)</th></tr></thead>
        <tbody>{rows}</tbody></table>"""


def generate_html_report(results: List[AnalysisResult], title: str = "Signal Analysis Report",
                         author: str = "") -> str:
    sections = []
    grouped: dict = {}
    for r in results:
        key = r.source_file or "(no source)"
        grouped.setdefault(key, []).append(r)

    for src, rs in grouped.items():
        file_sec = f"<h2>File: <code>{os.path.basename(src) if src else src}</code></h2>"
        if src:
            file_sec += f"<p class=\"muted\">Full path: <code>{src}</code></p>"
        for r in rs:
            fig_html = ""
            try:
                if r.result_type == "spectrum":
                    fig = _build_spectrum_fig(r)
                    fig_html = f"<p><img src=\"{_fig_to_png_base64(fig)}\" alt=\"{r.name}\" /></p>"
                elif r.result_type in ("timefreq",):
                    fig = _build_timefreq_fig(r)
                    fig_html = f"<p><img src=\"{_fig_to_png_base64(fig)}\" alt=\"{r.name}\" /></p>"
                elif r.result_type == "filter":
                    fig = _build_filter_fig(r)
                    fig_html = f"<p><img src=\"{_fig_to_png_base64(fig)}\" alt=\"{r.name}\" /></p>"
                elif r.result_type == "analysis":
                    fig = _build_analysis_fig(r)
                    fig_html = f"<p><img src=\"{_fig_to_png_base64(fig)}\" alt=\"{r.name}\" /></p>"
            except Exception as e:
                fig_html = f"<p class=\"warn\">Could not render figure: {e}</p>"

            extras = ""
            if r.result_type == "analysis":
                extras = _thd_table(r)

            file_sec += f"""<div class="result">
                <h3>{r.result_type.upper()} — {r.name}</h3>
                <p class="muted">Saved: {datetime.fromtimestamp(r.timestamp).strftime('%Y-%m-%d %H:%M:%S')}</p>
                {fig_html}
                <h4>Parameters</h4>
                {_params_table(r.params if r.params else {})}
                {extras}
                </div>"""
        sections.append(file_sec)

    body = "\n".join(sections) if sections else "<p><em>No results selected.</em></p>"
    author_line = f"<p class=\"muted\">Generated by: {author}</p>" if author else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
body {{ font-family: -apple-system, "Segoe UI", Arial, sans-serif;
        background: #ffffff; color: #222; max-width: 960px; margin: 24px auto; padding: 0 16px; }}
h1 {{ border-bottom: 2px solid #1e6fba; padding-bottom: 6px; color: #1e6fba; }}
h2 {{ color: #222; margin-top: 32px; }}
h3 {{ color: #333; }}
h4 {{ color: #555; margin-top: 14px; margin-bottom: 6px; }}
p {{ line-height: 1.5; }}
.muted {{ color: #777; font-size: 0.9em; }}
.warn {{ color: #c44e00; }}
.result {{ border: 1px solid #ddd; border-radius: 6px; padding: 10px 16px;
           margin: 12px 0; background: #fafafa; }}
img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
table.params, table.peaks {{ border-collapse: collapse; margin: 8px 0; width: auto; }}
table.params th, table.params td, table.peaks th, table.peaks td {{
    border: 1px solid #ccc; padding: 4px 10px; text-align: left; font-size: 0.95em; }}
table.params th, table.peaks th {{ background: #eef3fa; color: #222; }}
code {{ background: #f0f0f0; padding: 1px 4px; border-radius: 3px;
       font-family: Consolas, Menlo, monospace; font-size: 0.95em; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="muted">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
{author_line}
{body}
</body></html>"""


def save_html_report(results: List[AnalysisResult], output_path: str,
                     title: str = "Signal Analysis Report", author: str = "") -> str:
    html = generate_html_report(results, title, author)
    if not output_path.lower().endswith(".html"):
        output_path += ".html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path
