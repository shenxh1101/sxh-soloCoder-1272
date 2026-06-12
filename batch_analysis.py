from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks, stft, windows as sigwin, welch
import pywt
from PyQt6.QtCore import QThread, pyqtSignal
from audio_engine import AudioFileLoader, AudioData


@dataclass
class BatchTask:
    filepath: str
    channel: int = 0
    t_start: float = 0.0
    t_end: float = 0.0
    do_spectrum: bool = True
    do_stft: bool = True
    do_wavelet: bool = False
    do_thd: bool = True
    status: str = "pending"

    def name(self) -> str:
        return os.path.basename(self.filepath)


class BatchWorker(QThread):
    task_started = pyqtSignal(int)
    task_finished = pyqtSignal(int, dict)
    task_error = pyqtSignal(int, str)
    all_finished = pyqtSignal()
    log = pyqtSignal(str)

    def __init__(self, tasks: List[BatchTask]):
        super().__init__()
        self.tasks = tasks
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        for i, task in enumerate(self.tasks):
            if self._stop:
                break
            self.task_started.emit(i)
            self.log.emit(f"Processing {i+1}/{len(self.tasks)}: {task.name()}")
            try:
                results = self._process_one(task)
                self.task_finished.emit(i, results)
                self.log.emit(f"✓ Done: {task.name()}")
            except Exception as e:
                self.task_error.emit(i, str(e))
                self.log.emit(f"✗ Failed: {task.name()} — {e}")
        self.all_finished.emit()

    def _load(self, task: BatchTask) -> AudioData:
        loader = AudioFileLoader(task.filepath)
        loader.run()
        if loader.last_error:
            raise RuntimeError(loader.last_error)
        if loader.audio_data is None:
            raise RuntimeError("No audio data loaded")
        return loader.audio_data

    def _process_one(self, task: BatchTask) -> dict:
        ad = self._load(task)
        sr = ad.sample_rate
        sig = ad.get_channel(task.channel)
        if task.t_end > 0 and task.t_end > task.t_start:
            i0 = int(task.t_start * sr)
            i1 = int(task.t_end * sr)
            sig = sig[max(0, i0):min(len(sig), i1)]
        n = len(sig)
        out: Dict[str, Any] = {
            "source_file": task.filepath,
            "sample_rate": sr,
            "num_samples": n,
            "duration": n / sr,
            "channel": task.channel,
            "channel_name": (
                ad.channel_names[task.channel]
                if ad.channel_names and task.channel < len(ad.channel_names)
                else f"Ch {task.channel}"
            ),
        }

        if task.do_spectrum:
            out["spectrum"] = self._spectrum(sig, sr)
        if task.do_stft:
            out["stft"] = self._stft(sig, sr, task.t_start)
        if task.do_wavelet:
            out["wavelet"] = self._wavelet(sig, sr, task.t_start)
        if task.do_thd:
            out["thd"] = self._thd(sig, sr)

        return out

    def _spectrum(self, sig, sr):
        max_n = 262144
        s = sig[-max_n:] if len(sig) > max_n else sig
        nfft = 1
        while nfft < len(s):
            nfft <<= 1
        w = sigwin.hann(len(s))
        padded = np.zeros(nfft)
        padded[:len(s)] = s * w
        fft = np.abs(rfft(padded)) * 2.0 / np.sum(w)
        freqs = rfftfreq(nfft, d=1.0 / sr)
        step = max(1, len(freqs) // 8000)
        return {
            "freqs": freqs[::step],
            "magnitude_db": 20 * np.log10(np.abs(fft[::step]) + 1e-12),
            "phase": np.unwrap(np.angle(fft[::step])),
        }

    def _stft(self, sig, sr, t_offset):
        nperseg = min(1024, len(sig) // 4) if len(sig) >= 4096 else 256
        noverlap = int(nperseg * 0.75)
        freqs, times, Zxx = stft(sig, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap)
        power_db = 10.0 * np.log10(np.abs(Zxx) ** 2 + 1e-12)
        return {
            "times": times + t_offset,
            "freqs": freqs,
            "power_db": power_db,
            "params": {"nperseg": nperseg, "noverlap": noverlap, "window": "hann"},
        }

    def _wavelet(self, sig, sr, t_offset):
        scales = np.arange(1, 65)
        wavelet = "cmor1.5-1.0"
        max_s = min(len(sig), sr * 5)
        s_use = sig[:max_s]
        target_nt = 500
        if len(s_use) > target_nt:
            step = len(s_use) // target_nt
            s_use = s_use[::step]
        coeffs, freqs = pywt.cwt(s_use, scales, wavelet, 1.0 / sr)
        power_db = 10.0 * np.log10(np.abs(coeffs) ** 2 + 1e-12)
        duration = len(s_use) / sr
        times = np.linspace(t_offset, t_offset + duration, power_db.shape[1])
        return {
            "times": times,
            "freqs": freqs,
            "power_db": power_db,
            "params": {"wavelet": wavelet, "scales": scales},
        }

    def _thd(self, sig, sr):
        max_fft = 65536
        n = len(sig)
        step = max(1, n // max_fft)
        s = sig[::step]
        sr_eff = sr / step
        n_eff = len(s)
        w = sigwin.hann(n_eff)
        spectrum = np.abs(rfft(s * w)) * 2.0 / np.sum(w)
        freqs = rfftfreq(n_eff, d=1.0 / sr_eff)
        spectrum_db = 20 * np.log10(np.maximum(spectrum, 1e-12))
        peak_idx, _ = find_peaks(spectrum_db, height=-80, distance=3, prominence=3)
        peak_idx = peak_idx[np.argsort(spectrum_db[peak_idx])[::-1]]
        peak_freqs = freqs[peak_idx]
        peak_mags = spectrum_db[peak_idx]

        if len(peak_idx) >= 2:
            fund = peak_freqs[0]
            harm_powers = []
            for p in peak_freqs:
                ratio = p / fund
                if abs(ratio - round(ratio)) < 0.02 and round(ratio) > 1:
                    idx = np.argmin(np.abs(freqs - p))
                    harm_powers.append(spectrum[idx] ** 2)
            fund_power = spectrum[peak_idx[0]] ** 2
            if harm_powers and fund_power > 0:
                thd = 100 * np.sqrt(np.sum(harm_powers)) / fund_power
            else:
                thd = 0.0
        else:
            thd = 0.0

        noise_power = np.sum(spectrum ** 2) - (spectrum[peak_idx[0]] ** 2 if len(peak_idx) > 0 else 0)
        total_power = np.sum(spectrum ** 2)
        thdn = 100 * np.sqrt(max(0.0, noise_power)) / (spectrum[peak_idx[0]] if len(peak_idx) > 0 else 1.0) if total_power > 0 else 0.0

        return {
            "freqs": freqs,
            "spectrum_db": spectrum_db,
            "peak_indices": peak_idx,
            "peak_freqs": peak_freqs,
            "peak_mags": peak_mags,
            "thd": thd,
            "thdn": thdn,
        }
