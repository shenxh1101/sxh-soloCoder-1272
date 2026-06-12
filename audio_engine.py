import numpy as np
import soundfile as sf
import sounddevice as sd
from PyQt6.QtCore import QObject, QThread, pyqtSignal, QTimer


class AudioData:
    def __init__(self, signal=None, sample_rate=44100, channel_names=None, filename=""):
        self.signal = signal if signal is not None else np.array([])
        self.sample_rate = sample_rate
        self.channel_names = channel_names or []
        self.filename = filename

    @property
    def num_channels(self):
        if self.signal.ndim == 1:
            return 1
        return self.signal.shape[1] if self.signal.ndim == 2 else 1

    @property
    def num_samples(self):
        return self.signal.shape[0]

    @property
    def duration(self):
        if self.sample_rate <= 0:
            return 0.0
        return self.num_samples / self.sample_rate

    @property
    def time_array(self):
        return np.arange(self.num_samples) / self.sample_rate

    def get_channel(self, ch):
        if self.signal.ndim == 1:
            return self.signal
        return self.signal[:, ch]

    def get_channel_range(self, ch, start_sample=0, end_sample=None):
        data = self.get_channel(ch)
        end_sample = end_sample or len(data)
        return data[start_sample:end_sample]


class AudioFileLoader(QThread):
    loaded = pyqtSignal(AudioData)
    error = pyqtSignal(str)

    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        self.last_error: str = ""
        self.audio_data: AudioData | None = None

    def run(self):
        try:
            signal, sr = sf.read(self.filepath, always_2d=False)
            if signal.ndim == 1:
                signal = signal.reshape(-1, 1)
            num_ch = signal.shape[1]
            ch_names = [f"Channel {i+1}" for i in range(num_ch)]
            if num_ch >= 2:
                ch_names[0] = "Left"
                ch_names[1] = "Right"
            audio = AudioData(signal=signal, sample_rate=sr,
                              channel_names=ch_names, filename=self.filepath)
            self.audio_data = audio
            self.loaded.emit(audio)
        except Exception as e:
            self.last_error = str(e)
            self.error.emit(str(e))


class MicrophoneRecorder(QObject):
    chunk_ready = pyqtSignal(np.ndarray, int)
    recording_stopped = pyqtSignal()

    def __init__(self, sample_rate=44100, channels=1, blocksize=4096):
        super().__init__()
        self.sample_rate = sample_rate
        self.channels = channels
        self.blocksize = blocksize
        self._recording = False
        self._stream = None
        self._buffer = []
        self._chunk_accumulator = []

    def start(self):
        self._recording = True
        self._buffer = []
        self._chunk_accumulator = []
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                blocksize=self.blocksize,
                callback=self._callback
            )
            self._stream.start()
        except Exception as e:
            self._recording = False
            raise e

    def _callback(self, indata, frames, time_info, status):
        if self._recording:
            chunk = indata.copy()
            self._buffer.append(chunk)
            self._chunk_accumulator.append(chunk)

    def get_accumulated_chunks(self):
        if not self._chunk_accumulator:
            return None
        data = np.concatenate(self._chunk_accumulator, axis=0)
        self._chunk_accumulator = []
        return data

    def stop(self):
        self._recording = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._buffer:
            full_signal = np.concatenate(self._buffer, axis=0)
            if full_signal.ndim == 1:
                full_signal = full_signal.reshape(-1, 1)
            num_ch = full_signal.shape[1]
            ch_names = [f"Channel {i+1}" for i in range(num_ch)]
            if num_ch >= 2:
                ch_names[0] = "Left"
                ch_names[1] = "Right"
            audio = AudioData(signal=full_signal, sample_rate=self.sample_rate,
                              channel_names=ch_names, filename="Microphone Recording")
            return audio
        self.recording_stopped.emit()
        return None

    @property
    def is_recording(self):
        return self._recording
