"""
Project 66: Stream Handler & Voice Activity Buffer for Real-Time Transcriber
Manages incoming real-time audio chunks, audio buffering, RMS silence detection,
and progressive transcript generation with timestamps.
"""

import io
import time
import numpy as np
import soundfile as sf
import librosa


class AudioStreamBuffer:
    """
    Rolling circular audio buffer that accumulates streamed audio frames,
    detects voice activity (RMS), and extracts overlapping windows for ASR.
    """
    def __init__(self, sample_rate: int = 16000, min_chunk_seconds: float = 1.0, max_buffer_seconds: float = 10.0):
        self.sample_rate = sample_rate
        self.min_samples = int(sample_rate * min_chunk_seconds)
        self.max_samples = int(sample_rate * max_buffer_seconds)
        self.buffer = np.array([], dtype=np.float32)
        self.total_processed_samples = 0
        self.silence_threshold_rms = 0.008

    def push_audio_bytes(self, audio_bytes: bytes) -> None:
        """
        Accepts raw audio bytes (WAV, OGG, or WebM PCM) and appends float32 audio.
        """
        try:
            audio, sr = sf.read(io.BytesIO(audio_bytes))
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            self.buffer = np.concatenate([self.buffer, audio.astype(np.float32)])
        except Exception:
            # Fallback raw 16-bit PCM interpretation
            raw = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            self.buffer = np.concatenate([self.buffer, raw])

        # Prevent buffer overflow
        if len(self.buffer) > self.max_samples:
            overflow = len(self.buffer) - self.max_samples
            self.buffer = self.buffer[overflow:]
            self.total_processed_samples += overflow

    def push_audio_array(self, audio_array: np.ndarray, sr: int = 16000) -> None:
        """Appends a numpy audio array."""
        if audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=1)
        if sr != self.sample_rate:
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=self.sample_rate)
        self.buffer = np.concatenate([self.buffer, audio_array.astype(np.float32)])

    def get_rms_energy(self) -> float:
        """Calculates RMS audio level for silence detection."""
        if len(self.buffer) == 0:
            return 0.0
        return float(np.sqrt(np.mean(self.buffer ** 2)))

    def is_silent(self) -> bool:
        """Returns True if current buffer level is below silence threshold."""
        return self.get_rms_energy() < self.silence_threshold_rms

    def get_current_window(self) -> np.ndarray:
        """Returns the current audio window for transcription."""
        return self.buffer.copy()

    def get_audio_bytes_for_asr(self) -> bytes:
        """Encodes current buffer as WAV bytes for speech recognizer."""
        out = io.BytesIO()
        sf.write(out, self.buffer, self.sample_rate, format="WAV", subtype="PCM_16")
        return out.getvalue()

    def consume(self, seconds_to_keep: float = 0.5) -> None:
        """Keeps only the last `seconds_to_keep` seconds as context overlap."""
        samples_to_keep = int(self.sample_rate * seconds_to_keep)
        if len(self.buffer) > samples_to_keep:
            consumed = len(self.buffer) - samples_to_keep
            self.total_processed_samples += consumed
            self.buffer = self.buffer[-samples_to_keep:]

    def clear(self) -> None:
        """Resets the buffer."""
        self.buffer = np.array([], dtype=np.float32)
        self.total_processed_samples = 0


class StreamingTranscriptSession:
    """
    Maintains session transcript state, segments with timestamps,
    word count, and live status.
    """
    def __init__(self, session_id: str = "default_session"):
        self.session_id = session_id
        self.segments = []
        self.full_transcript = ""
        self.start_time = time.time()
        self.last_update_time = time.time()
        self.is_active = True

    def add_segment(self, text: str, confidence: float = 0.95, start_sec: float = None, end_sec: float = None):
        text = text.strip()
        if not text:
            return
        
        now = time.time() - self.start_time
        if start_sec is None:
            start_sec = max(0.0, now - 2.0)
        if end_sec is None:
            end_sec = now

        # Deduplicate consecutive identical segments
        if self.segments and self.segments[-1]["text"].lower() == text.lower():
            return

        seg = {
            "index": len(self.segments) + 1,
            "text": text,
            "confidence": round(confidence, 3),
            "start_sec": round(start_sec, 2),
            "end_sec": round(end_sec, 2),
            "timestamp": time.strftime("%H:%M:%S")
        }
        self.segments.append(seg)
        self.full_transcript = " ".join(s["text"] for s in self.segments)
        self.last_update_time = time.time()

    def get_summary(self) -> dict:
        words = self.full_transcript.split()
        duration = round(time.time() - self.start_time, 1)
        wpm = round((len(words) / (duration / 60.0)), 1) if duration > 5 else 0.0

        return {
            "session_id": self.session_id,
            "full_transcript": self.full_transcript,
            "word_count": len(words),
            "duration_sec": duration,
            "words_per_minute": wpm,
            "total_segments": len(self.segments),
            "segments": self.segments,
            "is_active": self.is_active
        }

    def reset(self) -> None:
        self.segments = []
        self.full_transcript = ""
        self.start_time = time.time()
        self.last_update_time = time.time()
