"""
Project 66: Real-Time Audio Transcriber Model Pipeline
Optimized for low-latency streaming ASR using Whisper (base/tiny)
and Google SpeechRecognition fallback.
"""

import io
import time
import numpy as np
import soundfile as sf
import speech_recognition as sr
from typing import Dict, Any, Optional

from project66_realtime_transcriber.stream_handler import AudioStreamBuffer, StreamingTranscriptSession

# Lazy-loaded fast Whisper model
_fast_whisper_model = None

def get_fast_whisper():
    global _fast_whisper_model
    if _fast_whisper_model is None:
        try:
            import whisper
            print("[Project 66] Loading Whisper model for real-time streaming...")
            _fast_whisper_model = whisper.load_model("base")
            print("[Project 66] Real-time Whisper model ready.")
        except Exception as e:
            print(f"[Project 66] Warning: Failed to load Whisper ({e}). Using SpeechRecognition.")
            _fast_whisper_model = None
    return _fast_whisper_model


class StreamingTranscriber:
    """
    Streaming transcription engine supporting continuous audio feeds,
    session management, silence gating, and live segment updates.
    """
    def __init__(self, default_engine: str = "whisper"):
        self.default_engine = default_engine
        self.sessions: Dict[str, StreamingTranscriptSession] = {}
        self.buffers: Dict[str, AudioStreamBuffer] = {}
        self.recognizer = sr.Recognizer()

    def get_or_create_session(self, session_id: str = "default") -> tuple[StreamingTranscriptSession, AudioStreamBuffer]:
        if session_id not in self.sessions:
            self.sessions[session_id] = StreamingTranscriptSession(session_id)
            self.buffers[session_id] = AudioStreamBuffer()
        return self.sessions[session_id], self.buffers[session_id]

    def process_chunk(self, session_id: str, audio_bytes: bytes, engine: Optional[str] = None) -> Dict[str, Any]:
        """
        Accepts a newly streamed chunk of audio bytes, updates the buffer,
        runs VAD silence detection, and transcribes when speech is present.
        """
        session, buffer = self.get_or_create_session(session_id)
        buffer.push_audio_bytes(audio_bytes)
        engine = engine or self.default_engine

        rms = buffer.get_rms_energy()
        is_silent = buffer.is_silent()

        # If silent or insufficient audio window (<0.8 sec), return current state without running ASR
        if len(buffer.buffer) < int(buffer.sample_rate * 0.8):
            return {
                "session_id": session_id,
                "status": "buffering",
                "is_silent": is_silent,
                "rms_energy": round(rms, 4),
                "new_text": "",
                "transcript": session.full_transcript,
                "segments": session.segments,
                "word_count": len(session.full_transcript.split())
            }

        if is_silent and len(session.full_transcript) > 0:
            # Consume buffer to prevent stale silence backlog
            buffer.consume(seconds_to_keep=0.2)
            return {
                "session_id": session_id,
                "status": "silent",
                "is_silent": True,
                "rms_energy": round(rms, 4),
                "new_text": "",
                "transcript": session.full_transcript,
                "segments": session.segments,
                "word_count": len(session.full_transcript.split())
            }

        # Transcribe active window
        window = buffer.get_current_window()
        text, conf = self._transcribe_window(window, engine)

        if text:
            session.add_segment(text, confidence=conf)
            # Advance buffer, leaving small 0.4s overlap
            buffer.consume(seconds_to_keep=0.4)

        summary = session.get_summary()
        summary["status"] = "transcribed" if text else "listening"
        summary["new_text"] = text
        summary["rms_energy"] = round(rms, 4)
        summary["is_silent"] = is_silent
        return summary

    def _transcribe_window(self, audio_array: np.ndarray, engine: str) -> tuple[str, float]:
        """Runs fast inference on the audio array."""
        if len(audio_array) == 0:
            return "", 0.0

        if engine == "whisper":
            model = get_fast_whisper()
            if model is not None:
                try:
                    audio_f32 = audio_array.astype(np.float32)
                    res = model.transcribe(
                        audio_f32,
                        fp16=False,
                        language="en"
                    )
                    text = res.get("text", "").strip()
                    hallucinations = ["you", "thank you", "subtitles by", "bye", "..."]
                    if text.lower() in hallucinations:
                        return "", 0.0
                    return text, 0.94
                except Exception as e:
                    print(f"[Project 66] Whisper chunk error: {e}")

        # Fallback to speech_recognition
        try:
            wav_io = io.BytesIO()
            sf.write(wav_io, audio_array, 16000, format="WAV", subtype="PCM_16")
            wav_io.seek(0)
            with sr.AudioFile(wav_io) as source:
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data)
                return text.strip(), 0.91
        except Exception:
            return "", 0.0

    def reset_session(self, session_id: str) -> dict:
        if session_id in self.sessions:
            self.sessions[session_id].reset()
        if session_id in self.buffers:
            self.buffers[session_id].clear()
        return {"session_id": session_id, "status": "reset", "transcript": ""}


# Global singleton instance
streaming_transcriber = StreamingTranscriber()
