"""
Project 69: End-to-End Speech Summarizer & Intelligence Engine
Transcribes spoken audio with Whisper (numpy float32 array bypass) and extracts
executive summaries, action items, and meeting takeaways.
"""

import os
import sys
import numpy as np
import soundfile as sf
import librosa
import whisper

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from project69_speech_summarizer.summarizer import analyze_speech_content

_WHISPER_MODEL = None


def get_whisper_engine():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        _WHISPER_MODEL = whisper.load_model("base")
    return _WHISPER_MODEL


class SpeechSummarizerEngine:
    def __init__(self):
        self.model = None

    def process_audio(self, audio_path: str) -> dict:
        """
        Transcribes audio file using Whisper (without external ffmpeg dependency)
        and generates executive summary and action items.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load audio with soundfile and librosa to guarantee pure float32 array
        audio_data, sr = sf.read(audio_path, dtype="float32")
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        if sr != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)

        duration_sec = float(len(audio_data) / 16000)

        # Transcribe with Whisper
        engine = get_whisper_engine()
        result = engine.transcribe(audio_data.astype(np.float32), fp16=False, language="en")
        transcript = result.get("text", "").strip()

        # Generate intelligence report
        report = analyze_speech_content(transcript, duration_sec=duration_sec)
        report["audio_file"] = os.path.basename(audio_path)
        report["duration_seconds"] = round(duration_sec, 2)
        report["language"] = result.get("language", "en")

        return report


# Global instance
speech_summarizer_engine = SpeechSummarizerEngine()
