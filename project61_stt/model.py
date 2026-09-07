"""
Project 61: Speech-to-Text Converter
Pipeline supporting both OpenAI Whisper (local offline) and SpeechRecognition (Google STT).
"""

import os
import io
import time
import speech_recognition as sr
import numpy as np
import soundfile as sf
import librosa

# Lazy-loaded Whisper model instance
_whisper_model = None

def get_whisper_model(model_name: str = "base"):
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisper
            print(f"[Project 61] Loading Whisper '{model_name}' model...")
            _whisper_model = whisper.load_model(model_name)
            print("[Project 61] Whisper model loaded successfully.")
        except Exception as e:
            print(f"[Project 61] Warning: Failed to load Whisper ({e}). Will fallback to SpeechRecognition.")
            _whisper_model = None
    return _whisper_model


class SpeechToTextConverter:
    def __init__(self, default_engine: str = "whisper"):
        self.default_engine = default_engine
        self.recognizer = sr.Recognizer()

    def transcribe_file(self, file_path: str, engine: str = None) -> dict:
        """Transcribe an audio file from disk."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        engine = engine or self.default_engine
        start_time = time.time()

        if engine == "whisper":
            return self._transcribe_with_whisper(file_path, start_time)
        else:
            return self._transcribe_with_speech_recognition(file_path, start_time)

    def _transcribe_with_whisper(self, file_path: str, start_time: float) -> dict:
        model = get_whisper_model("base")
        if model is None:
            return self._transcribe_with_speech_recognition(file_path, start_time)

        try:
            # Load audio using soundfile + librosa to avoid ffmpeg dependency
            audio_data, sr_val = sf.read(file_path, dtype='float32')
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
            if sr_val != 16000:
                audio_data = librosa.resample(audio_data, orig_sr=sr_val, target_sr=16000)

            result = model.transcribe(audio_data.astype(np.float32), fp16=False)
            transcription = result.get("text", "").strip()
            language = result.get("language", "en")
            segments = result.get("segments", [])
            elapsed = round(time.time() - start_time, 3)

            formatted_segments = [
                {
                    "start": round(seg.get("start", 0), 2),
                    "end": round(seg.get("end", 0), 2),
                    "text": seg.get("text", "").strip()
                }
                for seg in segments
            ]

            return {
                "engine": "OpenAI Whisper (base)",
                "success": True,
                "text": transcription,
                "language": language,
                "segments": formatted_segments,
                "word_count": len(transcription.split()),
                "processing_time_sec": elapsed
            }
        except Exception as e:
            print(f"[Project 61] Whisper transcription error: {e}. Falling back to Google STT...")
            return self._transcribe_with_speech_recognition(file_path, start_time)

    def _transcribe_with_speech_recognition(self, file_path: str, start_time: float) -> dict:
        try:
            with sr.AudioFile(file_path) as source:
                audio_data = self.recognizer.record(source)
            text = self.recognizer.recognize_google(audio_data)
            elapsed = round(time.time() - start_time, 3)

            return {
                "engine": "Google SpeechRecognition API",
                "success": True,
                "text": text,
                "language": "en-US",
                "segments": [{"start": 0.0, "end": elapsed, "text": text}],
                "word_count": len(text.split()),
                "processing_time_sec": elapsed
            }
        except sr.UnknownValueError:
            return {
                "engine": "Google SpeechRecognition API",
                "success": False,
                "text": "",
                "error": "Speech was unintelligible or silent.",
                "processing_time_sec": round(time.time() - start_time, 3)
            }
        except Exception as e:
            return {
                "engine": "Google SpeechRecognition API",
                "success": False,
                "text": "",
                "error": str(e),
                "processing_time_sec": round(time.time() - start_time, 3)
            }

# Default pipeline instance
converter = SpeechToTextConverter()
