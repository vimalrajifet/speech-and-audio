"""
Project 63: Language Identification from Audio
Analyzes spoken audio patterns to detect the language, returning ISO 639-1 code,
full name (via pycountry), confidence score, and top-5 probability distribution.
"""

import os
import time
import numpy as np
import soundfile as sf
import librosa
import pycountry

# Import cached Whisper model
from project61_stt.model import get_whisper_model

class LanguageIdentifier:
    def __init__(self, model_name: str = "base"):
        self.model_name = model_name

    def identify(self, file_path: str) -> dict:
        """Analyzes speech patterns in the audio file and predicts the spoken language."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        start_time = time.time()
        model = get_whisper_model(self.model_name)

        # Load audio safely without ffmpeg
        audio_data, sr_val = sf.read(file_path, dtype='float32')
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        if sr_val != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=sr_val, target_sr=16000)

        # Whisper expects 30-second padded or trimmed audio
        import whisper
        audio_padded = whisper.pad_or_trim(audio_data)

        # Compute log-Mel spectrogram
        mel = whisper.log_mel_spectrogram(audio_padded).to(model.device)

        # Detect language probabilities
        _, probs = model.detect_language(mel)

        # Sort languages by probability
        sorted_probs = sorted(probs.items(), key=lambda item: item[1], reverse=True)
        top_language_code, top_confidence = sorted_probs[0]

        def get_language_name(code: str) -> str:
            try:
                lang = pycountry.languages.get(alpha_2=code)
                if lang:
                    return lang.name
                lang3 = pycountry.languages.get(alpha_3=code)
                if lang3:
                    return lang3.name
                return code.upper()
            except Exception:
                return code.upper()

        top_5 = [
            {
                "code": code,
                "name": get_language_name(code),
                "probability": round(float(prob), 4),
                "percentage": f"{round(float(prob) * 100, 2)}%"
            }
            for code, prob in sorted_probs[:5]
        ]

        elapsed = round(time.time() - start_time, 3)

        return {
            "success": True,
            "detected_language_code": top_language_code,
            "detected_language_name": get_language_name(top_language_code),
            "confidence": round(float(top_confidence), 4),
            "confidence_percentage": f"{round(float(top_confidence) * 100, 2)}%",
            "top_predictions": top_5,
            "processing_time_sec": elapsed
        }

# Global instance
language_detector = LanguageIdentifier()
