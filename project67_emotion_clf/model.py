"""
Project 67: Voice Emotion Classifier Inference Pipeline
Performs acoustic feature extraction, model prediction, emotion probability breakdown,
and acoustic voice analysis (pitch, energy, valence/arousal).
"""

import os
import sys
import joblib
import numpy as np
import librosa

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from project67_emotion_clf.dataset import extract_acoustic_features

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "emotion_classifier.joblib")

# Emotion valence and arousal mapping (Russell's Circumplex Model)
EMOTION_COORDINATES = {
    "happy": {"valence": 0.85, "arousal": 0.80, "color": "#E5A93C", "emoji": "😄"},
    "calm": {"valence": 0.70, "arousal": -0.60, "color": "#589E78", "emoji": "😌"},
    "neutral": {"valence": 0.05, "arousal": 0.00, "color": "#9E8B7A", "emoji": "😐"},
    "surprised": {"valence": 0.40, "arousal": 0.90, "color": "#D4A373", "emoji": "😲"},
    "sad": {"valence": -0.75, "arousal": -0.65, "color": "#5F748B", "emoji": "😢"},
    "angry": {"valence": -0.80, "arousal": 0.85, "color": "#C25450", "emoji": "😡"},
    "fearful": {"valence": -0.65, "arousal": 0.75, "color": "#9B59B6", "emoji": "😨"},
    "disgust": {"valence": -0.70, "arousal": 0.20, "color": "#768E56", "emoji": "🤢"}
}


class VoiceEmotionClassifier:
    def __init__(self):
        self.payload = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(CHECKPOINT_PATH):
            from project67_emotion_clf.train import train_emotion_model
            self.payload = train_emotion_model()
        else:
            self.payload = joblib.load(CHECKPOINT_PATH)

    def analyze_audio_properties(self, y: np.ndarray, sr: int = 16000) -> dict:
        """Computes fundamental frequency (pitch), energy, tempo, and spectral features."""
        # RMS Energy
        rms = float(np.mean(librosa.feature.rms(y=y)))

        # Spectral Centroid (brightness)
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

        # Pitch estimation using piptrack
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        valid_pitches = pitches[magnitudes > np.median(magnitudes)]
        mean_pitch = float(np.mean(valid_pitches)) if len(valid_pitches) > 0 else 180.0
        # Clip pitch to human speech range (70Hz - 450Hz)
        mean_pitch = max(70.0, min(450.0, mean_pitch))

        # Tempo (speaking speed)
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
            tempo_val = float(tempo[0] if isinstance(tempo, (list, np.ndarray)) else tempo)
        except Exception:
            tempo_val = 110.0

        return {
            "mean_pitch_hz": round(mean_pitch, 1),
            "rms_energy": round(rms, 4),
            "spectral_centroid_hz": round(centroid, 1),
            "tempo_bpm": round(tempo_val, 1)
        }

    def predict(self, audio_path: str) -> dict:
        """
        Runs emotion classification on input audio file.
        Returns predicted emotion, 8-class probabilities, valence/arousal,
        and acoustic analysis.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        y, sr = librosa.load(audio_path, sr=16000)
        duration = float(len(y) / sr)

        # Extract features and scale
        feat = extract_acoustic_features(audio_path).reshape(1, -1)
        feat_scaled = self.payload["scaler"].transform(feat)

        model = self.payload["model"]
        probs = model.predict_proba(feat_scaled)[0]
        classes = self.payload["classes"]

        # Form probability map
        prob_dict = {cls_name: round(float(p), 4) for cls_name, p in zip(classes, probs)}
        top_emotion = classes[int(np.argmax(probs))]
        top_confidence = round(float(np.max(probs)), 4)

        # Acoustic characteristics
        acoustic = self.analyze_audio_properties(y, sr)
        coord = EMOTION_COORDINATES.get(top_emotion, {"valence": 0.0, "arousal": 0.0, "color": "#C68B59", "emoji": "🎙️"})

        # Format top 4 distribution for radar/bar charts
        sorted_distribution = [
            {
                "emotion": k,
                "probability": v,
                "percentage": f"{v * 100:.1f}%",
                "emoji": EMOTION_COORDINATES.get(k, {}).get("emoji", "•"),
                "color": EMOTION_COORDINATES.get(k, {}).get("color", "#C68B59")
            }
            for k, v in sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        ]

        return {
            "predicted_emotion": top_emotion,
            "confidence": top_confidence,
            "confidence_percentage": f"{top_confidence * 100:.1f}%",
            "emoji": coord.get("emoji", "🎭"),
            "color": coord.get("color", "#C68B59"),
            "valence": coord.get("valence", 0.0),
            "arousal": coord.get("arousal", 0.0),
            "distribution": sorted_distribution,
            "probabilities": prob_dict,
            "duration_sec": round(duration, 2),
            "acoustic_profile": acoustic
        }


# Global singleton instance
voice_emotion_classifier = VoiceEmotionClassifier()
