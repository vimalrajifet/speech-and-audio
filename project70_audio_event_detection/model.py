"""
Project 70: Audio Event Detection & Acoustic Scene Classification
Detects environmental and safety acoustic events with time-segmented localization.
"""

import os
import sys
import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from project70_audio_event_detection.dataset import (
    EVENT_CLASSES,
    EVENT_TO_IDX,
    IDX_TO_EVENT,
    extract_event_spectrogram
)

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "audio_event_cnn.pth")

EVENT_METADATA = {
    "siren": {
        "category": "EMERGENCY",
        "icon": "🚨",
        "color": "#E74C3C",
        "severity": "CRITICAL",
        "description": "Emergency service acoustic siren detected."
    },
    "glass_break": {
        "category": "EMERGENCY",
        "icon": "💥",
        "color": "#C0392B",
        "severity": "HIGH",
        "description": "High-frequency shatter impact / perimeter breach."
    },
    "gunshot": {
        "category": "EMERGENCY",
        "icon": "⚡",
        "color": "#962D22",
        "severity": "CRITICAL",
        "description": "Ballistic impulse shockwave signature."
    },
    "baby_crying": {
        "category": "ALERT",
        "icon": "👶",
        "color": "#F39C12",
        "severity": "MEDIUM",
        "description": "Infant vocal distress acoustic signature."
    },
    "car_horn": {
        "category": "ALERT",
        "icon": "🚗",
        "color": "#E67E22",
        "severity": "MEDIUM",
        "description": "Automotive warning horn honking."
    },
    "dog_bark": {
        "category": "ALERT",
        "icon": "🐕",
        "color": "#D35400",
        "severity": "LOW",
        "description": "Canine barking formant bursts."
    },
    "rain": {
        "category": "AMBIENT",
        "icon": "🌧️",
        "color": "#3498DB",
        "severity": "INFO",
        "description": "Weather precipitation acoustic texture."
    },
    "footsteps": {
        "category": "AMBIENT",
        "icon": "👣",
        "color": "#7F8C8D",
        "severity": "INFO",
        "description": "Periodic low-frequency gait impacts."
    },
    "clapping": {
        "category": "AMBIENT",
        "icon": "👏",
        "color": "#2ECC71",
        "severity": "INFO",
        "description": "Transient Poisson crowd applause."
    },
    "keyboard_typing": {
        "category": "AMBIENT",
        "icon": "⌨️",
        "color": "#9B59B6",
        "severity": "INFO",
        "description": "Mechanical keystroke actuation."
    }
}


class AudioEventCNN(nn.Module):
    """Multi-Scale 2D Convolutional Neural Network for Audio Event Spectrograms."""
    def __init__(self, num_classes: int = len(EVENT_CLASSES)):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.gap = nn.AdaptiveAvgPool2d((4, 4))

        self.fc1 = nn.Linear(256 * 4 * 4, 128)
        self.dropout = nn.Dropout(0.35)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (Batch, 1, 64, 64)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.gap(F.relu(self.bn4(self.conv4(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class AudioEventDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AudioEventCNN().to(self.device)
        self._load_checkpoint()

    def _load_checkpoint(self):
        if not os.path.exists(CHECKPOINT_PATH):
            from project70_audio_event_detection.train import train_audio_event_model
            train_audio_event_model()

        checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def detect(self, audio_path: str) -> dict:
        """
        Runs whole-file and time-windowed event detection on an audio clip.
        Returns top event, confidence, security severity, distribution,
        and temporal timeline segments.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        y, sr = sf.read(audio_path, dtype="float32")
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)

        duration = float(len(y) / 16000)

        # 1. Global classification
        full_spec = extract_event_spectrogram(y, 16000, n_frames=64).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(full_spec)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        top_idx = int(np.argmax(probs))
        top_event = IDX_TO_EVENT[top_idx]
        top_conf = float(probs[top_idx])
        meta = EVENT_METADATA.get(top_event, {})

        # Top-5 distribution
        distribution = [
            {
                "event": IDX_TO_EVENT[idx],
                "probability": round(float(probs[idx]), 4),
                "percentage": f"{float(probs[idx]) * 100:.1f}%",
                "icon": EVENT_METADATA.get(IDX_TO_EVENT[idx], {}).get("icon", "•"),
                "color": EVENT_METADATA.get(IDX_TO_EVENT[idx], {}).get("color", "#C68B59")
            }
            for idx in np.argsort(-probs)[:5]
        ]

        # 2. Time-localized window segmentation (1.0s window, 0.5s hop)
        window_samples = 16000
        hop_samples = 8000
        segments = []

        for start_idx in range(0, max(1, len(y) - window_samples // 2), hop_samples):
            end_idx = min(len(y), start_idx + window_samples)
            chunk = y[start_idx:end_idx]
            if len(chunk) < 8000:
                continue

            chunk_spec = extract_event_spectrogram(chunk, 16000, n_frames=64).unsqueeze(0).to(self.device)
            with torch.no_grad():
                c_logits = self.model(chunk_spec)
                c_probs = F.softmax(c_logits, dim=1).cpu().numpy()[0]

            c_top_idx = int(np.argmax(c_probs))
            c_event = IDX_TO_EVENT[c_top_idx]
            c_conf = float(c_probs[c_top_idx])

            start_t = round(start_idx / 16000, 2)
            end_t = round(end_idx / 16000, 2)

            segments.append({
                "start_time_sec": start_t,
                "end_time_sec": end_t,
                "event": c_event,
                "confidence": round(c_conf, 3),
                "icon": EVENT_METADATA.get(c_event, {}).get("icon", "•"),
                "severity": EVENT_METADATA.get(c_event, {}).get("severity", "INFO")
            })

        return {
            "top_event": top_event,
            "confidence": round(top_conf, 4),
            "confidence_percentage": f"{top_conf * 100:.1f}%",
            "category": meta.get("category", "UNKNOWN"),
            "severity": meta.get("severity", "INFO"),
            "icon": meta.get("icon", "🔊"),
            "color": meta.get("color", "#C68B59"),
            "description": meta.get("description", ""),
            "duration_sec": round(duration, 2),
            "distribution": distribution,
            "timeline_segments": segments
        }


# Global singleton instance
audio_event_detector = AudioEventDetector()
