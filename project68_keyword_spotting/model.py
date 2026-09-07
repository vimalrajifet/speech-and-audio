"""
Project 68: Audio Keyword Spotting PyTorch CNN Architecture & Spotter
Detects spoken trigger words and dispatches connected smart automation actions.
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

from project68_keyword_spotting.dataset import (
    KEYWORDS,
    KEYWORD_TO_INDEX,
    INDEX_TO_KEYWORD,
    waveform_to_melspectrogram
)

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "keyword_cnn.pth")

# Real-world smart automation trigger registry
KEYWORD_ACTIONS = {
    "on": {
        "action": "DEVICE_POWER_ON",
        "description": "Activated smart illumination and connected power relays.",
        "icon": "💡",
        "color": "#E5A93C"
    },
    "off": {
        "action": "DEVICE_POWER_OFF",
        "description": "Switched off ambient peripherals and engaged low-power standby.",
        "icon": "🌙",
        "color": "#5F748B"
    },
    "yes": {
        "action": "CONFIRM_TRANSACTION",
        "description": "Authorized prompt confirmation and approved transaction.",
        "icon": "✅",
        "color": "#589E78"
    },
    "no": {
        "action": "CANCEL_OPERATION",
        "description": "Revoked pending action and cleared staging memory.",
        "icon": "❌",
        "color": "#C25450"
    },
    "up": {
        "action": "VOLUME_OR_LIFT_UP",
        "description": "Increased acoustic gain and elevated desk actuator (+15%).",
        "icon": "🔊",
        "color": "#3498DB"
    },
    "down": {
        "action": "VOLUME_OR_LIFT_DOWN",
        "description": "Lowered acoustic gain and lowered desk actuator (-15%).",
        "icon": "🔉",
        "color": "#8E44AD"
    },
    "go": {
        "action": "EXECUTE_PIPELINE",
        "description": "Dispatched autonomous execution thread and began workflow.",
        "icon": "🚀",
        "color": "#2ECC71"
    },
    "stop": {
        "action": "EMERGENCY_HALT",
        "description": "Invoked safety interlock. All active motors and pipelines halted.",
        "icon": "🛑",
        "color": "#E74C3C"
    },
    "left": {
        "action": "PAN_CAMERA_LEFT",
        "description": "Rotated robotic pan-tilt head 45 degrees left.",
        "icon": "⬅️",
        "color": "#D4A373"
    },
    "right": {
        "action": "PAN_CAMERA_RIGHT",
        "description": "Rotated robotic pan-tilt head 45 degrees right.",
        "icon": "➡️",
        "color": "#C68B59"
    }
}


class KeywordCNN(nn.Module):
    """Deep 2D Convolutional Neural Network for Spectrogram Keyword Classification."""
    def __init__(self, num_classes: int = len(KEYWORDS)):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.gap = nn.AdaptiveAvgPool2d((4, 4))

        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.dropout = nn.Dropout(0.35)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (Batch, 1, 64, 32)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.gap(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class KeywordSpotter:
    def __init__(self, confidence_threshold: float = 0.50):
        self.confidence_threshold = confidence_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = KeywordCNN().to(self.device)
        self._load_checkpoint()

    def _load_checkpoint(self):
        if not os.path.exists(CHECKPOINT_PATH):
            from project68_keyword_spotting.train import train_keyword_spotter
            train_keyword_spotter()

        checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def spot(self, audio_input, sr: int = 16000) -> dict:
        """
        Accepts either an audio file path or numpy array.
        Returns detected keyword, confidence, probability distribution,
        and automated trigger action.
        """
        if isinstance(audio_input, str):
            if not os.path.exists(audio_input):
                raise FileNotFoundError(f"Audio file not found: {audio_input}")
            y, file_sr = sf.read(audio_input, dtype="float32")
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            if file_sr != 16000:
                y = librosa.resample(y, orig_sr=file_sr, target_sr=16000)
        elif isinstance(audio_input, np.ndarray):
            y = audio_input
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            if sr != 16000:
                y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        else:
            raise ValueError("Input must be a file path or numpy ndarray.")

        tensor = waveform_to_melspectrogram(y, 16000).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        top_idx = int(np.argmax(probs))
        top_keyword = INDEX_TO_KEYWORD[top_idx]
        top_prob = float(probs[top_idx])

        # Check threshold
        is_spotted = top_prob >= self.confidence_threshold
        action_info = KEYWORD_ACTIONS.get(top_keyword, {})

        # Top 5 ranking
        distribution = [
            {
                "keyword": INDEX_TO_KEYWORD[idx],
                "probability": round(float(probs[idx]), 4),
                "percentage": f"{float(probs[idx]) * 100:.1f}%",
                "icon": KEYWORD_ACTIONS.get(INDEX_TO_KEYWORD[idx], {}).get("icon", "•")
            }
            for idx in np.argsort(-probs)[:5]
        ]

        return {
            "spotted": is_spotted,
            "keyword": top_keyword if is_spotted else "unknown",
            "confidence": round(top_prob, 4),
            "confidence_percentage": f"{top_prob * 100:.1f}%",
            "threshold": self.confidence_threshold,
            "action": action_info.get("action") if is_spotted else "NONE",
            "action_description": action_info.get("description", "Confidence below threshold. No action dispatched.") if is_spotted else "No trigger.",
            "icon": action_info.get("icon", "❓") if is_spotted else "⚪",
            "color": action_info.get("color", "#C68B59") if is_spotted else "#9E8B7A",
            "distribution": distribution
        }


# Global singleton instance
keyword_spotter = KeywordSpotter()
