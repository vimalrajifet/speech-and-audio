"""
Project 64: UrbanSound8K Sound Classifier
PyTorch CNN architecture trained on 40-dimensional MFCCs for 10 environmental sound classes.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import soundfile as sf
import librosa

CLASSES = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
]

CLASS_LABELS = {i: c for i, c in enumerate(CLASSES)}

class SoundCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(SoundCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),

            nn.AdaptiveAvgPool2d((5, 5))
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, 1, n_mfcc, time)
        feat = self.conv_block(x)
        feat = feat.view(feat.size(0), -1)
        out = self.fc(feat)
        return out


def extract_mfcc(file_path_or_audio, sr: int = 16000, n_mfcc: int = 40, max_pad_len: int = 174) -> np.ndarray:
    """Extracts MFCC features and pads or trims to fixed length."""
    if isinstance(file_path_or_audio, (str, os.PathLike)):
        audio, sr_loaded = sf.read(file_path_or_audio, dtype='float32')
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr_loaded != sr:
            audio = librosa.resample(audio, orig_sr=sr_loaded, target_sr=sr)
    else:
        audio = file_path_or_audio

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    pad_width = max_pad_len - mfcc.shape[1]
    if pad_width > 0:
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    return mfcc


class SoundClassifier:
    def __init__(self, checkpoint_path: str = None):
        if checkpoint_path is None:
            checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints", "sound_cnn.pth")
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SoundCNN(num_classes=len(CLASSES)).to(self.device)
        self._load_weights()

    def _load_weights(self):
        if os.path.exists(self.checkpoint_path):
            try:
                state_dict = torch.load(self.checkpoint_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                print(f"[Project 64] Loaded trained weights from {self.checkpoint_path}")
            except Exception as e:
                print(f"[Project 64] Warning loading checkpoint: {e}")
        else:
            print(f"[Project 64] Checkpoint not found at {self.checkpoint_path}. Run train.py to train.")

    def predict(self, file_path_or_audio) -> dict:
        self.model.eval()
        mfcc = extract_mfcc(file_path_or_audio)
        tensor = torch.tensor(mfcc).unsqueeze(0).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()

        predicted_idx = int(np.argmax(probs))
        predicted_class = CLASSES[predicted_idx]
        confidence = float(probs[predicted_idx])

        class_probabilities = [
            {
                "class_name": cls,
                "probability": round(float(prob), 4),
                "percentage": f"{round(float(prob) * 100, 2)}%"
            }
            for cls, prob in zip(CLASSES, probs)
        ]
        # Sort descending
        class_probabilities = sorted(class_probabilities, key=lambda x: x["probability"], reverse=True)

        return {
            "success": True,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "confidence_percentage": f"{round(confidence * 100, 2)}%",
            "probabilities": class_probabilities,
            "mfcc_shape": list(mfcc.shape)
        }

# Global classifier instance
sound_classifier = SoundClassifier()
