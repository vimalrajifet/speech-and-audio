"""
Project 64: UrbanSound8K Dataset Preparation & Feature Extraction
Provides dataset creation, sample generation for 10 classes, and DataLoader setup.
"""

import os
import math
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader

from project64_sound_clf.model import CLASSES, extract_mfcc

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "samples")
os.makedirs(SAMPLE_DIR, exist_ok=True)

def generate_environmental_sound(class_name: str, duration: float = 3.0, sr: int = 16000) -> np.ndarray:
    """Generates synthetic acoustic signals simulating specific environmental sounds."""
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    np.random.seed(hash(class_name) % (2**32))

    if class_name == "siren":
        # Frequency modulated sinusoidal wave (600Hz to 1200Hz, 0.5Hz modulation)
        freq = 900 + 300 * np.sin(2 * np.pi * 0.8 * t)
        phase = 2 * np.pi * np.cumsum(freq) / sr
        audio = 0.7 * np.sin(phase)

    elif class_name == "dog_bark":
        # Rhythmic bursts of harmonic pitched noise
        audio = np.zeros(n_samples)
        bark_starts = [0.2, 0.8, 1.5, 2.2]
        for start in bark_starts:
            idx = int(start * sr)
            b_len = int(0.25 * sr)
            if idx + b_len < n_samples:
                tb = np.linspace(0, 0.25, b_len)
                env = np.sin(np.pi * tb / 0.25) ** 2
                bark = env * (0.6 * np.sin(2 * np.pi * 380 * tb) + 0.4 * np.random.randn(b_len))
                audio[idx:idx + b_len] += bark

    elif class_name == "engine_idling":
        # Low frequency rumble (35-70 Hz) with harmonics and broadband noise
        audio = 0.5 * np.sin(2 * np.pi * 45 * t) + 0.3 * np.sin(2 * np.pi * 90 * t) + 0.2 * np.random.randn(n_samples) * 0.4

    elif class_name == "drilling":
        # High frequency buzz (400-800Hz) with rapid amplitude modulation
        carrier = np.sin(2 * np.pi * 650 * t)
        mod = 0.5 * (1 + np.sin(2 * np.pi * 50 * t))
        audio = 0.6 * carrier * mod + 0.3 * np.random.randn(n_samples)

    elif class_name == "gun_shot":
        # Sharp impulse blast with exponential decay
        audio = np.zeros(n_samples)
        blast_len = int(0.4 * sr)
        tb = np.linspace(0, 0.4, blast_len)
        decay = np.exp(-12 * tb)
        audio[:blast_len] = decay * np.random.randn(blast_len) * 0.9

    elif class_name == "car_horn":
        # Dual-tone horn (400Hz and 500Hz)
        audio = 0.5 * np.sin(2 * np.pi * 415 * t) + 0.5 * np.sin(2 * np.pi * 520 * t)

    elif class_name == "air_conditioner":
        # Steady hum at 120Hz + filtered pink noise
        audio = 0.4 * np.sin(2 * np.pi * 120 * t) + 0.4 * np.random.randn(n_samples) * 0.5

    elif class_name == "children_playing":
        # Dynamic fluctuating pitch components in speech range (300-1500Hz)
        audio = 0.4 * np.sin(2 * np.pi * (500 + 200 * np.sin(2 * np.pi * 2 * t)) * t) + 0.3 * np.random.randn(n_samples) * 0.6

    elif class_name == "jackhammer":
        # Periodic heavy impacts (15 Hz repetition rate)
        audio = np.zeros(n_samples)
        rep_period = int(sr / 15)
        for i in range(0, n_samples - rep_period, rep_period):
            chunk_len = min(int(0.04 * sr), n_samples - i)
            audio[i:i + chunk_len] += np.exp(-50 * np.linspace(0, 0.04, chunk_len)) * np.random.randn(chunk_len)

    elif class_name == "street_music":
        # Melodic chords (C major: 261Hz, 329Hz, 392Hz)
        audio = 0.3 * np.sin(2 * np.pi * 261.63 * t) + 0.3 * np.sin(2 * np.pi * 329.63 * t) + 0.3 * np.sin(2 * np.pi * 392.00 * t)

    else:
        audio = 0.5 * np.random.randn(n_samples)

    # Normalize audio to [-0.95, 0.95]
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = 0.95 * (audio / max_val)

    return audio.astype(np.float32)

def prepare_sample_files():
    """Generates and saves sample WAV files for all 10 classes in the samples directory."""
    files = {}
    for cls in CLASSES:
        path = os.path.join(SAMPLE_DIR, f"{cls}_sample.wav")
        if not os.path.exists(path):
            audio = generate_environmental_sound(cls)
            sf.write(path, audio, 16000)
        files[cls] = path
    return files

class UrbanSoundSyntheticDataset(Dataset):
    def __init__(self, samples_per_class: int = 40):
        self.features = []
        self.labels = []
        print(f"[Project 64] Preparing dataset with {samples_per_class} samples per class across {len(CLASSES)} classes...")
        for label_idx, cls in enumerate(CLASSES):
            for i in range(samples_per_class):
                # Add slight variation
                dur = 2.5 + 0.02 * (i % 20)
                audio = generate_environmental_sound(cls, duration=dur)
                # Add slight noise augmentation
                audio = audio + 0.05 * np.random.randn(len(audio)).astype(np.float32)
                mfcc = extract_mfcc(audio)
                self.features.append(mfcc)
                self.labels.append(label_idx)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # returns (1, 40, 174) tensor and label
        feat = torch.tensor(self.features[idx]).unsqueeze(0).float()
        label = torch.tensor(self.labels[idx]).long()
        return feat, label
