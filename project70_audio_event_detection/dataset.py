"""
Project 70: Audio Event Detection Dataset & Acoustic Feature Processor
ESC-50 inspired environmental sound events with temporal frequency synthesis:
siren, glass_break, dog_bark, car_horn, rain, footsteps, baby_crying, clapping, keyboard_typing, gunshot
"""

import os
import numpy as np
import soundfile as sf
import librosa
import torch

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "samples")
os.makedirs(SAMPLE_DIR, exist_ok=True)

EVENT_CLASSES = [
    "siren",
    "glass_break",
    "dog_bark",
    "car_horn",
    "rain",
    "footsteps",
    "baby_crying",
    "clapping",
    "keyboard_typing",
    "gunshot"
]

EVENT_TO_IDX = {e: i for i, e in enumerate(EVENT_CLASSES)}
IDX_TO_EVENT = {i: e for i, e in enumerate(EVENT_CLASSES)}


def synthesize_acoustic_event(event_name: str, duration: float = 3.0, sr: int = 16000) -> np.ndarray:
    """
    Synthesizes physically grounded acoustic signatures for environmental sound events.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    rng = np.random.RandomState(abs(hash(event_name)) % (2**31))

    if event_name == "siren":
        # Frequency modulated sinusoidal sweep (600Hz to 1200Hz back and forth)
        freq_mod = 900 + 350 * np.sin(2 * np.pi * 1.2 * t)
        phase = 2 * np.pi * np.cumsum(freq_mod) / sr
        audio = 0.5 * np.sin(phase) + 0.15 * np.sin(2 * phase)

    elif event_name == "glass_break":
        # Sharp high-frequency shattering noise burst + decaying resonant harmonics
        audio = rng.normal(0, 0.4, len(t))
        envelope = np.exp(-t * 8.0)
        # High-pass filter resonant shimmer
        audio = audio * envelope * (1.0 + 0.5 * np.sin(2 * np.pi * 3200 * t) + 0.3 * np.sin(2 * np.pi * 4800 * t))

    elif event_name == "dog_bark":
        # Rhythmic explosive formant bursts (bark-bark cadence)
        audio = np.zeros_like(t)
        bark_times = [0.2, 0.8, 1.6, 2.2]
        for bt in bark_times:
            idx_start = int(bt * sr)
            idx_end = min(len(t), idx_start + int(0.35 * sr))
            seg_len = idx_end - idx_start
            if seg_len > 0:
                tb = np.linspace(0, 0.35, seg_len)
                bark_wave = (
                    0.6 * np.sin(2 * np.pi * 280 * tb) +
                    0.4 * np.sin(2 * np.pi * 560 * tb) +
                    0.2 * rng.normal(0, 0.3, seg_len)
                ) * np.sin(np.pi * tb / 0.35)
                audio[idx_start:idx_end] += bark_wave

    elif event_name == "car_horn":
        # Dual-tone consonant chord (415Hz + 500Hz) sustained with attack/decay
        chord = 0.5 * np.sin(2 * np.pi * 415 * t) + 0.5 * np.sin(2 * np.pi * 500 * t)
        pulse = (np.sin(2 * np.pi * 0.8 * t) > 0.0).astype(float)
        audio = chord * pulse * 0.7

    elif event_name == "rain":
        # Continuous stochastic pink noise texture
        noise = rng.normal(0, 0.3, len(t))
        # Moving average filter for pinkish spectrum
        audio = np.convolve(noise, np.ones(5)/5.0, mode='same') * 0.6

    elif event_name == "footsteps":
        # Low-frequency periodic thumps (1.8 Hz cadence)
        audio = np.zeros_like(t)
        step_times = np.arange(0.2, duration, 0.6)
        for st in step_times:
            idx = int(st * sr)
            end = min(len(t), idx + int(0.12 * sr))
            slen = end - idx
            if slen > 0:
                ts = np.linspace(0, 0.12, slen)
                thump = 0.7 * np.sin(2 * np.pi * 95 * ts) * np.exp(-ts * 30.0)
                audio[idx:end] += thump

    elif event_name == "baby_crying":
        # Repetitive distressed vocal wail (450Hz to 600Hz pitch glide)
        wail_freq = 520 + 110 * np.sin(2 * np.pi * 2.0 * t)
        phase = 2 * np.pi * np.cumsum(wail_freq) / sr
        audio = (0.5 * np.sin(phase) + 0.25 * np.sin(2 * phase)) * (0.6 + 0.4 * np.sin(2 * np.pi * 1.0 * t))

    elif event_name == "clapping":
        # Dense random transient impulses
        audio = np.zeros_like(t)
        impulses = rng.poisson(lam=14, size=len(t)) > 10
        audio[impulses] = rng.uniform(0.3, 0.8, np.sum(impulses))
        # Resonant acoustic smoothing
        audio = np.convolve(audio, np.exp(-np.linspace(0, 0.05, int(sr * 0.02)) * 80), mode='same')

    elif event_name == "keyboard_typing":
        # Rapid short clacks
        audio = np.zeros_like(t)
        clicks = rng.choice(len(t), size=24, replace=False)
        for clk in clicks:
            end = min(len(t), clk + int(0.015 * sr))
            audio[clk:end] += rng.normal(0, 0.5, end - clk)

    elif event_name == "gunshot":
        # Extremely loud initial transient shockwave + rapid exponential decay
        audio = rng.normal(0, 0.9, len(t)) * np.exp(-t * 12.0)
        audio[:int(0.02 * sr)] += 0.9 * np.sin(2 * np.pi * 180 * t[:int(0.02 * sr)])

    else:
        audio = rng.normal(0, 0.1, len(t))

    # Normalize audio
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.85
    return audio.astype(np.float32)


def extract_event_spectrogram(y: np.ndarray, sr: int = 16000, n_frames: int = 64) -> torch.Tensor:
    """Transforms audio into (1, 64, 64) normalized Log-Mel Spectrogram."""
    if len(y) < 16000:
        y = np.pad(y, (0, 16000 - len(y)))
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, n_fft=1024, hop_length=512)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    if mel_db.shape[1] < n_frames:
        mel_db = np.pad(mel_db, ((0, 0), (0, n_frames - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :n_frames]

    mean, std = np.mean(mel_db), np.std(mel_db) + 1e-6
    norm = (mel_db - mean) / std
    return torch.tensor(norm, dtype=torch.float32).unsqueeze(0)


def prepare_audio_event_dataset():
    """Generates sample audio clips for all 10 event classes and extracts spectrograms."""
    tensors, labels, paths = [], [], []

    for event in EVENT_CLASSES:
        for var in range(1, 4):  # 3 variations per event class
            filename = f"event_{event}_var{var}.wav"
            out_path = os.path.join(SAMPLE_DIR, filename)
            if not os.path.exists(out_path):
                # Add slight random duration or pitch variation
                dur = 3.0 + var * 0.2
                audio = synthesize_acoustic_event(event, duration=dur)
                sf.write(out_path, audio, 16000)

            audio, sr = sf.read(out_path, dtype="float32")
            spec = extract_event_spectrogram(audio, sr)
            tensors.append(spec)
            labels.append(EVENT_TO_IDX[event])
            paths.append(out_path)

    X = torch.stack(tensors)  # Shape: (N, 1, 64, 64)
    y = torch.tensor(labels, dtype=torch.long)
    return X, y, paths


if __name__ == "__main__":
    X, y, paths = prepare_audio_event_dataset()
    print(f"[Project 70] Prepared {len(X)} audio event samples across {len(EVENT_CLASSES)} classes.")
