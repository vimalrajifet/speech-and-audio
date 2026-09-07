"""
Project 68: Audio Keyword Spotting Dataset Loader & Audio Processor
Supports Google Speech Commands v2 10-word vocabulary:
"yes", "no", "up", "down", "go", "stop", "left", "right", "on", "off"
"""

import os
import asyncio
import numpy as np
import soundfile as sf
import librosa
import torch

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "samples")
os.makedirs(SAMPLE_DIR, exist_ok=True)

KEYWORDS = ["yes", "no", "up", "down", "go", "stop", "left", "right", "on", "off"]
KEYWORD_TO_INDEX = {w: i for i, w in enumerate(KEYWORDS)}
INDEX_TO_KEYWORD = {i: w for i, w in enumerate(KEYWORDS)}

VOICE_CONFIGS = [
    ("en-US-GuyNeural", "+0Hz", "+0%"),
    ("en-US-JennyNeural", "+20Hz", "+5%"),
    ("en-US-AriaNeural", "+40Hz", "+10%"),
    ("en-US-ChristopherNeural", "-20Hz", "-5%")
]


def waveform_to_melspectrogram(y: np.ndarray, sr: int = 16000, target_frames: int = 32) -> torch.Tensor:
    """
    Converts 16kHz audio waveform into (1, 64, 32) Mel-spectrogram tensor.
    """
    # Fix duration to 1.0 second (16000 samples)
    target_samples = 16000
    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)))
    else:
        y = y[:target_samples]

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=64, n_fft=1024, hop_length=512
    )
    # Convert to log scale (dB)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Pad or slice to target_frames (default 32)
    if mel_db.shape[1] < target_frames:
        mel_db = np.pad(mel_db, ((0, 0), (0, target_frames - mel_db.shape[1])), mode="constant")
    else:
        mel_db = mel_db[:, :target_frames]

    # Normalize to zero mean, unit variance
    mean = np.mean(mel_db)
    std = np.std(mel_db) + 1e-6
    mel_norm = (mel_db - mean) / std

    return torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 64, 32)


async def _generate_samples():
    """Generates audio samples for each of the 10 keywords across diverse voices."""
    import edge_tts
    for word in KEYWORDS:
        for v_idx, (voice, pitch, rate) in enumerate(VOICE_CONFIGS):
            filename = f"kw_{word}_{v_idx+1}.wav"
            out_path = os.path.join(SAMPLE_DIR, filename)
            if not os.path.exists(out_path):
                temp_mp3 = out_path.replace(".wav", ".mp3")
                comm = edge_tts.Communicate(text=f"{word}.", voice=voice, pitch=pitch, rate=rate)
                await comm.save(temp_mp3)
                audio, sr = sf.read(temp_mp3)
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=1)
                if sr != 16000:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                # Pad to 1 sec
                if len(audio) < 16000:
                    audio = np.pad(audio, (0, 16000 - len(audio)))
                else:
                    audio = audio[:16000]
                sf.write(out_path, audio.astype(np.float32), 16000)
                if os.path.exists(temp_mp3):
                    os.remove(temp_mp3)


def prepare_keyword_dataset():
    """Loads all keyword audio files and precomputes Mel-spectrogram tensors."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_generate_samples())
    loop.close()

    tensors, labels, paths = [], [], []
    for f in os.listdir(SAMPLE_DIR):
        if f.endswith(".wav") and f.startswith("kw_"):
            parts = f.replace(".wav", "").split("_")
            if len(parts) >= 2:
                word = parts[1]
                if word in KEYWORD_TO_INDEX:
                    file_path = os.path.join(SAMPLE_DIR, f)
                    y, sr = sf.read(file_path, dtype="float32")
                    mel_tensor = waveform_to_melspectrogram(y, sr)
                    tensors.append(mel_tensor)
                    labels.append(KEYWORD_TO_INDEX[word])
                    paths.append(file_path)

    if tensors:
        X = torch.stack(tensors)  # (N, 1, 64, 32)
        y = torch.tensor(labels, dtype=torch.long)
    else:
        X = torch.empty((0, 1, 64, 32))
        y = torch.empty((0,), dtype=torch.long)

    return X, y, paths


if __name__ == "__main__":
    X, y, paths = prepare_keyword_dataset()
    print(f"[Project 68] Dataset loaded: {X.shape[0]} samples with shape {X.shape[1:]} across {len(set(y.numpy()))} keywords.")
