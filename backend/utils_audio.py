"""
Audio Processing Utilities
Handles audio loading, format conversion, normalization, resampling to 16kHz without ffmpeg,
and generating visual Mel-Spectrogram and Waveform images.
"""

import os
import io
import math
import numpy as np
import soundfile as sf
import librosa
import matplotlib
matplotlib.use('Agg') # Headless mode for web server
import matplotlib.pyplot as plt

def load_audio_safe(file_or_path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """
    Safely loads audio using soundfile and librosa without requiring ffmpeg.
    Returns mono float32 numpy array and sample rate.
    """
    if isinstance(file_or_path, (str, os.PathLike)):
        audio, sr = sf.read(file_or_path, dtype='float32')
    elif hasattr(file_or_path, 'read'):
        audio, sr = sf.read(io.BytesIO(file_or_path.read()), dtype='float32')
    elif isinstance(file_or_path, bytes):
        audio, sr = sf.read(io.BytesIO(file_or_path), dtype='float32')
    else:
        raise ValueError(f"Unsupported audio input type: {type(file_or_path)}")

    # Convert to mono if multi-channel
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Resample if needed
    if target_sr and sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return audio.astype(np.float32), sr

def generate_spectrogram_image(audio: np.ndarray, sr: int = 16000, title: str = "Mel-Spectrogram") -> bytes:
    """
    Generates a high-resolution Mel-Spectrogram image in Sandalwood & Amber color tones.
    Returns raw PNG bytes.
    """
    fig, ax = plt.subplots(figsize=(8, 3), dpi=100)
    fig.patch.set_facecolor('#FDFBF7')
    ax.set_facecolor('#FDFBF7')

    # Compute Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Plot using a warm amber/copper colormap ('copper', 'YlOrBr', or 'magma')
    img = librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax, cmap='YlOrBr')

    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title(title, color='#3E2D22', fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Time (seconds)', color='#6B5E52')
    ax.set_ylabel('Hz', color='#6B5E52')
    ax.tick_params(colors='#6B5E52')
    for spine in ax.spines.values():
        spine.set_color('#C68B59')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def generate_waveform_image(audio: np.ndarray, sr: int = 16000, title: str = "Audio Waveform") -> bytes:
    """Generates an audio amplitude waveform image styled in Sandalwood tones."""
    fig, ax = plt.subplots(figsize=(8, 2), dpi=100)
    fig.patch.set_facecolor('#FDFBF7')
    ax.set_facecolor('#FDFBF7')

    time_axis = np.linspace(0, len(audio) / sr, num=len(audio))
    ax.plot(time_axis, audio, color='#C68B59', linewidth=0.8, alpha=0.9)
    ax.fill_between(time_axis, audio, color='#EFE8D8', alpha=0.7)

    ax.set_title(title, color='#3E2D22', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time (s)', color='#6B5E52', fontsize=9)
    ax.set_ylabel('Amplitude', color='#6B5E52', fontsize=9)
    ax.tick_params(colors='#6B5E52')
    for spine in ax.spines.values():
        spine.set_color('#C68B59')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
