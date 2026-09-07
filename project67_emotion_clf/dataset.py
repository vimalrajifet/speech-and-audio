"""
Project 67: Voice Emotion Classifier Dataset & Feature Extraction
Handles RAVDESS emotion audio dataset formatting, acoustic feature extraction,
and sample generation across 8 emotional states:
neutral, calm, happy, sad, angry, fearful, disgust, surprised.
"""

import os
import asyncio
import numpy as np
import librosa
import soundfile as sf

DATASET_DIR = os.path.join(os.path.dirname(__file__), "samples")
os.makedirs(DATASET_DIR, exist_ok=True)

EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

EMOTION_PROMPTS = {
    "neutral": ("The atmospheric temperature is recorded at twenty degrees Celsius.", "en-US-GuyNeural", "+0Hz", "+0%"),
    "calm": ("Take a deep breath and let the soothing evening breeze quiet your restless thoughts.", "en-US-JennyNeural", "-20Hz", "-15%"),
    "happy": ("This is extraordinary news! We won the grand competition with flying colors!", "en-US-AriaNeural", "+40Hz", "+25%"),
    "sad": ("Everything seems so hopeless today, and nothing turned out the way we wished.", "en-US-GuyNeural", "-35Hz", "-25%"),
    "angry": ("Stop that right now! This behavior is completely unacceptable and intolerable!", "en-US-ChristopherNeural", "+50Hz", "+20%"),
    "fearful": ("Did you hear that sound in the shadows? Please, don't leave me alone here!", "en-US-JennyNeural", "+30Hz", "+15%"),
    "disgust": ("That foul odor from the decaying garbage is thoroughly revolting.", "en-US-GuyNeural", "-25Hz", "-10%"),
    "surprised": ("Oh my goodness! I never expected to see you here today of all days!", "en-US-AriaNeural", "+60Hz", "+25%")
}


def extract_acoustic_features(file_path: str, max_duration: float = 3.5) -> np.ndarray:
    """
    Extracts high-dimensional acoustic feature vector (180 features):
    - 40 MFCC means & stds
    - 12 Chroma means & stds
    - 7 Spectral Contrast means & stds
    - 128 Mel spectrogram means
    """
    y, sr = librosa.load(file_path, sr=16000, duration=max_duration)
    if len(y) < 1600:
        y = np.pad(y, (0, 1600 - len(y)))

    # 1. MFCCs (40)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    mfcc_std = np.std(mfcc.T, axis=0)

    # 2. Chroma STFT (12)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)
    chroma_std = np.std(chroma.T, axis=0)

    # 3. Spectral Contrast (7)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast.T, axis=0)
    contrast_std = np.std(contrast.T, axis=0)

    # 4. Mel Spectrogram (40 bands)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    mel_mean = np.mean(mel.T, axis=0)

    # 5. RMS Energy & Zero Crossing Rate
    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))

    features = np.hstack([
        mfcc_mean, mfcc_std,
        chroma_mean, chroma_std,
        contrast_mean, contrast_std,
        mel_mean,
        np.array([rms, zcr])
    ])
    return features


async def _generate_samples():
    """Generates synthetic multi-speaker emotional samples matching RAVDESS acoustics."""
    import edge_tts
    for code, emotion in EMOTIONS.items():
        filename = f"03-01-{code}-01-01-01-01.wav"
        out_path = os.path.join(DATASET_DIR, filename)
        if not os.path.exists(out_path):
            text, voice, pitch, rate = EMOTION_PROMPTS[emotion]
            temp_mp3 = out_path.replace(".wav", ".mp3")
            comm = edge_tts.Communicate(text=text, voice=voice, pitch=pitch, rate=rate)
            await comm.save(temp_mp3)
            audio, sr = sf.read(temp_mp3)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sf.write(out_path, audio.astype(np.float32), 16000)
            if os.path.exists(temp_mp3):
                os.remove(temp_mp3)


def prepare_emotion_dataset():
    """Ensures samples exist and loads all feature vectors and labels."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_generate_samples())
    loop.close()

    X, y, file_paths = [], [], []
    for f in os.listdir(DATASET_DIR):
        if f.endswith(".wav"):
            path = os.path.join(DATASET_DIR, f)
            # Check RAVDESS naming or fallback to filename matching
            parts = f.replace(".wav", "").split("-")
            emotion_label = None
            if len(parts) >= 3 and parts[2] in EMOTIONS:
                emotion_label = EMOTIONS[parts[2]]
            else:
                for emo in EMOTIONS.values():
                    if emo in f.lower():
                        emotion_label = emo
                        break

            if emotion_label:
                feat = extract_acoustic_features(path)
                X.append(feat)
                y.append(emotion_label)
                file_paths.append(path)

    return np.array(X), np.array(y), file_paths


if __name__ == "__main__":
    X, y, files = prepare_emotion_dataset()
    print(f"[Project 67] Loaded {len(X)} samples with {X.shape[1] if len(X) > 0 else 0} features across {len(set(y))} emotion classes.")
