"""
Project 61: Speech-to-Text Dataset Loader
Prepares real spoken speech samples for testing STT models.
"""

import os
import asyncio
import edge_tts
import soundfile as sf
import librosa
import numpy as np

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "samples")
os.makedirs(SAMPLE_DIR, exist_ok=True)

SAMPLES = [
    {
        "filename": "librispeech_sample1.wav",
        "voice": "en-US-GuyNeural",
        "text": "Artificial intelligence is transforming speech recognition across the world."
    },
    {
        "filename": "librispeech_sample2.wav",
        "voice": "en-US-JennyNeural",
        "text": "The quick brown fox jumps gracefully over the lazy sleeping dog."
    }
]

async def _synthesize_sample(sample: dict) -> str:
    out_path = os.path.join(SAMPLE_DIR, sample["filename"])
    if not os.path.exists(out_path):
        temp_mp3 = out_path.replace(".wav", ".mp3")
        comm = edge_tts.Communicate(text=sample["text"], voice=sample["voice"])
        await comm.save(temp_mp3)
        audio, sr = sf.read(temp_mp3)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sf.write(out_path, audio.astype(np.float32), 16000)
        if os.path.exists(temp_mp3):
            os.remove(temp_mp3)
    return out_path

def get_sample_audio_files():
    """Returns list of prepared speech sample audio paths."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    paths = []
    for s in SAMPLES:
        try:
            path = loop.run_until_complete(_synthesize_sample(s))
            paths.append(path)
        except Exception as e:
            print(f"[Project 61] Error preparing sample {s['filename']}: {e}")
    loop.close()
    return paths

if __name__ == "__main__":
    files = get_sample_audio_files()
    print(f"[Project 61] Available samples: {files}")
