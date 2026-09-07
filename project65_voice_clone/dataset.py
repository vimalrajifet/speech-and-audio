"""
Project 65: Voice Cloning Dataset Loader & Reference Speaker Samples
Prepares distinct male and female speaker reference samples from VCTK / LibriTTS.
"""

import os
import asyncio
import edge_tts
import soundfile as sf
import librosa

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "samples")
os.makedirs(SAMPLE_DIR, exist_ok=True)

SPEAKERS = [
    {
        "id": "speaker_male_richard",
        "name": "Richard (US Male, Warm Baritone)",
        "voice": "en-US-GuyNeural",
        "text": "The quick brown fox jumps over the lazy dog. Voice cloning technology is reaching unprecedented levels of accuracy.",
        "filename": "reference_male_richard.wav"
    },
    {
        "id": "speaker_female_elena",
        "name": "Elena (US Female, Clear Mezzo)",
        "voice": "en-US-JennyNeural",
        "text": "Welcome to our laboratory! We are demonstrating zero-shot voice cloning using deep speaker embeddings.",
        "filename": "reference_female_elena.wav"
    }
]

async def _create_speaker_sample(speaker: dict) -> str:
    path = os.path.join(SAMPLE_DIR, speaker["filename"])
    if not os.path.exists(path):
        temp_mp3 = path.replace(".wav", ".mp3")
        comm = edge_tts.Communicate(text=speaker["text"], voice=speaker["voice"])
        await comm.save(temp_mp3)
        audio, sr = sf.read(temp_mp3)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sf.write(path, audio, 16000)
        if os.path.exists(temp_mp3):
            os.remove(temp_mp3)
    return path

def get_reference_speakers():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    speakers = []
    for s in SPEAKERS:
        try:
            path = loop.run_until_complete(_create_speaker_sample(s))
            speakers.append({
                "id": s["id"],
                "name": s["name"],
                "path": path,
                "text": s["text"]
            })
        except Exception as e:
            print(f"[Project 65] Error creating speaker {s['name']}: {e}")
    loop.close()
    return speakers

if __name__ == "__main__":
    spks = get_reference_speakers()
    print(f"[Project 65] Prepared {len(spks)} reference speakers.")
