"""
Project 63: Language Identification Dataset & Sample Audio Generator
Prepares multilingual speech samples (English, Spanish, French, German, Hindi).
"""

import os
import asyncio
import edge_tts

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "samples")
os.makedirs(SAMPLE_DIR, exist_ok=True)

MULTILINGUAL_SAMPLES = [
    {
        "language": "English",
        "code": "en",
        "voice": "en-US-JennyNeural",
        "text": "Artificial intelligence and deep learning are transforming modern speech recognition technology.",
        "filename": "sample_english.wav"
    },
    {
        "language": "Spanish",
        "code": "es",
        "voice": "es-ES-ElviraNeural",
        "text": "Hola, bienvenidos a nuestra demostracion de inteligencia artificial para reconocimiento de voz.",
        "filename": "sample_spanish.wav"
    },
    {
        "language": "French",
        "code": "fr",
        "voice": "fr-FR-DeniseNeural",
        "text": "Bonjour, cette technologie permet d'identifier automatiquement la langue parlee dans un enregistrement.",
        "filename": "sample_french.wav"
    },
    {
        "language": "German",
        "code": "de",
        "voice": "de-DE-KatjaNeural",
        "text": "Guten Tag, dieses neuronale Netzwerk erkennt die Sprache aus den akustischen Mustern.",
        "filename": "sample_german.wav"
    },
    {
        "language": "Hindi",
        "code": "hi",
        "voice": "hi-IN-SwaraNeural",
        "text": "Namaste, yeh audio artificial intelligence dwara bhasha ki pehchan karne ke liye banaya gaya hai.",
        "filename": "sample_hindi.wav"
    }
]

async def _generate_sample(sample_meta: dict) -> str:
    path = os.path.join(SAMPLE_DIR, sample_meta["filename"])
    if not os.path.exists(path):
        temp_mp3 = path.replace(".wav", ".mp3")
        comm = edge_tts.Communicate(text=sample_meta["text"], voice=sample_meta["voice"])
        await comm.save(temp_mp3)
        # Convert to standard WAV using soundfile/librosa for fast loading
        import soundfile as sf
        import librosa
        audio, sr = sf.read(temp_mp3)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sf.write(path, audio, 16000)
        if os.path.exists(temp_mp3):
            os.remove(temp_mp3)
    return path

def prepare_multilingual_samples():
    """Generates and returns available multilingual audio sample paths."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    paths = []
    for s in MULTILINGUAL_SAMPLES:
        try:
            p = loop.run_until_complete(_generate_sample(s))
            paths.append({"language": s["language"], "code": s["code"], "path": p, "text": s["text"]})
        except Exception as e:
            print(f"[Project 63] Warning generating {s['language']}: {e}")
    loop.close()
    return paths

if __name__ == "__main__":
    samples = prepare_multilingual_samples()
    print(f"[Project 63] Prepared {len(samples)} multilingual samples.")
