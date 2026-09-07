"""
Project 69: Speech Summarizer & Meeting Intelligence Dataset Loader
Generates realistic multi-turn meeting dialogues and technical lectures
for speech-to-text transcription and action item extraction.
"""

import os
import asyncio
import soundfile as sf
import librosa
import numpy as np

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "samples")
os.makedirs(SAMPLE_DIR, exist_ok=True)

SAMPLES = [
    {
        "filename": "sprint_planning_meeting.wav",
        "voice": "en-US-GuyNeural",
        "text": (
            "Good morning team. Today we are aligning on our Q4 deliverables for the speech AI platform. "
            "First, Sarah must finalize the FastAPI backend routes and database schemas by this Friday. "
            "Second, Alex will migrate our Whisper transcription service to batch inference to cut latency by forty percent. "
            "Third, we need David to conduct end-to-end security penetration testing before next Tuesday. "
            "Overall, our core objective is delivering high reliability, sub-second response times, and an intuitive user interface. "
            "Let us reconvene on Thursday to review testing results and finalize the launch checklist."
        )
    },
    {
        "filename": "machine_learning_lecture.wav",
        "voice": "en-US-JennyNeural",
        "text": (
            "Welcome everyone to lecture fourteen on deep neural networks and optimization. "
            "Today we explored gradient descent, stochastic mini-batch updates, and the critical role of adaptive learning rates. "
            "Remember that choosing a learning rate that is too high causes numerical divergence, while a rate that is too small leads to excruciatingly slow convergence. "
            "For homework, all students must implement backpropagation from scratch in Python before Monday evening. "
            "Please read chapter eight of the textbook covering regularization techniques including dropout and weight decay."
        )
    }
]


async def _synthesize_sample(sample: dict) -> str:
    out_path = os.path.join(SAMPLE_DIR, sample["filename"])
    if not os.path.exists(out_path):
        temp_mp3 = out_path.replace(".wav", ".mp3")
        import edge_tts
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


def prepare_speech_summarizer_dataset():
    """Generates and returns speech summarizer sample audio paths."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    paths = []
    for s in SAMPLES:
        try:
            path = loop.run_until_complete(_synthesize_sample(s))
            paths.append(path)
        except Exception as e:
            print(f"[Project 69] Error preparing sample {s['filename']}: {e}")
    loop.close()
    return paths


if __name__ == "__main__":
    files = prepare_speech_summarizer_dataset()
    print(f"[Project 69] Generated meeting and lecture audio samples: {files}")
