"""
Project 67: Voice Emotion Classifier Test Inference Script
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from project67_emotion_clf.dataset import prepare_emotion_dataset
from project67_emotion_clf.model import voice_emotion_classifier


def main():
    print("=== Testing Project 67: Voice Emotion Classifier ===")
    _, _, sample_files = prepare_emotion_dataset()

    if not sample_files:
        print("No emotion audio samples found.")
        return

    print(f"Found {len(sample_files)} emotion audio samples. Testing predictions on first 3 samples:\n")

    for i, file_path in enumerate(sample_files[:3]):
        filename = os.path.basename(file_path)
        print(f"--- Sample {i+1}: {filename} ---")
        res = voice_emotion_classifier.predict(file_path)
        print(f"Predicted Emotion: {res['emoji']} {res['predicted_emotion'].upper()} ({res['confidence_percentage']})")
        print(f"Emotional Coordinates: Valence={res['valence']}, Arousal={res['arousal']}")
        print(f"Acoustic Profile:")
        print(f"  - Mean Pitch: {res['acoustic_profile']['mean_pitch_hz']} Hz")
        print(f"  - RMS Energy: {res['acoustic_profile']['rms_energy']}")
        print(f"  - Spectral Centroid: {res['acoustic_profile']['spectral_centroid_hz']} Hz")
        print(f"  - Speaking Tempo: {res['acoustic_profile']['tempo_bpm']} BPM")
        print("Top 3 Probability Distribution:")
        for dist in res['distribution'][:3]:
            print(f"  - {dist['emoji']} {dist['emotion']}: {dist['percentage']}")
        print()

    print("[OK] Project 67 Voice Emotion Classifier test passed successfully.")
    print("========================================================\n")


if __name__ == "__main__":
    main()
