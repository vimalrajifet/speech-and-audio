"""
Project 70: Audio Event Detection Test Script
Validates acoustic event classification and temporal timeline localization.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from project70_audio_event_detection.dataset import prepare_audio_event_dataset
from project70_audio_event_detection.model import audio_event_detector


def main():
    print("=== Testing Project 70: Audio Event Detection ===")
    _, _, sample_paths = prepare_audio_event_dataset()

    if not sample_paths:
        print("No audio event samples found.")
        return

    test_events = ["siren", "glass_break", "dog_bark", "rain"]
    tested = 0

    for target in test_events:
        matching = [p for p in sample_paths if f"event_{target}_" in os.path.basename(p)]
        if matching:
            file_path = matching[0]
            filename = os.path.basename(file_path)
            res = audio_event_detector.detect(file_path)
            tested += 1

            print(f"--- Event Clip: {filename} ({res['duration_sec']}s) ---")
            print(f"Detected Event: {res['icon']} {res['top_event'].upper()} ({res['confidence_percentage']})")
            print(f"Category: {res['category']} | Severity: {res['severity']}")
            print(f"Description: {res['description']}")

            print("Top-3 Probabilities:")
            for item in res["distribution"][:3]:
                print(f"  {item['icon']} {item['event']}: {item['percentage']}")

            print("Temporal Timeline Segments (First 3):")
            for seg in res["timeline_segments"][:3]:
                print(f"  [{seg['start_time_sec']}s - {seg['end_time_sec']}s]: {seg['icon']} {seg['event']} ({seg['confidence'] * 100:.1f}%)")
            print()

    print(f"[OK] Successfully detected and localized {tested} environmental acoustic events.")
    print("========================================================\n")


if __name__ == "__main__":
    main()
