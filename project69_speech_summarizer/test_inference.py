"""
Project 69: Speech Summarizer & Action Item Extractor Test Script
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from project69_speech_summarizer.dataset import prepare_speech_summarizer_dataset
from project69_speech_summarizer.model import speech_summarizer_engine


def main():
    print("=== Testing Project 69: Speech Summarizer & Action Extractor ===")
    sample_paths = prepare_speech_summarizer_dataset()

    if not sample_paths:
        print("No audio samples found.")
        return

    for i, audio_path in enumerate(sample_paths):
        filename = os.path.basename(audio_path)
        print(f"\n--- Testing Sample {i+1}: {filename} ---")
        report = speech_summarizer_engine.process_audio(audio_path)

        print(f"\n[Whisper Transcript] ({report['duration_seconds']}s):")
        print(f'"{report["transcript"]}"')

        print(f"\n[Executive Summary]:")
        print(f"{report['executive_summary']}")

        print(f"\n[Key Takeaways]:")
        for t in report['takeaways']:
            print(f"  • {t}")

        print(f"\n[Extracted Action Items] ({len(report['action_items'])} items):")
        for item in report['action_items']:
            print(f"  [{item['priority'].upper()}] Task: {item['task']}")
            print(f"       Assignee: {item['assignee']} | Deadline: {item['deadline']} | Status: {item['status']}")

        print(f"\n[Key Topics]:")
        for top in report['topics'][:5]:
            print(f"  #{top['topic']} (weight: {top['relevance']})", end="  ")
        print()

        print(f"\n[Analytics]:")
        for k, v in report['metrics'].items():
            print(f"  {k}: {v}")
        print("-" * 50)

    print("\n[OK] Project 69 Speech Summarizer & Action Extractor test passed successfully.")
    print("========================================================\n")


if __name__ == "__main__":
    main()
