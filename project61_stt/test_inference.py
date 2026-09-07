"""
Project 61: Speech-to-Text Test Inference Script
"""

import os
import sys

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from project61_stt.dataset import get_sample_audio_files
from project61_stt.model import converter

def main():
    print("=== Testing Project 61: Speech-to-Text Converter ===")
    samples = get_sample_audio_files()
    if not samples:
        print("No sample audio files available.")
        return

    test_file = samples[0]
    print(f"Testing on: {test_file}")

    result = converter.transcribe_file(test_file)
    print("\n--- Transcription Result ---")
    print(f"Engine: {result.get('engine')}")
    print(f"Success: {result.get('success')}")
    print(f"Text: {result.get('text')}")
    print(f"Word Count: {result.get('word_count')}")
    print(f"Processing Time: {result.get('processing_time_sec')}s")
    print("===================================================\n")

if __name__ == "__main__":
    main()
