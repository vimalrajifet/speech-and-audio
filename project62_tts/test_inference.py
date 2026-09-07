"""
Project 62: Text-to-Speech Test Inference Script
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from project62_tts.model import tts_engine
from project62_tts.dataset import get_sample_prompts

def main():
    print("=== Testing Project 62: Text-to-Speech (Edge-TTS) ===")
    prompts = get_sample_prompts()
    prompt = prompts[1] # AI conversational prompt
    print(f"Synthesizing: \"{prompt['text']}\"")
    print(f"Using Voice: {prompt['recommended_voice']}")

    output_path = tts_engine.synthesize(
        text=prompt["text"],
        voice=prompt["recommended_voice"],
        output_filename="test_speech.mp3"
    )

    print(f"\n[OK] Synthesized audio saved to: {output_path}")
    print(f"File size: {os.path.getsize(output_path)} bytes")
    print("===================================================\n")

if __name__ == "__main__":
    main()
