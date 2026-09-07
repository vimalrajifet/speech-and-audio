"""
Project 63: Language Identification Test Inference Script
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from project63_lang_id.dataset import prepare_multilingual_samples
from project63_lang_id.model import language_detector

def main():
    print("=== Testing Project 63: Language Identification ===")
    samples = prepare_multilingual_samples()
    if not samples:
        print("No samples available.")
        return

    # Test on the first sample (e.g. English or Spanish)
    for sample in samples[:2]:
        print(f"\nEvaluating: {sample['language']} ({sample['path']})")
        print(f"Ground Truth Spoken Text: \"{sample['text']}\"")
        res = language_detector.identify(sample["path"])
        print(f"Detected Language: {res['detected_language_name']} ({res['detected_language_code']})")
        print(f"Confidence: {res['confidence_percentage']}")
        print("Top 3 Predictions:")
        for pred in res["top_predictions"][:3]:
            print(f"  - {pred['name']} ({pred['code']}): {pred['percentage']}")

    print("\n[OK] Project 63 test completed successfully.")
    print("===================================================\n")

if __name__ == "__main__":
    main()
