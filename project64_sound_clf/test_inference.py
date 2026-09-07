"""
Project 64: UrbanSound8K Sound Classification Test Inference Script
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from project64_sound_clf.dataset import prepare_sample_files
from project64_sound_clf.model import sound_classifier

def main():
    print("=== Testing Project 64: UrbanSound8K Sound Classification ===")
    sample_files = prepare_sample_files()

    test_classes = ["siren", "dog_bark", "engine_idling"]
    for cls in test_classes:
        path = sample_files[cls]
        print(f"\nTesting on sample: {cls} ({path})")
        res = sound_classifier.predict(path)
        print(f"Predicted Class: {res['predicted_class']}")
        print(f"Confidence: {res['confidence_percentage']}")
        print("Top 3 Predictions:")
        for p in res["probabilities"][:3]:
            print(f"  - {p['class_name']}: {p['percentage']}")

    print("\n[OK] Project 64 inference testing finished.")
    print("===================================================\n")

if __name__ == "__main__":
    main()
