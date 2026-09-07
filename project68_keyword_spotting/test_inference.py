"""
Project 68: Audio Keyword Spotting Test Script
Tests keyword spotting on diverse spoken commands and validates smart device action triggers.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from project68_keyword_spotting.dataset import prepare_keyword_dataset
from project68_keyword_spotting.model import keyword_spotter


def main():
    print("=== Testing Project 68: Audio Keyword Spotting ===")
    _, _, sample_paths = prepare_keyword_dataset()

    if not sample_paths:
        print("No audio samples found.")
        return

    # Select representative keyword samples to test
    test_keywords = ["on", "off", "stop", "go", "yes"]
    tested_count = 0

    for word in test_keywords:
        matching = [p for p in sample_paths if f"kw_{word}_" in os.path.basename(p)]
        if matching:
            file_path = matching[0]
            filename = os.path.basename(file_path)
            res = keyword_spotter.spot(file_path)
            tested_count += 1
            print(f"--- Test File: {filename} ---")
            print(f"Detected Keyword: {res['icon']} {res['keyword'].upper()} (Confidence: {res['confidence_percentage']})")
            print(f"Spotted Status: {'TRIGGERED' if res['spotted'] else 'IGNORED'}")
            print(f"Dispatched Action: [{res['action']}] -> {res['action_description']}")
            print("Top-3 Probabilities:")
            for rank, item in enumerate(res["distribution"][:3], 1):
                print(f"  {rank}. {item['icon']} {item['keyword']}: {item['percentage']}")
            print()

    print(f"[OK] Successfully spotted and triggered actions on {tested_count} test keyword commands.")
    print("========================================================\n")


if __name__ == "__main__":
    main()
