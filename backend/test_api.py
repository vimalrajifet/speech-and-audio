"""
Comprehensive End-to-End Test Suite for AURA Speech AI FastAPI Backend
Validates API endpoints across all 10 speech intelligence projects (61-70).
"""

import os
import sys

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

def run_suite():
    print("=" * 70)
    print("🚀 STARTING E2E INTEGRATION TEST SUITE: PROJECTS 61-70")
    print("=" * 70)

    # 1. Health check
    res = client.get("/api/health")
    assert res.status_code == 200, f"Health check failed: {res.text}"
    print("✅ System Health Check Passed:", res.json()["status"])

    # 2. Projects Catalog
    res = client.get("/api/projects")
    assert res.status_code == 200
    projects = res.json()["projects"]
    assert len(projects) == 10, f"Expected 10 projects, got {len(projects)}"
    print(f"✅ Projects Catalog Verified: {len(projects)}/10 projects present.")

    # 3. Project 61: STT
    print("\n--- Testing Project 61: Speech-to-Text ---")
    res = client.post("/api/project61/transcribe", data={"sample_file": "librispeech_sample1.wav"})
    assert res.status_code == 200, f"P61 failed: {res.text}"
    p61_data = res.json()
    print(f"✅ P61 Transcription: '{p61_data.get('text')[:40]}...' (Confidence: {p61_data.get('confidence')})")

    # 4. Project 62: TTS
    print("\n--- Testing Project 62: Neural TTS ---")
    res = client.post("/api/project62/synthesize", json={
        "text": "Antigravity voice synthesis operational.",
        "voice": "en-US-GuyNeural",
        "pitch": "+0Hz",
        "rate": "+0%"
    })
    assert res.status_code == 200, f"P62 failed: {res.text}"
    p62_data = res.json()
    print(f"✅ P62 Synthesized Audio: URL {p62_data.get('audio_url')} ({p62_data.get('text_length')} chars)")

    # 5. Project 63: Language Identification
    print("\n--- Testing Project 63: Language Identification ---")
    res = client.post("/api/project63/identify-language", data={"sample_file": "sample_spanish.wav"})
    assert res.status_code == 200, f"P63 failed: {res.text}"
    p63_data = res.json()
    print(f"✅ P63 Language Detected: {p63_data.get('detected_language_name')} ({p63_data.get('detected_language_code')}) - {p63_data.get('confidence_percentage')}")

    # 6. Project 64: UrbanSound8K Classifier
    print("\n--- Testing Project 64: UrbanSound8K Classifier ---")
    res = client.post("/api/project64/classify-sound", data={"sample_file": "siren_sample.wav"})
    assert res.status_code == 200, f"P64 failed: {res.text}"
    p64_data = res.json()
    print(f"✅ P64 Sound Class: {p64_data.get('predicted_class')} (Confidence: {p64_data.get('confidence_percentage')})")

    # 7. Project 65: Voice Cloning Mini Project
    print("\n--- Testing Project 65: Voice Cloning ---")
    res = client.post("/api/project65/clone-voice", data={
        "text": "Testing voice cloning timbre transfer.",
        "sample_file": "reference_female_elena.wav"
    })
    assert res.status_code == 200, f"P65 failed: {res.text}"
    p65_data = res.json()
    print(f"✅ P65 Cloned Audio: URL {p65_data.get('cloned_audio_url')} (Similarity: {p65_data.get('similarity_score')})")

    # 8. Project 66: Real-Time Transcriber
    print("\n--- Testing Project 66: Streaming Transcriber Session ---")
    res = client.post("/api/project66/start-session")
    assert res.status_code == 200, f"P66 session failed: {res.text}"
    sess_id = res.json()["session_id"]
    print(f"✅ P66 Streaming Session Initialized: {sess_id}")

    res = client.post("/api/project66/stream-chunk", data={"session_id": sess_id, "sample_file": "librispeech_sample1.wav"})
    assert res.status_code == 200, f"P66 chunk failed: {res.text}"
    print(f"✅ P66 Chunk Streamed: Status={res.json().get('status')}")

    # 9. Project 67: Voice Emotion Classification
    print("\n--- Testing Project 67: Emotion Classifier ---")
    res = client.post("/api/project67/emotion", data={"sample_file": "03-01-03-01-01-01-01.wav"})
    assert res.status_code == 200, f"P67 failed: {res.text}"
    p67_data = res.json()
    print(f"✅ P67 Predicted Emotion: {p67_data.get('predicted_emotion').upper()} (Confidence: {p67_data.get('confidence'):.2f})")

    # 10. Project 68: Keyword Spotting
    print("\n--- Testing Project 68: Keyword Spotting & Automation ---")
    res = client.post("/api/project68/keyword", data={"sample_file": "kw_on_1.wav"})
    assert res.status_code == 200, f"P68 failed: {res.text}"
    p68_data = res.json()
    print(f"✅ P68 Spotted Keyword: '{p68_data.get('spotted_keyword')}' -> Action: {p68_data.get('triggered_action', {}).get('action')}")

    # 11. Project 69: Speech Summarizer
    print("\n--- Testing Project 69: Meeting Summarizer ---")
    res = client.post("/api/project69/summarize", data={"sample_file": "sprint_planning_meeting.wav"})
    assert res.status_code == 200, f"P69 failed: {res.text}"
    p69_data = res.json()
    actions = p69_data.get("action_items", [])
    print(f"✅ P69 Summary Generated: {len(actions)} action items parsed.")

    # 12. Project 70: Audio Event Detection
    print("\n--- Testing Project 70: Audio Event Detection ---")
    res = client.post("/api/project70/events", data={"sample_file": "event_siren_var1.wav"})
    assert res.status_code == 200, f"P70 failed: {res.text}"
    p70_data = res.json()
    print(f"✅ P70 Primary Event: {p70_data.get('primary_event')} (Severity: {p70_data.get('severity')})")

    print("\n" + "=" * 70)
    print("🏆 ALL 10 PROJECTS VERIFIED & PASSING END-TO-END IN FASTAPI SERVER!")
    print("=" * 70)

if __name__ == "__main__":
    run_suite()
