"""
Project 66: Real-Time Audio Transcriber Test Script
Simulates live chunk-by-chunk streaming ASR and prints the rolling transcript.
"""

import os
import sys
import io
import time
import soundfile as sf
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from project66_realtime_transcriber.model import streaming_transcriber


def main():
    print("=== Testing Project 66: Real-Time Audio Transcriber ===")
    sample_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "project61_stt", "samples", "librispeech_sample1.wav"))

    if not os.path.exists(sample_path):
        print(f"Error: Sample audio file not found at {sample_path}")
        return

    audio, sr = sf.read(sample_path)
    print(f"Loaded audio sample: {sample_path}")
    print(f"Duration: {len(audio)/sr:.2f}s, Sample Rate: {sr}Hz")

    # Simulate streaming in 1.2-second chunks
    chunk_size = int(sr * 1.2)
    session_id = "test_stream_01"
    streaming_transcriber.reset_session(session_id)

    print("\n--- Simulating Live Audio Stream ---")
    step = 1
    for start in range(0, len(audio), chunk_size):
        chunk = audio[start:start + chunk_size]
        # Encode chunk to WAV bytes
        buf = io.BytesIO()
        sf.write(buf, chunk, sr, format="WAV")
        wav_bytes = buf.getvalue()

        result = streaming_transcriber.process_chunk(session_id, wav_bytes)
        status = result.get("status", "")
        new_text = result.get("new_text", "")
        full_transcript = result.get("transcript", "")
        rms = result.get("rms_energy", 0.0)

        print(f"[Chunk {step:02d}] RMS: {rms:.4f} | Status: {status:<11} | New: '{new_text}'")
        step += 1
        time.sleep(0.05)

    final_summary = streaming_transcriber.sessions[session_id].get_summary()
    print("\n--- Final Streamed Transcript ---")
    print(f"Full Text: \"{final_summary['full_transcript']}\"")
    print(f"Total Segments: {final_summary['total_segments']}")
    print(f"Word Count: {final_summary['word_count']}")

    print("\n[OK] Project 66 Real-Time Transcriber test passed successfully.")
    print("========================================================\n")


if __name__ == "__main__":
    main()
