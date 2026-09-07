"""
Project 62: Text-to-Speech (TTS) Studio
Uses Microsoft Edge Neural TTS (edge-tts) for ultra-realistic human speech,
with multi-voice selection (male/female, US/UK/India/Australia), pitch and speed control.
"""

import os
import asyncio
import edge_tts

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Curated high-fidelity neural voices
AVAILABLE_VOICES = {
    "en-US-GuyNeural": {"name": "Guy (US Male, Natural Narration)", "gender": "Male", "locale": "en-US"},
    "en-US-JennyNeural": {"name": "Jenny (US Female, Warm & Expressive)", "gender": "Female", "locale": "en-US"},
    "en-US-AriaNeural": {"name": "Aria (US Female, Professional)", "gender": "Female", "locale": "en-US"},
    "en-GB-SoniaNeural": {"name": "Sonia (British Female, Elegant)", "gender": "Female", "locale": "en-GB"},
    "en-GB-RyanNeural": {"name": "Ryan (British Male, Clear)", "gender": "Male", "locale": "en-GB"},
    "en-IN-PrabhatNeural": {"name": "Prabhat (Indian English Male)", "gender": "Male", "locale": "en-IN"},
    "en-IN-NeerjaNeural": {"name": "Neerja (Indian English Female)", "gender": "Female", "locale": "en-IN"},
    "en-AU-NatashaNeural": {"name": "Natasha (Australian Female)", "gender": "Female", "locale": "en-AU"}
}

class TextToSpeechSynthesizer:
    def __init__(self, default_voice: str = "en-US-JennyNeural"):
        self.default_voice = default_voice

    async def synthesize_async(
        self,
        text: str,
        voice: str = None,
        rate: str = "+0%",
        pitch: str = "+0Hz",
        output_filename: str = "speech_output.mp3"
    ) -> str:
        """Synthesizes text to speech audio using edge-tts asynchronously."""
        voice = voice if (voice and voice in AVAILABLE_VOICES) else self.default_voice
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)
        await communicate.save(output_path)
        return output_path

    def synthesize(
        self,
        text: str,
        voice: str = None,
        rate: str = "+0%",
        pitch: str = "+0Hz",
        output_filename: str = "speech_output.mp3"
    ) -> str:
        """Synchronous wrapper for synthesize_async."""
        return asyncio.run(
            self.synthesize_async(text, voice=voice, rate=rate, pitch=pitch, output_filename=output_filename)
        )

# Global synthesizer instance
tts_engine = TextToSpeechSynthesizer()
