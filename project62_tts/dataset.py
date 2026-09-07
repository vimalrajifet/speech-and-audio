"""
Project 62: Text-to-Speech Prompts & Voice Presets Catalog
Curated sample sentences inspired by LJSpeech and conversational AI scenarios.
"""

from project62_tts.model import AVAILABLE_VOICES

# Sample prompts from LJSpeech & Conversational Scenarios
SAMPLE_PROMPTS = [
    {
        "title": "LJSpeech Acoustic Reference",
        "text": "The examination and testimony of the experts enabled the commission to conclude that five shots may have been fired.",
        "recommended_voice": "en-US-JennyNeural"
    },
    {
        "title": "AI Conversational Agent",
        "text": "Hello! Welcome to the Speech and Audio AI platform. How can I assist your research today?",
        "recommended_voice": "en-US-GuyNeural"
    },
    {
        "title": "Audiobook Narration",
        "text": "Deep in the enchanted forest, ancient trees whispered secrets to the twilight breeze.",
        "recommended_voice": "en-GB-SoniaNeural"
    },
    {
        "title": "Academic Presentation",
        "text": "Our deep learning architecture uses multi-head self-attention to process acoustic spectrograms with high fidelity.",
        "recommended_voice": "en-IN-PrabhatNeural"
    }
]

def get_sample_prompts():
    return SAMPLE_PROMPTS

def get_voice_catalog():
    return AVAILABLE_VOICES
