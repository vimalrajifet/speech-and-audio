"""
FastAPI Master Backend Server for Speech & Audio AI Projects 61-70
Unified platform serving all 10 modular speech intelligence systems matching the official syllabus:
- Project 61: Speech-to-Text Converter (Whisper ASR)
- Project 62: Text-to-Speech (Neural TTS)
- Project 63: Language Identification from Audio (Whisper + pycountry)
- Project 64: UrbanSound8K Sound Classifier (PyTorch CNN)
- Project 65: Voice Cloning Mini Project (Timbre & Vocal Transfer)
- Project 66: Real-Time Audio Transcriber (Low-latency streaming)
- Project 67: Speech Emotion Recognition (RAVDESS Affect)
- Project 68: Keyword Spotting (Speech Commands CNN & Automation)
- Project 69: Speech Summarization & Meeting Assistant (Whisper + NLP)
- Project 70: Audio Event Detection & Safety Alerts (ESC-50 CNN)
"""

import os
import sys
import uuid
import shutil
import asyncio
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

AUDIO_CACHE_DIR = os.path.join(BASE_DIR, "audio_cache")
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# Import all 10 official project models
from project61_stt.model import converter as p61_converter
from project62_tts.model import tts_engine as p62_tts_engine, AVAILABLE_VOICES
from project63_lang_id.model import language_detector as p63_lang_detector
from project64_sound_clf.model import sound_classifier as p64_sound_classifier
from project65_voice_clone.model import voice_cloner as p65_voice_cloner
from project66_realtime_transcriber.model import streaming_transcriber as p66_streaming_transcriber
from project67_emotion_clf.model import voice_emotion_classifier as p67_emotion_classifier
from project68_keyword_spotting.model import keyword_spotter as p68_keyword_spotter
from project69_speech_summarizer.model import speech_summarizer_engine as p69_summarizer_engine
from project70_audio_event_detection.model import audio_event_detector as p70_event_detector

app = FastAPI(
    title="AURA Speech & Audio AI Suite",
    description="Enterprise Multi-Model Speech Processing Platform (Projects 61-70)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STREAMING_SESSIONS: dict[str, Any] = {}


def save_upload_file(upload_file: UploadFile) -> str:
    ext = os.path.splitext(upload_file.filename)[1] or ".wav"
    unique_name = f"upload_{uuid.uuid4().hex[:10]}{ext}"
    dest_path = os.path.join(AUDIO_CACHE_DIR, unique_name)
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return dest_path


PROJECT_METADATA = [
    {
        "id": "61",
        "name": "Speech-to-Text Converter",
        "subtitle": "Robust Multilingual Speech Recognition",
        "model": "OpenAI Whisper Base",
        "category": "Transcription",
        "icon": "fa-microphone-lines",
        "color": "#C68B59"
    },
    {
        "id": "62",
        "name": "Text-to-Speech (TTS)",
        "subtitle": "Neural Voice Synthesis with Dynamics",
        "model": "Microsoft Edge Neural TTS",
        "category": "Synthesis",
        "icon": "fa-volume-high",
        "color": "#D4A373"
    },
    {
        "id": "63",
        "name": "Language Identification",
        "subtitle": "Multilingual Acoustic Pattern Classifier",
        "model": "Whisper Mel-Spectrogram + pycountry",
        "category": "Language Detection",
        "icon": "fa-earth-americas",
        "color": "#A0522D"
    },
    {
        "id": "64",
        "name": "UrbanSound8K Classifier",
        "subtitle": "Environmental Acoustic Sound Classifier",
        "model": "PyTorch 2D-CNN (MFCCs)",
        "category": "Acoustic ML",
        "icon": "fa-city",
        "color": "#8B5A2B"
    },
    {
        "id": "65",
        "name": "Voice Cloning Mini Project",
        "subtitle": "Acoustic Speaker Embedding & Timbre Transfer",
        "model": "Vocal Embedding Encoder",
        "category": "Voice Cloning",
        "icon": "fa-dna",
        "color": "#CD853F"
    },
    {
        "id": "66",
        "name": "Real-Time Audio Transcriber",
        "subtitle": "Live Streaming Buffer & Low-Latency STT",
        "model": "Whisper Streaming Chunk Engine",
        "category": "Real-Time",
        "icon": "fa-bolt-lightning",
        "color": "#B8860B"
    },
    {
        "id": "67",
        "name": "Speech Emotion Recognition",
        "subtitle": "RAVDESS Circumplex Affect Analyzer",
        "model": "RandomForest Acoustic Classifier",
        "category": "Emotion AI",
        "icon": "fa-masks-theater",
        "color": "#DEB887"
    },
    {
        "id": "68",
        "name": "Keyword Spotting & Automation",
        "subtitle": "Speech Commands PyTorch 2D-CNN",
        "model": "KeywordCNN Classifier",
        "category": "Trigger Word",
        "icon": "fa-bullseye",
        "color": "#C68B59"
    },
    {
        "id": "69",
        "name": "Speech Summarization",
        "subtitle": "Executive Briefing & Action Item Extractor",
        "model": "Whisper + NLP Intelligence",
        "category": "NLP Summarizer",
        "icon": "fa-list-check",
        "color": "#8C6239"
    },
    {
        "id": "70",
        "name": "Audio Event Detection",
        "subtitle": "ESC-50 Safety & Acoustic Scene Classifier",
        "model": "AudioEventCNN Multi-Scale",
        "category": "Audio Surveillance",
        "icon": "fa-triangle-exclamation",
        "color": "#A0522D"
    }
]

PROJECT_SAMPLES_MAP = {
    "61": "project61_stt/samples",
    "62": "project62_tts/samples",
    "63": "project63_lang_id/samples",
    "64": "project64_sound_clf/samples",
    "65": "project65_voice_clone/samples",
    "66": "project61_stt/samples",
    "67": "project67_emotion_clf/samples",
    "68": "project68_keyword_spotting/samples",
    "69": "project69_speech_summarizer/samples",
    "70": "project70_audio_event_detection/samples"
}


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "platform": "AURA Speech & Audio AI Suite",
        "projects_available": 10,
        "aesthetic": "White & Sandalwood",
        "cache_dir": AUDIO_CACHE_DIR
    }


@app.get("/api/projects")
def get_projects():
    return {"projects": PROJECT_METADATA}


@app.get("/api/samples/{project_id}")
def get_project_samples(project_id: str):
    dir_rel = PROJECT_SAMPLES_MAP.get(project_id)
    if not dir_rel:
        return {"samples": []}

    sample_dir = os.path.join(BASE_DIR, dir_rel)
    if not os.path.exists(sample_dir):
        return {"samples": []}

    samples = []
    for f in os.listdir(sample_dir):
        if f.endswith(".wav") or f.endswith(".mp3"):
            samples.append({
                "filename": f,
                "display_name": f.replace(".wav", "").replace("_", " ").title(),
                "url": f"/api/sample-audio/{project_id}/{f}"
            })
    return {"samples": samples}


@app.get("/api/sample-audio/{project_id}/{filename}")
def serve_sample_audio(project_id: str, filename: str):
    dir_rel = PROJECT_SAMPLES_MAP.get(project_id)
    if not dir_rel:
        raise HTTPException(status_code=404, detail="Project not found")

    file_path = os.path.join(BASE_DIR, dir_rel, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    media_type = "audio/mpeg" if filename.endswith(".mp3") else "audio/wav"
    return FileResponse(file_path, media_type=media_type)


@app.get("/audio-cache/{filename}")
def serve_cached_audio(filename: str):
    file_path = os.path.join(AUDIO_CACHE_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Cached audio not found")
    media_type = "audio/wav" if filename.endswith(".wav") else "audio/mpeg"
    return FileResponse(file_path, media_type=media_type)


# --- PROJECT 61: SPEECH TO TEXT ---
@app.post("/api/project61/transcribe")
async def api_project61_transcribe(
    file: Optional[UploadFile] = File(None),
    sample_file: Optional[str] = Form(None)
):
    target_path = None
    if file:
        target_path = save_upload_file(file)
    elif sample_file:
        target_path = os.path.join(BASE_DIR, "project61_stt", "samples", sample_file)

    if not target_path or not os.path.exists(target_path):
        raise HTTPException(status_code=400, detail="No audio file or sample provided.")

    try:
        result = p61_converter.transcribe_file(target_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- PROJECT 62: TEXT TO SPEECH ---
class TTSRequest(BaseModel):
    text: str
    voice: str = "en-US-GuyNeural"
    pitch: str = "+0Hz"
    rate: str = "+0%"

@app.post("/api/project62/synthesize")
async def api_project62_synthesize(req: TTSRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    unique_filename = f"tts_{uuid.uuid4().hex[:8]}.mp3"

    try:
        saved_path = await p62_tts_engine.synthesize_async(
            text=req.text,
            voice=req.voice,
            pitch=req.pitch,
            rate=req.rate,
            output_filename=unique_filename
        )
        cached_path = os.path.join(AUDIO_CACHE_DIR, unique_filename)
        shutil.copyfile(saved_path, cached_path)
        return {
            "success": True,
            "audio_url": f"/audio-cache/{unique_filename}",
            "voice": req.voice,
            "pitch": req.pitch,
            "rate": req.rate,
            "text_length": len(req.text)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/project62/voices")
def api_project62_voices():
    return {"voices": AVAILABLE_VOICES}


# --- PROJECT 63: LANGUAGE IDENTIFICATION ---
@app.post("/api/project63/identify-language")
async def api_project63_identify_language(
    file: Optional[UploadFile] = File(None),
    sample_file: Optional[str] = Form(None)
):
    target_path = None
    if file:
        target_path = save_upload_file(file)
    elif sample_file:
        target_path = os.path.join(BASE_DIR, "project63_lang_id", "samples", sample_file)

    if not target_path or not os.path.exists(target_path):
        raise HTTPException(status_code=400, detail="Audio file required for language identification.")

    try:
        res = p63_lang_detector.identify(target_path)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- PROJECT 64: URBANSOUND8K SOUND CLASSIFIER ---
@app.post("/api/project64/classify-sound")
async def api_project64_classify_sound(
    file: Optional[UploadFile] = File(None),
    sample_file: Optional[str] = Form(None)
):
    target_path = None
    if file:
        target_path = save_upload_file(file)
    elif sample_file:
        target_path = os.path.join(BASE_DIR, "project64_sound_clf", "samples", sample_file)

    if not target_path or not os.path.exists(target_path):
        raise HTTPException(status_code=400, detail="Audio file required for sound classification.")

    try:
        res = p64_sound_classifier.predict(target_path)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- PROJECT 65: VOICE CLONING MINI PROJECT ---
@app.post("/api/project65/clone-voice")
async def api_project65_clone_voice(
    text: str = Form(...),
    file: Optional[UploadFile] = File(None),
    sample_file: Optional[str] = Form(None)
):
    target_path = None
    if file:
        target_path = save_upload_file(file)
    elif sample_file:
        target_path = os.path.join(BASE_DIR, "project65_voice_clone", "samples", sample_file)

    if not target_path or not os.path.exists(target_path):
        raise HTTPException(status_code=400, detail="Reference speaker audio required.")

    unique_filename = f"clone_{uuid.uuid4().hex[:8]}.wav"
    out_path = os.path.join(AUDIO_CACHE_DIR, unique_filename)

    try:
        res = await p65_voice_cloner.clone_voice_async(
            reference_audio_path=target_path,
            target_text=text,
            output_filename=unique_filename
        )
        # Ensure file is in audio_cache
        res["cloned_audio_url"] = f"/audio-cache/{unique_filename}"
        if os.path.exists(res.get("cloned_audio_path", "")):
            shutil.copyfile(res["cloned_audio_path"], out_path)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- PROJECT 66: REAL-TIME TRANSCRIBER ---
@app.post("/api/project66/start-session")
def api_project66_start_session():
    session_id = uuid.uuid4().hex[:12]
    p66_streaming_transcriber.get_or_create_session(session_id)
    return {"session_id": session_id, "status": "initialized"}


@app.post("/api/project66/stream-chunk")
async def api_project66_stream_chunk(
    session_id: str = Form(...),
    chunk_file: Optional[UploadFile] = File(None),
    sample_file: Optional[str] = Form(None)
):
    if chunk_file:
        chunk_bytes = await chunk_file.read()
    elif sample_file:
        sample_path = os.path.join(BASE_DIR, "project61_stt", "samples", sample_file)
        if not os.path.exists(sample_path):
            sample_path = os.path.join(BASE_DIR, "project63_lang_id", "samples", sample_file)
        with open(sample_path, "rb") as f:
            chunk_bytes = f.read()
    else:
        raise HTTPException(status_code=400, detail="Audio chunk required.")

    try:
        res = p66_streaming_transcriber.process_chunk(session_id, chunk_bytes)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- PROJECT 67: SPEECH EMOTION RECOGNITION ---
@app.post("/api/project67/emotion")
async def api_project67_emotion(
    file: Optional[UploadFile] = File(None),
    sample_file: Optional[str] = Form(None)
):
    target_path = None
    if file:
        target_path = save_upload_file(file)
    elif sample_file:
        target_path = os.path.join(BASE_DIR, "project67_emotion_clf", "samples", sample_file)

    if not target_path or not os.path.exists(target_path):
        raise HTTPException(status_code=400, detail="Voice audio file required.")

    try:
        res = p67_emotion_classifier.predict(target_path)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- PROJECT 68: KEYWORD SPOTTING & AUTOMATION ---
@app.post("/api/project68/keyword")
async def api_project68_keyword(
    file: Optional[UploadFile] = File(None),
    sample_file: Optional[str] = Form(None)
):
    target_path = None
    if file:
        target_path = save_upload_file(file)
    elif sample_file:
        target_path = os.path.join(BASE_DIR, "project68_keyword_spotting", "samples", sample_file)

    if not target_path or not os.path.exists(target_path):
        raise HTTPException(status_code=400, detail="Keyword command audio required.")

    try:
        res = p68_keyword_spotter.spot(target_path)
        # Ensure standard keys for frontend display
        res["spotted_keyword"] = res.get("keyword", "unknown")
        res["triggered_action"] = {
            "device": "Smart Home Automation Hub",
            "action": res.get("action_description", "No trigger"),
            "status": "TRIGGERED" if res.get("spotted") else "STANDBY"
        }
        res["top_probabilities"] = res.get("distribution", [])
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- PROJECT 69: SPEECH SUMMARIZATION ---
@app.post("/api/project69/summarize")
async def api_project69_summarize(
    file: Optional[UploadFile] = File(None),
    sample_file: Optional[str] = Form(None)
):
    target_path = None
    if file:
        target_path = save_upload_file(file)
    elif sample_file:
        target_path = os.path.join(BASE_DIR, "project69_speech_summarizer", "samples", sample_file)

    if not target_path or not os.path.exists(target_path):
        raise HTTPException(status_code=400, detail="Meeting recording required.")

    try:
        res = p69_summarizer_engine.process_audio(target_path)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- PROJECT 70: AUDIO EVENT DETECTION ---
@app.post("/api/project70/events")
async def api_project70_events(
    file: Optional[UploadFile] = File(None),
    sample_file: Optional[str] = Form(None)
):
    target_path = None
    if file:
        target_path = save_upload_file(file)
    elif sample_file:
        target_path = os.path.join(BASE_DIR, "project70_audio_event_detection", "samples", sample_file)

    if not target_path or not os.path.exists(target_path):
        raise HTTPException(status_code=400, detail="Acoustic event audio required.")

    try:
        res = p70_event_detector.detect(target_path)
        res["primary_event"] = res.get("top_event", "unknown")
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- MOUNT FRONTEND STATIC ASSETS ---
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    print("Starting AURA Speech AI Master Platform on http://localhost:8000 ...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
