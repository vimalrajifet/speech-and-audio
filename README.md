# 🎙️ AURA | Speech & Audio AI Suite (Projects 61 to 70)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![OpenAI Whisper](https://img.shields.io/badge/ASR-Whisper%20Base-412991?style=for-the-badge&logo=openai&logoColor=white)](https://github.com/openai/whisper)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-Ready-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)

A comprehensive, production-grade speech and audio intelligence suite comprising **10 standalone AI projects (Projects 61 through 70)** from the master syllabus. Built with an elite **Luxury White & Sandalwood** full-stack web interface and a standalone **Google Colab Notebook** for interactive experimentation.

---

## 🌟 Key Features

- **10 Modular Neural Engines**: Each project includes its own model architecture, dataset generator/loader, training pipeline, and standalone test script.
- **100% Free & Open-Source**: Powered by OpenAI Whisper (`base`), Microsoft Edge Neural TTS, PyTorch CNNs, Librosa, and Scikit-Learn. Zero paid API keys or subscription services required.
- **Bypass FFmpeg Dependencies**: Optimized audio loading using `soundfile` and `librosa` directly from memory and disk.
- **Luxury White & Sandalwood Design System**: Responsive web UI featuring ivory glassmorphism (`#FDFBF7`), sandalwood gold accents (`#C68B59`), real-time HTML5 canvas audio visualizers, and interactive acoustic parameter controls.
- **Dual Verification Modes**:
  1. **Full-Stack Web App**: FastAPI REST backend + live browser frontend with microphone recording, drag-and-drop file upload, and 1-click curated audio samples.
  2. **Standalone Colab Notebook**: [`Speech_and_Audio_Projects_61_to_70_Colab.ipynb`](Speech_and_Audio_Projects_61_to_70_Colab.ipynb) with self-contained audio synthesis, training loops, and inline audio playback.

---

## 📋 Projects Catalog (Projects 61 to 70)

| Project | Project Title | Neural Architecture | Dataset / Source | Output & Key Deliverable |
|:---:|---|---|---|---|
| **61** | **Speech-to-Text (STT)** | OpenAI Whisper (`base`) + Google STT | LibriSpeech Clean Speech | Robust speech transcription, word metrics, timestamps, and audio playback. |
| **62** | **Neural Text-to-Speech (TTS)** | Microsoft Edge Neural Voices | LJSpeech prompts & custom text | Natural multi-speaker voice synthesis (Guy, Jenny, Aria, Sonia) with rate and pitch prosody. |
| **63** | **Language Identification (LID)** | Whisper 80-channel Log-Mel + `pycountry` | Multilingual Speech Corpus | Spoken language detector, confidence scores, and top-5 language probability bar chart. |
| **64** | **UrbanSound8K Classifier** | Custom PyTorch 2D-CNN (`SoundCNN`) | UrbanSound8K (10 classes) | Environmental sound classification (sirens, car horns, dog barks, etc.) with 40-dim MFCC heatmaps. |
| **65** | **Voice Cloning Mini Studio** | Speaker Timbre Encoder + Transfer | Reference Vocal Samples | Extracts speaker pitch F0 and formant envelope to synthesize custom text in cloned voice timbre. |
| **66** | **Real-Time Streaming Transcriber** | Rolling Ring Buffer + RMS VAD | Chunked audio stream | Low-latency live streaming speech transcription with silence gating. |
| **67** | **Speech Emotion Classifier** | Multi-Feature Affect Classifier | RAVDESS (8 emotions) | Classifies 8 emotional states (Happy, Sad, Angry, Fearful, Calm, etc.) using pitch and spectral energy. |
| **68** | **Keyword Spotting & Automation** | Speech Commands PyTorch CNN | Speech Commands (10 words) | Low-latency trigger word detection (*yes, no, up, down, on, off, stop, go*) + automated smart home actions. |
| **69** | **Speech Summarizer & Briefing** | Whisper ASR + Extractive NLP | Multi-Speaker Meeting Dialogue | Meeting audio transcription, executive summary brief, key takeaways, and parsed action items table. |
| **70** | **Audio Event Detection** | Multi-Scale PyTorch 2D-CNN | ESC-50 Environmental Audio | Surveillance audio event detection with safety threat severity escalation (Normal, Warning, Critical). |

---

## 📁 Repository Structure

```
.
├── backend/
│   ├── app.py                         # Master FastAPI server exposing endpoints for Projects 61-70
│   ├── test_api.py                    # Comprehensive end-to-end integration test suite
│   ├── utils_audio.py                 # Audio conversion, spectrogram, and waveform plotting
│   └── requirements.txt               # Backend Python dependencies
│
├── frontend/
│   ├── index.html                     # Responsive single-page application with 10-engine navigation
│   ├── style.css                      # Luxury White & Sandalwood design system
│   └── app.js                         # Web Audio recording, visualizers, and API controllers
│
├── project61_stt/                     # Project 61: Speech-to-Text (Whisper)
├── project62_tts/                     # Project 62: Text-to-Speech (Edge-TTS)
├── project63_lang_id/                 # Project 63: Language Identification
├── project64_sound_clf/               # Project 64: UrbanSound8K PyTorch CNN
├── project65_voice_clone/             # Project 65: Voice Cloning Mini Studio
├── project66_realtime_transcriber/    # Project 66: Real-Time Streaming Transcriber
├── project67_emotion_clf/             # Project 67: Speech Emotion Recognition
├── project68_keyword_spotting/        # Project 68: Keyword Spotting & Automation
├── project69_speech_summarizer/       # Project 69: Meeting Summarization & Action Parser
├── project70_audio_event_detection/   # Project 70: Audio Event Detection & Safety Alarms
│
├── Speech_and_Audio_Projects_61_to_70_Colab.ipynb  # Self-contained Google Colab Master Notebook
├── generate_colab_notebook.py         # Automated notebook generation script
└── README.md                          # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/vimalrajifet/speech-and-audio.git
cd speech-and-audio
```

### 2. Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### 3. Launch the Web Platform
```bash
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser and navigate to **`http://127.0.0.1:8000`** to explore all 10 speech intelligence engines.

### 4. Run Automated End-to-End Tests
```bash
python backend/test_api.py
```
This tests all 10 models and API endpoints sequentially.

---

## 📓 Running in Google Colab

To experiment without local setup:
1. Open [Google Colab](https://colab.research.google.com).
2. Click **Upload** and select [`Speech_and_Audio_Projects_61_to_70_Colab.ipynb`](Speech_and_Audio_Projects_61_to_70_Colab.ipynb).
3. Select a GPU runtime via `Runtime` ➔ `Change runtime type` ➔ `T4 GPU` (optional, CPU also supported).
4. Click `Runtime` ➔ `Run all` (`Ctrl + F9`).
5. All 10 models will synthesize test audio, train CNNs, and produce interactive audio players and visualizations directly in your browser.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
