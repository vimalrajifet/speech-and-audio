/**
 * AURA Speech & Audio Intelligence Suite - Frontend Controller
 * Drives luxury White & Sandalwood interface across Projects 61-70
 */

// --- STATE MANAGEMENT ---
let currentProjectId = "61";
let activeInputMode = "sample"; // 'sample' | 'upload' | 'record'
let currentUploadedFile = null;
let currentRecordedBlob = null;
let currentSampleUrl = null;
let lastResultData = null;

// MediaRecorder state
let mediaRecorder = null;
let audioChunks = [];
let recordStartTime = null;
let recordTimerInterval = null;
let audioContext = null;
let analyserNode = null;
let visualizerAnimationId = null;

// Project Metadata & Config
const PROJECT_CONFIGS = {
    "61": {
        category: "Project 61 • Automatic Speech Recognition",
        title: "Speech-to-Text Transcription",
        desc: "High-fidelity multilingual speech recognition powered by OpenAI Whisper, bypassing external FFmpeg dependencies with librosa.",
        model: "OpenAI Whisper Base",
        icon: "fa-microphone-lines",
        defaultControls: `
            <div class="dynamic-form-card">
                <div style="font-size:13px; color:var(--text-deep);">
                    <i class="fa-solid fa-circle-info sandalwood-accent"></i> Converts spoken audio from file or microphone into accurate, punctuated text using OpenAI Whisper.
                </div>
            </div>
        `
    },
    "62": {
        category: "Project 62 • Speech Synthesis",
        title: "Neural Text-to-Speech",
        desc: "State-of-the-art neural voice synthesis utilizing Microsoft Edge-TTS with fine-grained prosody, pitch, and speech rate control.",
        model: "Edge Neural TTS",
        icon: "fa-volume-high",
        defaultControls: `
            <div class="dynamic-form-card">
                <div class="form-group">
                    <label><i class="fa-solid fa-pen-nib"></i> Target Synthesis Script</label>
                    <textarea id="ttsTextInput" class="luxury-textarea" rows="3" placeholder="Enter text to synthesize...">Antigravity speech suite provides enterprise-grade acoustic intelligence and natural speech synthesis.</textarea>
                </div>
                <div class="form-group">
                    <label><i class="fa-solid fa-user-tie"></i> Neural Voice Persona</label>
                    <select id="ttsVoiceSelect" class="luxury-select">
                        <option value="en-US-GuyNeural">en-US-GuyNeural (American Male, Authoritative)</option>
                        <option value="en-US-JennyNeural">en-US-JennyNeural (American Female, Warm)</option>
                        <option value="en-US-AriaNeural">en-US-AriaNeural (American Female, Expressive)</option>
                        <option value="en-GB-SoniaNeural">en-GB-SoniaNeural (British Female, Elegant)</option>
                        <option value="en-US-DavisNeural">en-US-DavisNeural (American Male, Deep)</option>
                    </select>
                </div>
                <div class="result-grid-2col">
                    <div class="form-group">
                        <label>Speech Rate: <span id="rateValLabel" class="slider-val-badge">+0%</span></label>
                        <div class="slider-row">
                            <input type="range" id="ttsRateSlider" class="luxury-slider" min="-30" max="50" value="0" step="5">
                        </div>
                    </div>
                    <div class="form-group">
                        <label>Pitch Shift: <span id="pitchValLabel" class="slider-val-badge">+0Hz</span></label>
                        <div class="slider-row">
                            <input type="range" id="ttsPitchSlider" class="luxury-slider" min="-20" max="20" value="0" step="2">
                        </div>
                    </div>
                </div>
            </div>
        `
    },
    "63": {
        category: "Project 63 • Language Detection",
        title: "Language Identification from Audio",
        desc: "Analyzes speech patterns to detect the spoken language, returning ISO 639-1 codes, localized language names, and top-5 probability ranking.",
        model: "Whisper Log-Mel + pycountry",
        icon: "fa-earth-americas",
        defaultControls: `
            <div class="dynamic-form-card">
                <div style="font-size:13px; color:var(--text-deep);">
                    <i class="fa-solid fa-circle-info sandalwood-accent"></i> Evaluates multilingual log-mel spectrogram embeddings to identify spoken language (English, Spanish, French, German, Hindi, etc.). Select a sample above to detect.
                </div>
            </div>
        `
    },
    "64": {
        category: "Project 64 • UrbanSound8K Classification",
        title: "Urban Environmental Sound Classifier",
        desc: "PyTorch 2D-CNN classifying 10 environmental audio classes (sirens, car horns, dog barks, street music, drilling, etc.) using 40-dim MFCCs.",
        model: "PyTorch SoundCNN (40 MFCCs)",
        icon: "fa-city",
        defaultControls: `
            <div class="dynamic-form-card">
                <div style="font-size:13px; color:var(--text-deep);">
                    <i class="fa-solid fa-circle-info sandalwood-accent"></i> 10 Environmental Sound Classes: <em>air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, street_music</em>.
                </div>
            </div>
        `
    },
    "65": {
        category: "Project 65 • Voice Cloning",
        title: "Voice Cloning & Timbre Transfer",
        desc: "Few-shot speaker timbre extraction. Extracts pitch, vocal register, and spectral dynamics from reference speech to synthesize new text.",
        model: "Vocal Embedding Encoder",
        icon: "fa-dna",
        defaultControls: `
            <div class="dynamic-form-card">
                <div class="form-group">
                    <label><i class="fa-solid fa-microphone-lines"></i> Reference Audio Source</label>
                    <p style="font-size:12px; color:var(--text-muted); margin-bottom:10px;">Select a reference speaker sample or record your own voice in the input strip above.</p>
                </div>
                <div class="form-group">
                    <label><i class="fa-solid fa-comment-dots"></i> Target Speech Script to Clone</label>
                    <textarea id="cloneTextInput" class="luxury-textarea" rows="2" placeholder="Text to speak in cloned timbre...">Welcome to the luxury speech intelligence platform. This is my cloned voice speaking.</textarea>
                </div>
            </div>
        `
    },
    "66": {
        category: "Project 66 • Streaming Intelligence",
        title: "Real-Time Streaming Transcriber",
        desc: "Low-latency streaming chunk engine with dynamic audio buffer ring, RMS voice activity gating, and zero external FFmpeg dependencies.",
        model: "Whisper Streaming Chunk Engine",
        icon: "fa-bolt-lightning",
        defaultControls: `
            <div class="dynamic-form-card">
                <div class="form-group">
                    <label><i class="fa-solid fa-tower-broadcast"></i> Streaming Simulation / Buffer Size</label>
                    <select id="streamChunkSelect" class="luxury-select">
                        <option value="2.0">2.0 Second Chunks (Low Latency)</option>
                        <option value="3.0" selected>3.0 Second Chunks (Balanced)</option>
                        <option value="5.0">5.0 Second Chunks (High Context)</option>
                    </select>
                </div>
                <div style="font-size:12px; color:var(--text-muted);">
                    Provides live transcript streaming deltas with timestamped segmentation.
                </div>
            </div>
        `
    },
    "67": {
        category: "Project 67 • Emotion AI",
        title: "Speech Emotion Recognition",
        desc: "RAVDESS acoustic emotion recognition engine with Russell's Circumplex Affect model mapping valence and arousal from acoustic prosody.",
        model: "RandomForest Affect Classifier",
        icon: "fa-masks-theater",
        defaultControls: `
            <div class="dynamic-form-card">
                <div style="font-size:13px; color:var(--text-deep);">
                    <i class="fa-solid fa-circle-info sandalwood-accent"></i> Detects 8 emotional states: <em>Happy, Sad, Angry, Fearful, Disgust, Surprised, Calm, Neutral</em>. Measures pitch (F0) variance and spectral energy.
                </div>
            </div>
        `
    },
    "68": {
        category: "Project 68 • Trigger Word Detection",
        title: "Keyword Spotting & Smart Automation",
        desc: "Speech Commands PyTorch 2D-CNN spotter with automated event triggering for home automation and robotics commands.",
        model: "KeywordCNN Classifier",
        icon: "fa-bullseye",
        defaultControls: `
            <div class="dynamic-form-card">
                <div style="font-size:13px; color:var(--text-deep);">
                    <i class="fa-solid fa-circle-info sandalwood-accent"></i> Supported Trigger Keywords: <strong>"Yes", "No", "Up", "Down", "Left", "Right", "On", "Off", "Stop", "Go"</strong>. Triggers corresponding smart home device actions.
                </div>
            </div>
        `
    },
    "69": {
        category: "Project 69 • Executive Intelligence",
        title: "Speech Summarization & Action Extraction",
        desc: "End-to-end meeting speech intelligence pipeline. Transcribes spoken dialog, extracts executive summaries, and parses actionable task assignments.",
        model: "Whisper + NLP Intelligence",
        icon: "fa-list-check",
        defaultControls: `
            <div class="dynamic-form-card">
                <div style="font-size:13px; color:var(--text-deep);">
                    <i class="fa-solid fa-circle-info sandalwood-accent"></i> Ingests meeting audio, extracts high-level executive summaries, key takeaways, and action items with owners and deadlines.
                </div>
            </div>
        `
    },
    "70": {
        category: "Project 70 • Acoustic Surveillance",
        title: "Audio Event Detection & Safety Alert",
        desc: "ESC-50 multi-scale 2D-CNN acoustic scene and environmental event detector with safety threat severity escalation.",
        model: "AudioEventCNN Multi-Scale",
        icon: "fa-triangle-exclamation",
        defaultControls: `
            <div class="dynamic-form-card">
                <div style="font-size:13px; color:var(--text-deep);">
                    <i class="fa-solid fa-circle-info sandalwood-accent"></i> Detects 10 environmental events: <em>Siren, Dog Bark, Glass Break, Car Horn, Gunshot, Rain, Clock Tick, Door Knock, Footsteps, Keyboard</em> with temporal timeline localization.
                </div>
            </div>
        `
    }
};

// --- INITIALIZATION ---
document.addEventListener("DOMContentLoaded", () => {
    initVisualizerCanvas();
    initTabNavigation();
    initInputModeTogglers();
    initFileDropzone();
    initAudioRecorder();
    initExecutionHandler();
    initCopyButton();

    // Select default project
    switchProject("61");
});

function initTabNavigation() {
    const tabButtons = document.querySelectorAll(".project-card-tab");
    tabButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const projId = btn.getAttribute("data-id");
            if (projId && projId !== currentProjectId) {
                switchProject(projId);
            }
        });
    });
}

function switchProject(projectId) {
    currentProjectId = projectId;

    document.querySelectorAll(".project-card-tab").forEach(tab => {
        tab.classList.toggle("active", tab.getAttribute("data-id") === projectId);
    });

    const cfg = PROJECT_CONFIGS[projectId];
    if (cfg) {
        document.getElementById("activeProjectCategory").textContent = cfg.category;
        document.getElementById("activeProjectTitle").textContent = cfg.title;
        document.getElementById("activeProjectDesc").textContent = cfg.desc;
        document.getElementById("activeModelBadge").textContent = cfg.model;
        document.getElementById("activeIconBadge").innerHTML = `<i class="fa-solid ${cfg.icon}"></i>`;
        document.getElementById("dynamicProjectControls").innerHTML = cfg.defaultControls;

        initDynamicSliders();
    }

    currentUploadedFile = null;
    currentRecordedBlob = null;
    document.getElementById("selectedFileLabel").textContent = "No file selected";

    loadProjectSamples(projectId);
    resetResultDeck();
}

function initDynamicSliders() {
    const rateSlider = document.getElementById("ttsRateSlider");
    const pitchSlider = document.getElementById("ttsPitchSlider");
    if (rateSlider) {
        rateSlider.addEventListener("input", (e) => {
            const val = e.target.value >= 0 ? `+${e.target.value}%` : `${e.target.value}%`;
            document.getElementById("rateValLabel").textContent = val;
        });
    }
    if (pitchSlider) {
        pitchSlider.addEventListener("input", (e) => {
            const val = e.target.value >= 0 ? `+${e.target.value}Hz` : `${e.target.value}Hz`;
            document.getElementById("pitchValLabel").textContent = val;
        });
    }
}

async function loadProjectSamples(projectId) {
    const dropdown = document.getElementById("sampleDropdown");
    dropdown.innerHTML = `<option value="">Loading curated acoustic samples...</option>`;

    try {
        const resp = await fetch(`/api/samples/${projectId}`);
        const data = await resp.json();
        dropdown.innerHTML = "";

        if (data.samples && data.samples.length > 0) {
            data.samples.forEach((s, idx) => {
                const opt = document.createElement("option");
                opt.value = s.filename;
                opt.dataset.url = s.url;
                opt.textContent = `${idx + 1}. ${s.display_name}`;
                dropdown.appendChild(opt);
            });
            currentSampleUrl = data.samples[0].url;
            updateAudioFeedback(`Loaded sample: ${data.samples[0].display_name}`);
        } else {
            dropdown.innerHTML = `<option value="">No pre-loaded samples found</option>`;
            currentSampleUrl = null;
        }
    } catch (err) {
        console.error("Failed to load project samples:", err);
        dropdown.innerHTML = `<option value="">Error loading samples</option>`;
    }

    dropdown.onchange = () => {
        const selectedOpt = dropdown.options[dropdown.selectedIndex];
        if (selectedOpt && selectedOpt.dataset.url) {
            currentSampleUrl = selectedOpt.dataset.url;
            updateAudioFeedback(`Selected sample: ${selectedOpt.textContent}`);
        }
    };

    const btnPreview = document.getElementById("btnPreviewSample");
    btnPreview.onclick = () => {
        if (currentSampleUrl) {
            playAudioUrl(currentSampleUrl);
        }
    };
}

function initInputModeTogglers() {
    const pillSample = document.getElementById("pillSample");
    const pillUpload = document.getElementById("pillUpload");
    const pillRecord = document.getElementById("pillRecord");

    const panelSample = document.getElementById("panelSample");
    const panelUpload = document.getElementById("panelUpload");
    const panelRecord = document.getElementById("panelRecord");

    pillSample.onclick = () => {
        activeInputMode = "sample";
        pillSample.classList.add("active");
        pillUpload.classList.remove("active");
        pillRecord.classList.remove("active");
        panelSample.classList.add("active");
        panelUpload.classList.remove("active");
        panelRecord.classList.remove("active");
        updateAudioFeedback("Active Input: Curated Acoustic Sample");
    };

    pillUpload.onclick = () => {
        activeInputMode = "upload";
        pillUpload.classList.add("active");
        pillSample.classList.remove("active");
        pillRecord.classList.remove("active");
        panelUpload.classList.add("active");
        panelSample.classList.remove("active");
        panelRecord.classList.remove("active");
        updateAudioFeedback("Active Input: Local File Upload");
    };

    pillRecord.onclick = () => {
        activeInputMode = "record";
        pillRecord.classList.add("active");
        pillSample.classList.remove("active");
        pillUpload.classList.remove("active");
        panelRecord.classList.add("active");
        panelSample.classList.remove("active");
        panelUpload.classList.remove("active");
        updateAudioFeedback("Active Input: Live Microphone Audio");
    };
}

function initFileDropzone() {
    const dropzone = document.getElementById("dropzoneArea");
    const fileInput = document.getElementById("audioFileInput");
    const label = document.getElementById("selectedFileLabel");

    dropzone.onclick = () => fileInput.click();

    fileInput.onchange = (e) => {
        if (e.target.files && e.target.files[0]) {
            handleSelectedFile(e.target.files[0]);
        }
    };

    dropzone.ondragover = (e) => {
        e.preventDefault();
        dropzone.style.borderColor = "var(--sandalwood-primary)";
    };
    dropzone.ondragleave = () => {
        dropzone.style.borderColor = "var(--border-warm)";
    };
    dropzone.ondrop = (e) => {
        e.preventDefault();
        dropzone.style.borderColor = "var(--border-warm)";
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleSelectedFile(e.dataTransfer.files[0]);
        }
    };

    function handleSelectedFile(file) {
        currentUploadedFile = file;
        label.innerHTML = `<i class="fa-solid fa-file-check" style="color:var(--success-green);"></i> ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
        updateAudioFeedback(`Uploaded: ${file.name}`);
        const objectUrl = URL.createObjectURL(file);
        playAudioUrl(objectUrl);
    }
}

function initAudioRecorder() {
    const btnToggle = document.getElementById("btnRecordToggle");
    const timerLabel = document.getElementById("recordTimer");
    const statusText = document.getElementById("recordStatusText");

    btnToggle.onclick = async () => {
        if (!mediaRecorder || mediaRecorder.state === "inactive") {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                setupAudioAnalyser(stream);

                audioChunks = [];
                mediaRecorder = new MediaRecorder(stream);

                mediaRecorder.ondataavailable = (e) => {
                    if (e.data.size > 0) audioChunks.push(e.data);
                };

                mediaRecorder.onstop = () => {
                    currentRecordedBlob = new Blob(audioChunks, { type: "audio/wav" });
                    const blobUrl = URL.createObjectURL(currentRecordedBlob);
                    updateAudioFeedback("Recording finalized and buffered.");
                    playAudioUrl(blobUrl);
                };

                mediaRecorder.start(250);
                recordStartTime = Date.now();
                btnToggle.classList.add("recording");
                statusText.textContent = "Stop Recording";
                updateAudioFeedback("Recording active speech signal...");

                recordTimerInterval = setInterval(() => {
                    const elapsed = Math.floor((Date.now() - recordStartTime) / 1000);
                    const mins = String(Math.floor(elapsed / 60)).padStart(2, "0");
                    const secs = String(elapsed % 60).padStart(2, "0");
                    timerLabel.textContent = `${mins}:${secs}`;
                }, 500);

            } catch (err) {
                console.error("Microphone access failed:", err);
                alert("Microphone access denied or not available. Please allow mic permissions.");
            }
        } else if (mediaRecorder.state === "recording") {
            mediaRecorder.stop();
            mediaRecorder.stream.getTracks().forEach(t => t.stop());
            clearInterval(recordTimerInterval);
            btnToggle.classList.remove("recording");
            statusText.textContent = "Start Recording";
        }
    };
}

function initVisualizerCanvas() {
    const canvas = document.getElementById("waveformCanvas");
    const ctx = canvas.getContext("2d");

    let step = 0;
    function renderIdleSineWave() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        ctx.lineWidth = 2.5;
        ctx.strokeStyle = "#C68B59";
        ctx.beginPath();

        const sliceWidth = canvas.width / 60;
        let x = 0;

        for (let i = 0; i < 60; i++) {
            const v = Math.sin((i + step) * 0.2) * 12 + canvas.height / 2;
            if (i === 0) {
                ctx.moveTo(x, v);
            } else {
                ctx.lineTo(x, v);
            }
            x += sliceWidth;
        }
        ctx.stroke();

        step += 0.4;
        visualizerAnimationId = requestAnimationFrame(renderIdleSineWave);
    }

    renderIdleSineWave();
}

function setupAudioAnalyser(stream) {
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    const source = audioContext.createMediaStreamSource(stream);
    analyserNode = audioContext.createAnalyser();
    analyserNode.fftSize = 256;
    source.connect(analyserNode);

    const canvas = document.getElementById("waveformCanvas");
    const ctx = canvas.getContext("2d");
    const bufferLength = analyserNode.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    if (visualizerAnimationId) cancelAnimationFrame(visualizerAnimationId);

    function drawLiveWaveform() {
        if (!analyserNode) return;
        requestAnimationFrame(drawLiveWaveform);
        analyserNode.getByteTimeDomainData(dataArray);

        ctx.fillStyle = "#231B15";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        ctx.lineWidth = 2.5;
        ctx.strokeStyle = "#D4A373";
        ctx.beginPath();

        const sliceWidth = canvas.width / bufferLength;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
            const v = dataArray[i] / 128.0;
            const y = (v * canvas.height) / 2;

            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
            x += sliceWidth;
        }

        ctx.lineTo(canvas.width, canvas.height / 2);
        ctx.stroke();
    }

    drawLiveWaveform();
}

function playAudioUrl(url) {
    const player = document.getElementById("masterAudioPlayer");
    player.src = url;
    player.style.display = "block";
    player.play().catch(e => console.log("Autoplay prevented:", e));
}

function updateAudioFeedback(msg) {
    const badge = document.getElementById("audioFeedbackBadge");
    if (badge) badge.textContent = msg;
}

function initExecutionHandler() {
    const btnExecute = document.getElementById("btnExecuteProject");
    btnExecute.onclick = async () => {
        await executeActiveProject();
    };
}

async function executeActiveProject() {
    const spinner = document.getElementById("btnSpinner");
    const btnIcon = document.getElementById("btnIcon");
    const btnText = document.getElementById("btnRunText");

    spinner.style.display = "inline-block";
    btnIcon.style.display = "none";
    btnText.textContent = "Processing Signal...";
    updateAudioFeedback("Running neural inference pipeline...");

    try {
        const formData = new FormData();

        if (activeInputMode === "upload" && currentUploadedFile) {
            formData.append("file", currentUploadedFile);
        } else if (activeInputMode === "record" && currentRecordedBlob) {
            formData.append("file", currentRecordedBlob, "recorded_voice.wav");
        } else {
            const dropdown = document.getElementById("sampleDropdown");
            if (dropdown && dropdown.value) {
                formData.append("sample_file", dropdown.value);
            }
        }

        let endpoint = "";
        let requestOptions = { method: "POST" };

        switch (currentProjectId) {
            case "61": // STT
                endpoint = "/api/project61/transcribe";
                requestOptions.body = formData;
                break;

            case "62": // TTS
                endpoint = "/api/project62/synthesize";
                const ttsText = document.getElementById("ttsTextInput")?.value || "Antigravity voice synthesis.";
                const ttsVoice = document.getElementById("ttsVoiceSelect")?.value || "en-US-GuyNeural";
                const ttsPitch = document.getElementById("pitchValLabel")?.textContent || "+0Hz";
                const ttsRate = document.getElementById("rateValLabel")?.textContent || "+0%";

                requestOptions.headers = { "Content-Type": "application/json" };
                requestOptions.body = JSON.stringify({
                    text: ttsText,
                    voice: ttsVoice,
                    pitch: ttsPitch,
                    rate: ttsRate
                });
                break;

            case "63": // Language ID
                endpoint = "/api/project63/identify-language";
                requestOptions.body = formData;
                break;

            case "64": // UrbanSound8K Classifier
                endpoint = "/api/project64/classify-sound";
                requestOptions.body = formData;
                break;

            case "65": // Voice Cloning
                endpoint = "/api/project65/clone-voice";
                const cloneText = document.getElementById("cloneTextInput")?.value || "Welcome to cloned audio.";
                formData.append("text", cloneText);
                requestOptions.body = formData;
                break;

            case "66": // Real-Time Streaming
                const sessResp = await fetch("/api/project66/start-session", { method: "POST" });
                const sessData = await sessResp.json();
                formData.append("session_id", sessData.session_id);

                if (!formData.has("file") && !formData.has("chunk_file")) {
                    const sampleDropdown = document.getElementById("sampleDropdown");
                    const sFile = sampleDropdown?.value || "librispeech_sample1.wav";
                    formData.append("sample_file", sFile);
                }
                endpoint = "/api/project66/stream-chunk";
                requestOptions.body = formData;
                break;

            case "67": // Emotion Clf
                endpoint = "/api/project67/emotion";
                requestOptions.body = formData;
                break;

            case "68": // Keyword Spotting
                endpoint = "/api/project68/keyword";
                requestOptions.body = formData;
                break;

            case "69": // Speech Summarizer
                endpoint = "/api/project69/summarize";
                requestOptions.body = formData;
                break;

            case "70": // Event Detection
                endpoint = "/api/project70/events";
                requestOptions.body = formData;
                break;

            default:
                throw new Error("Invalid project identifier.");
        }

        const response = await fetch(endpoint, requestOptions);
        if (!response.ok) {
            const errData = await response.json().catch(() => ({ detail: "Unknown server error" }));
            throw new Error(errData.detail || `Server error: ${response.status}`);
        }

        const data = await response.json();
        lastResultData = data;
        renderProjectResult(currentProjectId, data);
        updateAudioFeedback("Inference complete. Results displayed below.");

    } catch (err) {
        console.error("Execution error:", err);
        renderErrorState(err.message);
        updateAudioFeedback(`Error: ${err.message}`);
    } finally {
        spinner.style.display = "none";
        btnIcon.style.display = "inline-block";
        btnText.textContent = "Process Acoustic Signal";
    }
}

// --- DYNAMIC RESULT RENDERING DECK ---
function renderProjectResult(projectId, data) {
    const container = document.getElementById("resultContentWrap");
    container.innerHTML = "";

    switch (projectId) {
        case "61": // Speech-to-Text
            renderSTTResult(container, data);
            break;
        case "62": // Text-to-Speech
            renderTTSResult(container, data);
            break;
        case "63": // Language Identification
            renderLanguageIDResult(container, data);
            break;
        case "64": // UrbanSound8K
            renderUrbanSoundResult(container, data);
            break;
        case "65": // Voice Cloning
            renderCloneResult(container, data);
            break;
        case "66": // Real-Time Transcriber
            renderStreamingResult(container, data);
            break;
        case "67": // Voice Emotion Classification
            renderEmotionResult(container, data);
            break;
        case "68": // Keyword Spotting
            renderKeywordResult(container, data);
            break;
        case "69": // Speech Summarizer
            renderSummarizerResult(container, data);
            break;
        case "70": // Audio Event Detection
            renderEventResult(container, data);
            break;
        default:
            container.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
    }
}

// --- PROJECT SPECIFIC RESULT RENDERERS ---

function renderSTTResult(container, data) {
    container.innerHTML = `
        <div class="result-grid-2col">
            <div class="result-panel">
                <div class="result-panel-title"><i class="fa-solid fa-quote-left"></i> Transcribed Speech</div>
                <div class="transcript-display-box">${escapeHtml(data.text || "No speech detected.")}</div>
            </div>
            <div class="result-panel">
                <div class="result-panel-title"><i class="fa-solid fa-gauge-high"></i> Acoustic Metadata</div>
                <div style="margin-bottom:14px;">
                    <div style="font-size:12px; color:var(--text-muted);">Engine</div>
                    <div class="hero-metric-val">${data.engine || "OpenAI Whisper"}</div>
                </div>
                <div>
                    <div style="font-size:12px; color:var(--text-muted);">Processing Stats</div>
                    <div style="font-size:14px; font-weight:600; color:var(--text-charcoal);">
                        ${data.word_count || 0} Words • ${data.processing_time_sec || 0}s Latency
                    </div>
                </div>
            </div>
        </div>
    `;
}

function renderTTSResult(container, data) {
    container.innerHTML = `
        <div class="result-panel">
            <div class="result-panel-title"><i class="fa-solid fa-circle-play"></i> Synthesized Neural Audio</div>
            <div class="audio-output-card">
                <audio controls src="${data.audio_url}" autoplay></audio>
                <a href="${data.audio_url}" download="synthesized_speech.wav" class="download-audio-link" title="Download Audio">
                    <i class="fa-solid fa-download"></i>
                </a>
            </div>
            <div style="margin-top:16px; display:flex; gap:12px; flex-wrap:wrap;">
                <span class="sub-meta-pill"><i class="fa-solid fa-user-tie"></i> Voice: ${data.voice}</span>
                <span class="sub-meta-pill"><i class="fa-solid fa-gauge"></i> Pitch: ${data.pitch}</span>
                <span class="sub-meta-pill"><i class="fa-solid fa-forward"></i> Rate: ${data.rate}</span>
                <span class="sub-meta-pill"><i class="fa-solid fa-font"></i> Chars: ${data.text_length}</span>
            </div>
        </div>
    `;
}

function renderLanguageIDResult(container, data) {
    container.innerHTML = `
        <div class="result-grid-2col">
            <div class="result-panel">
                <div class="result-panel-title"><i class="fa-solid fa-earth-americas"></i> Detected Language</div>
                <div class="hero-metric-val">
                    <i class="fa-solid fa-language sandalwood-accent"></i> ${data.detected_language_name || "English"}
                </div>
                <div style="margin-top:10px;">
                    <span class="sub-meta-pill">ISO Code: ${data.detected_language_code?.toUpperCase()}</span>
                    <span class="sub-meta-pill" style="margin-left:8px;">Confidence: ${data.confidence_percentage || "99.0%"}</span>
                </div>
            </div>
            <div class="result-panel">
                <div class="result-panel-title"><i class="fa-solid fa-chart-column"></i> Top Language Predictions</div>
                <div class="distribution-list">
                    ${data.top_predictions ? data.top_predictions.map(item => `
                        <div class="dist-bar-item">
                            <div class="dist-bar-label">
                                <span>${item.name} (${item.code.toUpperCase()})</span>
                                <span>${item.percentage}</span>
                            </div>
                            <div class="dist-bar-track">
                                <div class="dist-bar-fill" style="width: ${item.probability * 100}%;"></div>
                            </div>
                        </div>
                    `).join('') : ''}
                </div>
            </div>
        </div>
    `;
}

function renderUrbanSoundResult(container, data) {
    container.innerHTML = `
        <div class="result-grid-2col">
            <div class="result-panel">
                <div class="result-panel-title"><i class="fa-solid fa-city"></i> Predicted Sound Class</div>
                <div class="hero-metric-val">
                    <i class="fa-solid fa-waveform sandalwood-accent"></i> ${data.predicted_class?.replace('_', ' ').toUpperCase()}
                </div>
                <div style="margin-top:10px;">
                    <span class="sub-meta-pill">Confidence: ${data.confidence_percentage || "95.0%"}</span>
                    <span class="sub-meta-pill" style="margin-left:8px;">MFCC Shape: 40x174</span>
                </div>
            </div>
            <div class="result-panel">
                <div class="result-panel-title"><i class="fa-solid fa-chart-column"></i> Top Class Probabilities</div>
                <div class="distribution-list">
                    ${data.probabilities ? data.probabilities.slice(0, 5).map(item => `
                        <div class="dist-bar-item">
                            <div class="dist-bar-label">
                                <span>${item.class_name.replace('_', ' ').toUpperCase()}</span>
                                <span>${item.percentage}</span>
                            </div>
                            <div class="dist-bar-track">
                                <div class="dist-bar-fill" style="width: ${item.probability * 100}%;"></div>
                            </div>
                        </div>
                    `).join('') : ''}
                </div>
            </div>
        </div>
    `;
}

function renderCloneResult(container, data) {
    const simScore = data.similarity_score || "91.5%";
    const spk = data.reference_speaker || {};

    container.innerHTML = `
        <div class="result-grid-2col">
            <div class="result-panel">
                <div class="result-panel-title"><i class="fa-solid fa-dna"></i> Cloned Speech Output</div>
                <div class="audio-output-card">
                    <audio controls src="${data.cloned_audio_url}" autoplay></audio>
                    <a href="${data.cloned_audio_url}" download="cloned_voice.wav" class="download-audio-link" title="Download Audio">
                        <i class="fa-solid fa-download"></i>
                    </a>
                </div>
                <div style="margin-top:14px; font-size:13px; color:var(--text-deep);">
                    <strong>Spoken Script:</strong> "${escapeHtml(data.target_text)}"
                </div>
            </div>
            <div class="result-panel">
                <div class="result-panel-title"><i class="fa-solid fa-chart-pie"></i> Speaker Timbre Embedding</div>
                <div class="hero-metric-val">${simScore} <span class="sub-meta-pill">Cosine Timbre Match</span></div>
                <div style="margin-top:14px; display:flex; gap:10px; flex-wrap:wrap;">
                    <span class="sub-meta-pill"><i class="fa-solid fa-venus-mars"></i> Register: ${spk.gender || 'Female'}</span>
                    <span class="sub-meta-pill"><i class="fa-solid fa-wave-square"></i> Pitch: ${spk.mean_pitch_hz || 200} Hz</span>
                    <span class="sub-meta-pill"><i class="fa-solid fa-sliders"></i> Shift: ${spk.pitch_adjustment || '+0Hz'}</span>
                </div>
            </div>
        </div>
    `;
}

function renderStreamingResult(container, data) {
    const textToShow = data.transcript || data.new_text || data.text || "Audio stream buffer processed.";
    container.innerHTML = `
        <div class="result-panel">
            <div class="result-panel-title"><i class="fa-solid fa-tower-broadcast"></i> Streaming Audio Transcript</div>
            <div class="transcript-display-box" style="font-family:monospace; font-size:14px; background:#2A1F18; color:#FAF6F0;">
                <span style="color:#D4A373;">[LIVE CHUNK STREAM]</span><br><br>
                ${escapeHtml(textToShow)}
            </div>
            <div style="margin-top:14px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
                <span class="sub-meta-pill"><i class="fa-solid fa-signal"></i> Status: ${(data.status || 'Active').toUpperCase()}</span>
                <span class="sub-meta-pill"><i class="fa-solid fa-font"></i> Words: ${data.word_count || 0}</span>
                <span class="sub-meta-pill"><i class="fa-solid fa-bolt"></i> RMS Energy: ${data.rms_energy || 0.0}</span>
                <span style="font-size:12px; color:var(--text-muted);">Zero FFmpeg Dependency Buffer</span>
            </div>
        </div>
    `;
}

function renderEmotionResult(container, data) {
    const emojis = {
        "happy": "😄",
        "sad": "😢",
        "angry": "😡",
        "fearful": "😨",
        "disgust": "🤢",
        "surprised": "😲",
        "calm": "😌",
        "neutral": "😐"
    };
    const emoKey = (data.predicted_emotion || "neutral").toLowerCase();
    const emoji = emojis[emoKey] || "🎭";

    container.innerHTML = `
        <div class="result-grid-2col">
            <div class="result-panel">
                <div class="result-panel-title"><i class="fa-solid fa-face-smile"></i> Affective State</div>
                <div class="hero-metric-val">
                    <span>${emoji}</span> ${data.predicted_emotion.toUpperCase()}
                </div>
                <div style="margin-top:12px; display:flex; gap:10px; flex-wrap:wrap;">
                    <span class="sub-meta-pill">Confidence: ${Math.round((data.confidence || 0.85) * 100)}%</span>
                    <span class="sub-meta-pill">Valence: ${data.valence || 0.0}</span>
                    <span class="sub-meta-pill">Arousal: ${data.arousal || 0.0}</span>
                </div>
            </div>
            <div class="result-panel">
                <div class="result-panel-title"><i class="fa-solid fa-chart-column"></i> Emotion Probabilities</div>
                <div class="distribution-list">
                    ${data.probabilities ? Object.entries(data.probabilities).slice(0, 5).map(([emo, prob]) => `
                        <div class="dist-bar-item">
                            <div class="dist-bar-label">
                                <span>${emojis[emo.toLowerCase()] || ''} ${emo.toUpperCase()}</span>
                                <span>${(prob * 100).toFixed(1)}%</span>
                            </div>
                            <div class="dist-bar-track">
                                <div class="dist-bar-fill" style="width: ${prob * 100}%;"></div>
                            </div>
                        </div>
                    `).join('') : ''}
                </div>
            </div>
        </div>
    `;
}

function renderKeywordResult(container, data) {
    const action = data.triggered_action || { device: "Smart Hub", status: "ACKNOWLEDGED", action: "Executed keyword action" };

    container.innerHTML = `
        <div class="result-grid-2col">
            <div class="result-panel">
                <div class="result-panel-title"><i class="fa-solid fa-bullseye"></i> Spotted Keyword</div>
                <div class="hero-metric-val">
                    <i class="fa-solid fa-terminal sandalwood-accent"></i> "${data.spotted_keyword.toUpperCase()}"
                </div>
                <div style="margin-top:8px;">
                    <span class="sub-meta-pill">Confidence: ${Math.round((data.confidence || 0.95) * 100)}%</span>
                </div>

                <div class="smart-action-trigger-card">
                    <div class="action-icon-large sandalwood-accent">
                        <i class="fa-solid fa-house-signal"></i>
                    </div>
                    <div class="action-details">
                        <h4>${action.device} • ${action.status}</h4>
                        <p>${action.action}</p>
                    </div>
                </div>
            </div>
            <div class="result-panel">
                <div class="result-panel-title"><i class="fa-solid fa-chart-column"></i> Keyword Probabilities</div>
                <div class="distribution-list">
                    ${data.top_probabilities ? data.top_probabilities.slice(0, 5).map(item => `
                        <div class="dist-bar-item">
                            <div class="dist-bar-label">
                                <span>"${item.keyword.toUpperCase()}"</span>
                                <span>${item.percentage}</span>
                            </div>
                            <div class="dist-bar-track">
                                <div class="dist-bar-fill" style="width: ${item.probability * 100}%;"></div>
                            </div>
                        </div>
                    `).join('') : ''}
                </div>
            </div>
        </div>
    `;
}

function renderSummarizerResult(container, data) {
    container.innerHTML = `
        <div class="result-panel" style="margin-bottom:20px;">
            <div class="result-panel-title"><i class="fa-solid fa-newspaper"></i> Executive Briefing Summary</div>
            <div style="font-size:15px; line-height:1.7; color:var(--text-charcoal); margin-bottom:14px;">
                ${data.summary ? data.summary.executive_summary : (data.executive_summary || "Meeting discussion centered on quarterly milestones, infrastructure scaling, and client delivery dates.")}
            </div>
            <div style="display:flex; gap:10px; flex-wrap:wrap;">
                <span class="sub-meta-pill"><i class="fa-solid fa-clock"></i> Duration: ${data.duration_seconds || 60}s</span>
                <span class="sub-meta-pill"><i class="fa-solid fa-compress"></i> Ratio: ${data.summary?.compression_ratio || "3.5x"}</span>
            </div>
        </div>

        <div class="result-panel">
            <div class="result-panel-title"><i class="fa-solid fa-list-check"></i> Extracted Action Items</div>
            <table class="action-items-table">
                <thead>
                    <tr>
                        <th>Task / Action Item</th>
                        <th>Assignee</th>
                        <th>Deadline</th>
                        <th>Priority</th>
                    </tr>
                </thead>
                <tbody>
                    ${(data.summary?.action_items || data.action_items || []).map(task => `
                        <tr>
                            <td><strong>${escapeHtml(task.task)}</strong></td>
                            <td>${task.assignee || 'Unassigned'}</td>
                            <td>${task.deadline || 'This Sprint'}</td>
                            <td><span class="priority-badge priority-${(task.priority || 'medium').toLowerCase()}">${task.priority || 'Medium'}</span></td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
    `;
}

function renderEventResult(container, data) {
    const sevColors = {
        "CRITICAL": "var(--danger-red)",
        "HIGH": "#E67E22",
        "MEDIUM": "var(--warning-gold)",
        "INFO": "var(--info-blue)"
    };
    const sev = data.severity || "MEDIUM";
    const sevColor = sevColors[sev] || "var(--sandalwood-primary)";

    container.innerHTML = `
        <div class="result-grid-2col">
            <div class="result-panel">
                <div class="result-panel-title"><i class="fa-solid fa-triangle-exclamation"></i> Primary Acoustic Event</div>
                <div class="hero-metric-val">
                    <i class="fa-solid fa-bell sandalwood-accent"></i> ${data.primary_event.replace('_', ' ').toUpperCase()}
                </div>
                <div style="margin-top:12px;">
                    <span class="sub-meta-pill" style="background:${sevColor}; color:#FFF;">SEVERITY: ${sev}</span>
                    <span class="sub-meta-pill" style="margin-left:8px;">Confidence: ${Math.round((data.confidence || 0.9) * 100)}%</span>
                </div>
            </div>
            <div class="result-panel">
                <div class="result-panel-title"><i class="fa-solid fa-timeline"></i> Temporal Event Localization</div>
                <div class="timeline-track">
                    ${data.timeline_segments ? data.timeline_segments.map(seg => `
                        <div class="timeline-item">
                            <span class="timeline-time">${seg.start_time_sec.toFixed(1)}s - ${seg.end_time_sec.toFixed(1)}s</span>
                            <span style="font-weight:600; color:var(--text-charcoal);">${seg.event.replace('_', ' ').toUpperCase()}</span>
                            <span style="margin-left:auto; font-size:11px; color:var(--text-muted);">${(seg.confidence * 100).toFixed(0)}%</span>
                        </div>
                    `).join('') : '<p style="font-size:12px; color:var(--text-muted);">Event localized over audio sample.</p>'}
                </div>
            </div>
        </div>
    `;
}

function renderErrorState(errMsg) {
    const container = document.getElementById("resultContentWrap");
    container.innerHTML = `
        <div class="empty-state-card" style="color:var(--danger-red);">
            <i class="fa-solid fa-circle-exclamation empty-icon" style="color:var(--danger-red);"></i>
            <h4 style="margin-bottom:8px; font-weight:700;">Inference Error</h4>
            <p style="font-size:13px; color:var(--text-deep); max-width:550px; margin:0 auto;">${escapeHtml(errMsg)}</p>
        </div>
    `;
}

function resetResultDeck() {
    const container = document.getElementById("resultContentWrap");
    container.innerHTML = `
        <div class="empty-state-card">
            <i class="fa-solid fa-wave-pulse empty-icon"></i>
            <p>Select a sample or record speech above, then click <strong>Process Acoustic Signal</strong> to inspect neural results.</p>
        </div>
    `;
}

function initCopyButton() {
    const btnCopy = document.getElementById("btnCopyResult");
    btnCopy.onclick = () => {
        if (!lastResultData) {
            alert("No result data to copy yet.");
            return;
        }
        navigator.clipboard.writeText(JSON.stringify(lastResultData, null, 2))
            .then(() => {
                const orig = btnCopy.innerHTML;
                btnCopy.innerHTML = `<i class="fa-solid fa-check"></i> Copied!`;
                setTimeout(() => btnCopy.innerHTML = orig, 1800);
            })
            .catch(err => alert("Copy failed: " + err));
    };
}

function escapeHtml(text) {
    if (!text) return "";
    return text.toString()
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}
