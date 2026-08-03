/* PianoMagic Frontend — v7.2 */

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────
const FE_VERSION = '7.2';
const API_BASE = 'https://pianomagic-api.onrender.com';  // Update if needed
// const API_BASE = 'http://localhost:8000';  // For local dev

// ─────────────────────────────────────────────
// State
// ─────────────────────────────────────────────
let currentTaskId = null;
let pollInterval = null;
let notes = [];
let audioContext = null;
let isPlaying = false;
let playStartTime = 0;
let playOffset = 0;
let animationFrame = null;
let staffZoom = 1.0;
let staffOffsetX = 0;
let isDraggingStaff = false;
let lastMouseX = 0;
let currentDuration = 0;

// ─────────────────────────────────────────────
// DOM Elements
// ─────────────────────────────────────────────
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const uploadSection = document.getElementById('uploadSection');
const progressSection = document.getElementById('progressSection');
const resultsSection = document.getElementById('resultsSection');
const progressFill = document.getElementById('progressFill');
const progressStatus = document.getElementById('progressStatus');
const progressPercent = document.getElementById('progressPercent');
const staffCanvas = document.getElementById('staffCanvas');
const playhead = document.getElementById('playhead');
const seekBar = document.getElementById('seekBar');
const currentTimeEl = document.getElementById('currentTime');
const totalTimeEl = document.getElementById('totalTime');
const audioPlayer = document.getElementById('audioPlayer');

// ─────────────────────────────────────────────
// Initialization
// ─────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('feVersion').textContent = `v${FE_VERSION}`;
    checkApiStatus();
    setupEventListeners();
    resizeStaff();
});

function setupEventListeners() {
    // Drag & Drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) handleFile(files[0]);
    });

    // File input
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleFile(e.target.files[0]);
    });

    // Staff controls
    document.getElementById('playBtn').addEventListener('click', togglePlay);
    document.getElementById('stopBtn').addEventListener('click', stopPlayback);
    document.getElementById('zoomInBtn').addEventListener('click', () => setZoom(staffZoom * 1.2));
    document.getElementById('zoomOutBtn').addEventListener('click', () => setZoom(staffZoom / 1.2));

    // Seek bar
    seekBar.addEventListener('input', (e) => {
        playOffset = (e.target.value / 100) * currentDuration;
        updatePlayhead();
        if (isPlaying) {
            playStartTime = audioContext.currentTime - playOffset;
        }
    });

    // Staff drag
    staffCanvas.addEventListener('mousedown', (e) => {
        isDraggingStaff = true;
        lastMouseX = e.clientX;
        staffCanvas.style.cursor = 'grabbing';
    });
    window.addEventListener('mousemove', (e) => {
        if (!isDraggingStaff) return;
        const dx = e.clientX - lastMouseX;
        staffOffsetX += dx;
        lastMouseX = e.clientX;
        drawStaff();
    });
    window.addEventListener('mouseup', () => {
        isDraggingStaff = false;
        staffCanvas.style.cursor = 'grab';
    });

    window.addEventListener('resize', resizeStaff);
}

// ─────────────────────────────────────────────
// API Status
// ─────────────────────────────────────────────
async function checkApiStatus() {
    const statusEl = document.getElementById('apiStatus');
    const beVersionEl = document.getElementById('beVersion');
    try {
        const res = await fetch(`${API_BASE}/health`, { method: 'GET', mode: 'cors' });
        if (res.ok) {
            const data = await res.json();
            statusEl.textContent = '🟢 API онлайн';
            statusEl.className = 'api-status online';
            beVersionEl.textContent = data.version || 'v?';
        } else {
            throw new Error('Not OK');
        }
    } catch (e) {
        statusEl.textContent = '🔴 API недоступен';
        statusEl.className = 'api-status offline';
        beVersionEl.textContent = '—';
    }
}

// ─────────────────────────────────────────────
// File Upload
// ─────────────────────────────────────────────
function handleFile(file) {
    const validTypes = ['audio/mpeg','audio/wav','audio/x-wav','audio/flac','audio/ogg','audio/mp4','audio/x-m4a'];
    const validExts = ['.mp3','.wav','.flac','.ogg','.m4a'];
    const ext = file.name.slice(file.name.lastIndexOf('.')).toLowerCase();

    if (!validTypes.includes(file.type) && !validExts.includes(ext)) {
        alert('Неподдерживаемый формат. Используйте MP3, WAV, FLAC, OGG или M4A.');
        return;
    }

    uploadFile(file);
}

async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    showProgress();
    updateProgress(0, 'Загрузка файла...');

    try {
        const res = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        if (data.task_id) {
            currentTaskId = data.task_id;
            startPolling(currentTaskId);
        } else {
            throw new Error('No task_id');
        }
    } catch (e) {
        updateProgress(0, `Ошибка загрузки: ${e.message}`);
        console.error(e);
    }
}

// ─────────────────────────────────────────────
// Progress Polling
// ─────────────────────────────────────────────
function startPolling(taskId) {
    if (pollInterval) clearInterval(pollInterval);
    pollInterval = setInterval(() => pollStatus(taskId), 1000);
}

async function pollStatus(taskId) {
    try {
        const res = await fetch(`${API_BASE}/status/${taskId}`);
        const data = await res.json();

        updateProgress(data.progress || 0, getStatusText(data.status));
        highlightStage(data.status);

        if (data.status === 'completed') {
            clearInterval(pollInterval);
            showResults(data.result);
        } else if (data.status === 'error') {
            clearInterval(pollInterval);
            updateProgress(0, `Ошибка: ${data.error || 'Unknown error'}`);
        }
    } catch (e) {
        console.error('Poll error:', e);
    }
}

function getStatusText(status) {
    const map = {
        'queued': 'В очереди...',
        'loading': 'Загрузка аудио...',
        'analyzing': 'Анализ мелодии...',
        'synthesizing': 'Синтез фортепиано...',
        'generating_score': 'Генерация нот...',
        'completed': 'Готово!',
        'error': 'Ошибка'
    };
    return map[status] || status;
}

function highlightStage(status) {
    document.querySelectorAll('.stage').forEach(el => {
        el.classList.toggle('active', el.dataset.stage === status);
        el.classList.toggle('done', 
            ['synthesizing','generating_score','completed'].includes(status) && 
            ['loading','analyzing'].includes(el.dataset.stage) ||
            status === 'completed'
        );
    });
}

function showProgress() {
    uploadSection.classList.add('hidden');
    progressSection.classList.remove('hidden');
    resultsSection.classList.add('hidden');
}

function updateProgress(percent, text) {
    progressFill.style.width = `${percent}%`;
    progressPercent.textContent = `${percent}%`;
    progressStatus.textContent = text;
}

// ─────────────────────────────────────────────
// Results Display
// ─────────────────────────────────────────────
function showResults(result) {
    progressSection.classList.add('hidden');
    resultsSection.classList.remove('hidden');

    // Info panel
    document.getElementById('noteCount').textContent = result.notes_count || '—';
    document.getElementById('rhCount').textContent = result.rh_notes || '—';
    document.getElementById('lhCount').textContent = result.lh_notes || '—';
    document.getElementById('tempoValue').textContent = result.tempo ? `${result.tempo} BPM` : '—';
    document.getElementById('keyValue').textContent = result.key || '—';

    // Notes data
    notes = result.notes || [];
    currentDuration = result.duration || 0;
    totalTimeEl.textContent = formatTime(currentDuration);
    seekBar.max = 100;
    seekBar.value = 0;

    // Setup downloads
    const wavUrl = result.wav_url ? `${API_BASE}${result.wav_url}` : '#';
    const xmlUrl = result.xml_url ? `${API_BASE}${result.xml_url}` : '#';
    document.getElementById('downloadWav').href = wavUrl;
    document.getElementById('downloadXml').href = xmlUrl;

    // Setup audio player
    audioPlayer.src = wavUrl;
    audioPlayer.load();

    // Draw staff
    resizeStaff();
    drawStaff();
}

// ─────────────────────────────────────────────
// Staff Canvas Rendering
// ─────────────────────────────────────────────
function resizeStaff() {
    const wrapper = document.querySelector('.staff-wrapper');
    const dpr = window.devicePixelRatio || 1;
    staffCanvas.width = wrapper.clientWidth * dpr;
    staffCanvas.height = 400 * dpr;
    staffCanvas.style.width = wrapper.clientWidth + 'px';
    staffCanvas.style.height = '400px';
    const ctx = staffCanvas.getContext('2d');
    ctx.scale(dpr, dpr);
    drawStaff();
}

function setZoom(z) {
    staffZoom = Math.max(0.1, Math.min(10, z));
    drawStaff();
}

function drawStaff() {
    const ctx = staffCanvas.getContext('2d');
    const width = staffCanvas.width / (window.devicePixelRatio || 1);
    const height = staffCanvas.height / (window.devicePixelRatio || 1);
    const dpr = window.devicePixelRatio || 1;

    ctx.clearRect(0, 0, width, height);

    if (!notes.length) {
        ctx.fillStyle = '#888';
        ctx.font = '16px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Ноты появятся здесь после обработки', width/2, height/2);
        return;
    }

    const marginLeft = 50;
    const marginRight = 20;
    const staffTop = 60;
    const staffGap = 120;
    const lineSpacing = 8;

    // Time scaling
    const maxTime = Math.max(...notes.map(n => n.end), currentDuration || 1);
    const pixelsPerSecond = (width - marginLeft - marginRight) / maxTime * staffZoom;

    // Draw time grid
    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = 0.5;
    for (let t = 0; t <= maxTime; t += 1) {
        const x = marginLeft + t * pixelsPerSecond + staffOffsetX;
        if (x < marginLeft || x > width - marginRight) continue;
        ctx.beginPath();
        ctx.moveTo(x, 20);
        ctx.lineTo(x, height - 20);
        ctx.stroke();
        ctx.fillStyle = '#999';
        ctx.font = '10px sans-serif';
        ctx.fillText(t + 's', x + 2, 15);
    }

    // Draw staves (5 lines each)
    function drawFiveLines(yCenter) {
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 1;
        for (let i = -2; i <= 2; i++) {
            const y = yCenter + i * lineSpacing;
            ctx.beginPath();
            ctx.moveTo(marginLeft, y);
            ctx.lineTo(width - marginRight, y);
            ctx.stroke();
        }
        // Clef symbols
        ctx.font = '24px serif';
        ctx.fillStyle = '#333';
        ctx.fillText('𝄞', marginLeft - 35, yCenter + 8);
    }

    // Treble staff (RH)
    drawFiveLines(staffTop + 2 * lineSpacing);
    // Bass staff (LH)  
    drawFiveLines(staffTop + staffGap + 2 * lineSpacing);
    ctx.fillText('𝄢', marginLeft - 35, staffTop + staffGap + 2 * lineSpacing + 8);

    // Draw notes
    notes.forEach(note => {
        const x = marginLeft + note.start * pixelsPerSecond + staffOffsetX;
        const w = Math.max(3, (note.end - note.start) * pixelsPerSecond);

        if (x + w < marginLeft || x > width - marginRight) return;

        const isRH = note.hand === 'RH';
        const staffY = isRH ? staffTop : staffTop + staffGap;
        const color = isRH ? 'rgba(220, 50, 50, 0.85)' : 'rgba(50, 80, 220, 0.85)';
        const borderColor = isRH ? '#a02020' : '#2030a0';

        // MIDI to staff position (simplified)
        // Treble: middle C (60) = line below staff
        // Bass: middle C (60) = line above staff
        let midi = note.pitch_midi;
        let yPos;

        if (isRH) {
            // Treble clef: E4 (64) = bottom line, F5 (77) = top line
            // Each semitone = lineSpacing/2
            const refMidi = 64; // E4, bottom line
            const refY = staffTop + 4 * lineSpacing; // bottom line
            yPos = refY - (midi - refMidi) * (lineSpacing / 2);
        } else {
            // Bass clef: G2 (43) = bottom line, A3 (57) = top line
            const refMidi = 43; // G2, bottom line
            const refY = staffTop + staffGap + 4 * lineSpacing;
            yPos = refY - (midi - refMidi) * (lineSpacing / 2);
        }

        // Draw note body
        ctx.fillStyle = color;
        ctx.strokeStyle = borderColor;
        ctx.lineWidth = 1;

        const noteHeight = lineSpacing * 1.2;
        const noteWidth = Math.max(8, w);

        ctx.beginPath();
        ctx.roundRect(x, yPos - noteHeight/2, noteWidth, noteHeight, 3);
        ctx.fill();
        ctx.stroke();

        // Note name
        if (w > 20) {
            ctx.fillStyle = '#fff';
            ctx.font = 'bold 9px sans-serif';
            ctx.textAlign = 'center';
            const name = midiToNoteName(midi);
            ctx.fillText(name, x + noteWidth/2, yPos + 3);
        }

        // Ledger lines
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 0.5;
        const staffBottom = staffY + 4 * lineSpacing;
        const staffTopLine = staffY - 4 * lineSpacing;

        if (yPos > staffBottom) {
            for (let ly = staffBottom + lineSpacing; ly <= yPos + noteHeight/2; ly += lineSpacing) {
                ctx.beginPath();
                ctx.moveTo(x - 3, ly);
                ctx.lineTo(x + noteWidth + 3, ly);
                ctx.stroke();
            }
        }
        if (yPos < staffTopLine) {
            for (let ly = staffTopLine - lineSpacing; ly >= yPos - noteHeight/2; ly -= lineSpacing) {
                ctx.beginPath();
                ctx.moveTo(x - 3, ly);
                ctx.lineTo(x + noteWidth + 3, ly);
                ctx.stroke();
            }
        }
    });

    // Update playhead position
    updatePlayhead();
}

function midiToNoteName(midi) {
    const names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
    return names[midi % 12] + (Math.floor(midi/12)-1);
}

function updatePlayhead() {
    const width = staffCanvas.width / (window.devicePixelRatio || 1);
    const marginLeft = 50;
    const marginRight = 20;
    const maxTime = Math.max(...notes.map(n => n.end), currentDuration || 1);
    const pixelsPerSecond = (width - marginLeft - marginRight) / maxTime * staffZoom;

    const x = marginLeft + playOffset * pixelsPerSecond + staffOffsetX;
    playhead.style.left = x + 'px';

    currentTimeEl.textContent = formatTime(playOffset);
}

function formatTime(s) {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, '0')}`;
}

// ─────────────────────────────────────────────
// Web Audio Synthesizer Playback
// ─────────────────────────────────────────────
function getAudioContext() {
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    return audioContext;
}

function togglePlay() {
    if (isPlaying) {
        pausePlayback();
    } else {
        startPlayback();
    }
}

function startPlayback() {
    if (!notes.length) return;
    const ctx = getAudioContext();
    if (ctx.state === 'suspended') ctx.resume();

    isPlaying = true;
    playStartTime = ctx.currentTime - playOffset;
    document.getElementById('playBtn').textContent = '⏸ Пауза';

    // Schedule all notes
    notes.forEach(note => {
        if (note.start < playOffset) return; // Skip notes before current position
        const t = note.start - playOffset;
        scheduleNote(ctx, note, t);
    });

    animatePlayhead();
}

function scheduleNote(ctx, note, when) {
    const freq = 440 * Math.pow(2, (note.pitch_midi - 69) / 12);
    const dur = note.end - note.start;

    // Oscillator
    const osc = ctx.createOscillator();
    osc.type = 'triangle';
    osc.frequency.value = freq;

    // Gain envelope (ADSR)
    const gain = ctx.createGain();
    const panner = ctx.createStereoPanner();
    panner.pan.value = note.hand === 'RH' ? 0.6 : -0.6;

    const now = ctx.currentTime + when;
    gain.gain.setValueAtTime(0, now);
    gain.gain.linearRampToValueAtTime(0.3, now + 0.01);   // Attack
    gain.gain.linearRampToValueAtTime(0.2, now + 0.1);    // Decay
    gain.gain.linearRampToValueAtTime(0.15, now + dur - 0.05); // Sustain
    gain.gain.linearRampToValueAtTime(0, now + dur);      // Release

    osc.connect(gain);
    gain.connect(panner);
    panner.connect(ctx.destination);

    osc.start(now);
    osc.stop(now + dur + 0.1);
}

function pausePlayback() {
    isPlaying = false;
    document.getElementById('playBtn').textContent = '▶️ Играть';
    if (animationFrame) cancelAnimationFrame(animationFrame);
    playOffset = audioContext.currentTime - playStartTime;
}

function stopPlayback() {
    isPlaying = false;
    playOffset = 0;
    document.getElementById('playBtn').textContent = '▶️ Играть';
    if (animationFrame) cancelAnimationFrame(animationFrame);
    updatePlayhead();
    seekBar.value = 0;
}

function animatePlayhead() {
    if (!isPlaying) return;
    const ctx = getAudioContext();
    playOffset = ctx.currentTime - playStartTime;

    if (playOffset >= currentDuration) {
        stopPlayback();
        return;
    }

    updatePlayhead();
    seekBar.value = (playOffset / currentDuration) * 100;

    animationFrame = requestAnimationFrame(animatePlayhead);
}
