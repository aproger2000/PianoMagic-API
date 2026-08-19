/* PianoMagic Frontend — v7.2.1 */

const FE_VERSION = '7.6.0';
const API_BASE = 'https://pianomagic-api.onrender.com';

// State
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
let scheduledNodes = [];

// DOM Elements
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
        if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleFile(e.target.files[0]);
    });

    document.getElementById('playBtn').addEventListener('click', togglePlay);
    document.getElementById('stopBtn').addEventListener('click', stopPlayback);
    document.getElementById('zoomInBtn').addEventListener('click', () => setZoom(staffZoom * 1.3));
    document.getElementById('zoomOutBtn').addEventListener('click', () => setZoom(staffZoom / 1.3));

    seekBar.addEventListener('input', (e) => {
        playOffset = (e.target.value / 100) * currentDuration;
        updatePlayhead();
        if (isPlaying) {
            stopAllSounds();
            playStartTime = audioContext.currentTime - playOffset;
            scheduleAllNotes();
        }
    });

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
        clampOffset();
        drawStaff();
    });
    window.addEventListener('mouseup', () => {
        isDraggingStaff = false;
        staffCanvas.style.cursor = 'grab';
    });

    staffCanvas.addEventListener('touchstart', (e) => {
        isDraggingStaff = true;
        lastMouseX = e.touches[0].clientX;
    }, {passive: false});
    window.addEventListener('touchmove', (e) => {
        if (!isDraggingStaff) return;
        e.preventDefault();
        const dx = e.touches[0].clientX - lastMouseX;
        staffOffsetX += dx;
        lastMouseX = e.touches[0].clientX;
        clampOffset();
        drawStaff();
    }, {passive: false});
    window.addEventListener('touchend', () => {
        isDraggingStaff = false;
    });

    window.addEventListener('resize', () => {
        resizeStaff();
        autoZoom();
    });
}

function clampOffset() {
    const wrapper = document.querySelector('.staff-wrapper');
    const width = wrapper.clientWidth;
    const marginLeft = 50;
    const marginRight = 20;
    const maxTime = Math.max(...notes.map(n => n.end), currentDuration || 1);
    const contentWidth = marginLeft + maxTime * getPixelsPerSecond() + marginRight;
    const minOffset = Math.min(0, width - contentWidth);
    staffOffsetX = Math.max(minOffset, Math.min(0, staffOffsetX));
}

function getPixelsPerSecond() {
    const wrapper = document.querySelector('.staff-wrapper');
    const width = wrapper.clientWidth;
    const maxTime = Math.max(...notes.map(n => n.end), currentDuration || 1);
    return (width - 70) / maxTime * staffZoom;
}

function autoZoom() {
    if (!notes.length || !currentDuration) return;
    const wrapper = document.querySelector('.staff-wrapper');
    const width = wrapper.clientWidth;
    const targetPps = width / 30;
    const basePps = (width - 70) / currentDuration;
    staffZoom = targetPps / basePps;
    staffZoom = Math.max(0.5, Math.min(20, staffZoom));
    staffOffsetX = 0;
    drawStaff();
}

// ─────────────────────────────────────────────
// API
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
// Upload
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
// Polling
// ─────────────────────────────────────────────
function startPolling(taskId) {
    if (pollInterval) clearInterval(pollInterval);
    pollInterval = setInterval(() => pollStatus(taskId), 1500);
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
        const doneStages = ['synthesizing','generating_score','completed'];
        const currentDone = doneStages.includes(status);
        const elDone = ['loading','analyzing'].includes(el.dataset.stage) && currentDone;
        el.classList.toggle('done', elDone || status === 'completed');
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
// Results
// ─────────────────────────────────────────────
function showResults(result) {
    progressSection.classList.add('hidden');
    resultsSection.classList.remove('hidden');

    document.getElementById('noteCount').textContent = result.notes_count || '—';
    document.getElementById('rhCount').textContent = result.rh_notes || '—';
    document.getElementById('lhCount').textContent = result.lh_notes || '—';
    document.getElementById('tempoValue').textContent = result.tempo ? `${result.tempo} BPM` : '—';
    document.getElementById('keyValue').textContent = result.key || '—';

    // Comparison metrics
    const comp = result.comparison || {};
    document.getElementById('chromaCorr').textContent = comp.chroma_correlation !== undefined ? comp.chroma_correlation.toFixed(3) : '—';
    document.getElementById('spectralCorr').textContent = comp.spectral_contrast_correlation !== undefined ? comp.spectral_contrast_correlation.toFixed(3) : '—';
    document.getElementById('onsetCorr').textContent = comp.onset_correlation !== undefined ? comp.onset_correlation.toFixed(3) : '—';
    document.getElementById('overallCorr').textContent = comp.overall_similarity !== undefined ? comp.overall_similarity.toFixed(3) : '—';

    // Chroma charts
    if (comp.chroma_orig && comp.chroma_synth) {
        drawChromaChart('chromaOrigCanvas', comp.chroma_orig, '#4a90d9');
        drawChromaChart('chromaSynthCanvas', comp.chroma_synth, '#e74c3c');
    }

    notes = result.notes || [];
    currentDuration = result.duration || 0;
    totalTimeEl.textContent = formatTime(currentDuration);
    seekBar.value = 0;

    const wavUrl = result.wav_url ? `${API_BASE}${result.wav_url}` : '#';
    const xmlUrl = result.xml_url ? `${API_BASE}${result.xml_url}` : '#';
    document.getElementById('downloadWav').href = wavUrl;
    document.getElementById('downloadXml').href = xmlUrl;

    audioPlayer.src = wavUrl;
    audioPlayer.load();

    resizeStaff();
    autoZoom();
}

function drawChromaChart(canvasId, data, color) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !data || data.length !== 12) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.width / dpr;
    const h = canvas.height / dpr;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.scale(dpr, dpr);

    const labels = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
    const maxVal = Math.max(...data, 0.001);
    const barW = (w - 40) / 12;
    const barMaxH = h - 30;

    data.forEach((val, i) => {
        const barH = (val / maxVal) * barMaxH;
        const x = 20 + i * barW;
        const y = h - 20 - barH;
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.8;
        ctx.fillRect(x + 2, y, barW - 4, barH);
        ctx.globalAlpha = 1;
        ctx.fillStyle = '#555';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(labels[i], x + barW/2, h - 5);
    });
}

// ─────────────────────────────────────────────
// Staff Canvas
// ─────────────────────────────────────────────
function resizeStaff() {
    const wrapper = document.querySelector('.staff-wrapper');
    const dpr = window.devicePixelRatio || 1;
    staffCanvas.width = wrapper.clientWidth * dpr;
    staffCanvas.height = 420 * dpr;
    staffCanvas.style.width = wrapper.clientWidth + 'px';
    staffCanvas.style.height = '420px';
    const ctx = staffCanvas.getContext('2d');
    ctx.scale(dpr, dpr);
    drawStaff();
}

function setZoom(z) {
    staffZoom = Math.max(0.1, Math.min(50, z));
    clampOffset();
    drawStaff();
}

function drawStaff() {
    const ctx = staffCanvas.getContext('2d');
    const width = staffCanvas.width / (window.devicePixelRatio || 1);
    const height = staffCanvas.height / (window.devicePixelRatio || 1);

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
    const staffTop = 70;
    const staffGap = 130;
    const lineSpacing = 9;

    const maxTime = Math.max(...notes.map(n => n.end), currentDuration || 1);
    const pps = getPixelsPerSecond();

    ctx.fillStyle = '#fffef8';
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = '#e8e4d8';
    ctx.lineWidth = 0.5;
    for (let t = 0; t <= maxTime + 1; t += 1) {
        const x = marginLeft + t * pps + staffOffsetX;
        if (x < marginLeft - 5 || x > width) continue;
        ctx.beginPath();
        ctx.moveTo(x, 10);
        ctx.lineTo(x, height - 10);
        ctx.stroke();
        ctx.fillStyle = '#aaa';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(t + 's', x, 24);
    }

    const beatDuration = 60.0 / (parseFloat(document.getElementById('tempoValue').textContent) || 120);
    ctx.strokeStyle = '#f0ece0';
    ctx.lineWidth = 0.3;
    for (let t = 0; t <= maxTime; t += beatDuration) {
        const x = marginLeft + t * pps + staffOffsetX;
        if (x < marginLeft || x > width) continue;
        ctx.beginPath();
        ctx.moveTo(x, staffTop - 30);
        ctx.lineTo(x, staffTop + staffGap + 50);
        ctx.stroke();
    }

    function drawFiveLines(yCenter, label) {
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 1;
        for (let i = -2; i <= 2; i++) {
            const y = yCenter + i * lineSpacing;
            ctx.beginPath();
            ctx.moveTo(marginLeft, y);
            ctx.lineTo(width - marginRight, y);
            ctx.stroke();
        }
        ctx.font = '22px serif';
        ctx.fillStyle = '#333';
        ctx.fillText(label, marginLeft - 38, yCenter + 8);
    }

    drawFiveLines(staffTop + 2 * lineSpacing, '𝄞');
    drawFiveLines(staffTop + staffGap + 2 * lineSpacing, '𝄢');

    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(marginLeft - 8, staffTop + 2 * lineSpacing - 15);
    ctx.quadraticCurveTo(marginLeft - 18, staffTop + staffGap/2 + 2 * lineSpacing, marginLeft - 8, staffTop + staffGap + 2 * lineSpacing + 15);
    ctx.stroke();

    notes.forEach(note => {
        const x = marginLeft + note.start * pps + staffOffsetX;
        const w = Math.max(4, (note.end - note.start) * pps);

        if (x + w < 0 || x > width) return;

        const isRH = note.hand === 'RH';
        const staffY = isRH ? staffTop : staffTop + staffGap;
        const color = isRH ? 'rgba(200, 60, 60, 0.8)' : 'rgba(50, 90, 200, 0.8)';
        const borderColor = isRH ? '#a03030' : '#2030a0';

        let midi = note.pitch_midi;
        let yPos;

        if (isRH) {
            const refMidi = 64;
            const refY = staffTop + 4 * lineSpacing;
            yPos = refY - (midi - refMidi) * (lineSpacing / 2);
        } else {
            const refMidi = 43;
            const refY = staffTop + staffGap + 4 * lineSpacing;
            yPos = refY - (midi - refMidi) * (lineSpacing / 2);
        }

        const noteHeight = lineSpacing * 1.3;
        const noteWidth = Math.max(6, w);

        ctx.fillStyle = color;
        ctx.strokeStyle = borderColor;
        ctx.lineWidth = 1;
        roundRect(ctx, x, yPos - noteHeight/2, noteWidth, noteHeight, 3);
        ctx.fill();
        ctx.stroke();

        if (w > 18) {
            ctx.fillStyle = '#fff';
            ctx.font = 'bold 8px sans-serif';
            ctx.textAlign = 'center';
            const name = midiToNoteName(midi);
            ctx.fillText(name, x + noteWidth/2, yPos + 3);
        }

        ctx.strokeStyle = '#555';
        ctx.lineWidth = 0.8;
        const staffBottom = staffY + 4 * lineSpacing;
        const staffTopLine = staffY - 4 * lineSpacing;

        if (yPos + noteHeight/2 > staffBottom + 2) {
            for (let ly = staffBottom + lineSpacing; ly <= yPos + noteHeight/2 + 2; ly += lineSpacing) {
                ctx.beginPath();
                ctx.moveTo(x - 4, ly);
                ctx.lineTo(x + noteWidth + 4, ly);
                ctx.stroke();
            }
        }
        if (yPos - noteHeight/2 < staffTopLine - 2) {
            for (let ly = staffTopLine - lineSpacing; ly >= yPos - noteHeight/2 - 2; ly -= lineSpacing) {
                ctx.beginPath();
                ctx.moveTo(x - 4, ly);
                ctx.lineTo(x + noteWidth + 4, ly);
                ctx.stroke();
            }
        }
    });

    updatePlayhead();
}

function roundRect(ctx, x, y, w, h, r) {
    if (w < 2 * r) r = w / 2;
    if (h < 2 * r) r = h / 2;
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
}

function midiToNoteName(midi) {
    const names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
    return names[midi % 12] + (Math.floor(midi/12)-1);
}

function updatePlayhead() {
    const wrapper = document.querySelector('.staff-wrapper');
    const width = wrapper.clientWidth;
    const marginLeft = 50;
    const pps = getPixelsPerSecond();

    const x = marginLeft + playOffset * pps + staffOffsetX;
    playhead.style.left = x + 'px';
    currentTimeEl.textContent = formatTime(playOffset);
}

function formatTime(s) {
    if (!isFinite(s) || s < 0) s = 0;
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, '0')}`;
}

// ─────────────────────────────────────────────
// Web Audio Playback
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

function stopAllSounds() {
    scheduledNodes.forEach(n => {
        try { n.stop(); } catch(e) {}
        try { n.disconnect(); } catch(e) {}
    });
    scheduledNodes = [];
}

function startPlayback() {
    if (!notes.length) return;
    const ctx = getAudioContext();
    if (ctx.state === 'suspended') ctx.resume();

    isPlaying = true;
    playStartTime = ctx.currentTime - playOffset;
    document.getElementById('playBtn').textContent = '⏸ Пауза';

    scheduleAllNotes();
    animatePlayhead();
}

function scheduleAllNotes() {
    const ctx = getAudioContext();
    notes.forEach(note => {
        if (note.start < playOffset) return;
        const when = note.start - playOffset;
        scheduleNote(ctx, note, when);
    });
}

function scheduleNote(ctx, note, when) {
    const freq = 440 * Math.pow(2, (note.pitch_midi - 69) / 12);
    const dur = note.end - note.start;
    if (dur <= 0) return;

    const osc = ctx.createOscillator();
    osc.type = 'triangle';
    osc.frequency.value = freq;

    const gain = ctx.createGain();
    const panner = ctx.createStereoPanner();
    panner.pan.value = note.hand === 'RH' ? 0.6 : -0.6;

    const now = ctx.currentTime + when;
    gain.gain.setValueAtTime(0, now);
    gain.gain.linearRampToValueAtTime(0.25, now + 0.01);
    gain.gain.linearRampToValueAtTime(0.15, now + 0.1);
    gain.gain.linearRampToValueAtTime(0.1, now + dur - 0.05);
    gain.gain.linearRampToValueAtTime(0, now + dur);

    osc.connect(gain);
    gain.connect(panner);
    panner.connect(ctx.destination);

    osc.start(now);
    osc.stop(now + dur + 0.05);
    scheduledNodes.push(osc);
}

function pausePlayback() {
    isPlaying = false;
    stopAllSounds();
    document.getElementById('playBtn').textContent = '▶️ Играть';
    if (animationFrame) cancelAnimationFrame(animationFrame);
    playOffset = audioContext.currentTime - playStartTime;
}

function stopPlayback() {
    isPlaying = false;
    stopAllSounds();
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
    seekBar.value = Math.min(100, (playOffset / currentDuration) * 100);

    animationFrame = requestAnimationFrame(animatePlayhead);
}
