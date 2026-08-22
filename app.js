/* PianoMagic Frontend — v7.11.0 */

const FE_VERSION = '7.11.0';
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
            setRunLog(data);
            showResults(data.result);
        } else if (data.status === 'error') {
            clearInterval(pollInterval);
            // A failed run is exactly when the log matters most, so make
            // it available here too rather than only on success.
            setRunLog(data);
            const lb = document.getElementById('logBox');
            lb.classList.remove('hidden');
            lb.open = true;
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

    notes = result.notes || [];
    currentDuration = result.duration || 0;
    totalTimeEl.textContent = formatTime(currentDuration);
    seekBar.value = 0;

    const wavUrl = result.wav_url ? `${API_BASE}${result.wav_url}` : '#';
    const xmlUrl = result.xml_url ? `${API_BASE}${result.xml_url}` : '#';
    document.getElementById('downloadWav').href = wavUrl;
    document.getElementById('downloadXml').href = xmlUrl;

    // The PDF is the readable score, so it leads - but only when the
    // engraver actually produced one. A dead download button is worse
    // than no button, and pdf_error says in the log why it is missing.
    const pdfLink = document.getElementById('downloadPdf');
    if (result.pdf_url) {
        pdfLink.href = `${API_BASE}${result.pdf_url}`;
        pdfLink.classList.remove('hidden');
    } else {
        pdfLink.classList.add('hidden');
        if (result.pdf_error) console.warn('[PianoMagic] PDF not produced:', result.pdf_error);
    }

    // Say plainly which engine produced this result. The neural engine
    // can import fine and still fail per request, in which case the notes
    // silently come from the old monophonic fallback - that happened in
    // v7.6.0 and was invisible in the UI. It should never be invisible.
    const banner = document.getElementById('engineBanner');
    if (result.engine === 'basic-pitch') {
        banner.textContent = '🧠 Движок: Basic Pitch (нейросетевой, полифонический)';
        banner.className = 'engine-banner engine-ok';
    } else {
        banner.textContent = '⚠️ Движок: librosa PYIN (монофонический запасной вариант)'
            + (result.engine_error ? ' — нейросетевой движок упал, подробности в логе' : '');
        banner.className = 'engine-banner engine-warn';
    }

    audioPlayer.src = wavUrl;
    audioPlayer.load();

    resizeStaff();
    autoZoom();
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

// ─────────────────────────────────────────────
// Run log (v7.6.1)
// ─────────────────────────────────────────────
// The pipeline reports every stage it runs, but until now that output
// existed only in the server console. Surfacing it here is what makes a
// silent engine fallback or a runtime failure diagnosable from a normal
// browser session instead of requiring deploy-log access.
let currentRunLog = '';

function setRunLog(data) {
    const lines = Array.isArray(data.log) ? data.log.slice() : [];
    if (data.traceback) {
        lines.push('', '--- TRACEBACK ---', data.traceback);
    }
    currentRunLog = lines.join('\n');
    const logText = document.getElementById('logText');
    const logBox = document.getElementById('logBox');
    if (logText) logText.textContent = currentRunLog || '(лог пуст)';
    if (logBox && currentRunLog) logBox.classList.remove('hidden');
    // Reveal the diagnostics block itself - it starts hidden so it doesn't
    // clutter the page before a run has produced anything to show.
    const diag = document.getElementById('diagnosticsSection');
    if (diag) diag.classList.remove('hidden');
}

async function copyRunLog() {
    const btn = document.getElementById('copyLog');
    let text = currentRunLog;

    // Fall back to fetching the log directly if polling didn't capture it
    // (e.g. the page was reloaded mid-run).
    if (!text && currentTaskId) {
        try {
            const res = await fetch(`${API_BASE}/logs/${currentTaskId}`);
            if (res.ok) text = await res.text();
        } catch (e) { /* fall through to the empty-log message below */ }
    }
    if (!text) {
        btn.textContent = '❌ Лог пуст';
        setTimeout(() => { btn.textContent = '📋 Копировать лог'; }, 2000);
        return;
    }

    let ok = false;
    try {
        // navigator.clipboard needs a secure context and can be blocked by
        // permissions policy, so treat it as best-effort.
        await navigator.clipboard.writeText(text);
        ok = true;
    } catch (e) {
        // Legacy path: works without clipboard permissions.
        try {
            const ta = document.createElement('textarea');
            ta.value = text;
            ta.style.position = 'fixed';
            ta.style.opacity = '0';
            document.body.appendChild(ta);
            ta.select();
            ok = document.execCommand('copy');
            document.body.removeChild(ta);
        } catch (e2) { ok = false; }
    }

    if (ok) {
        btn.textContent = '✅ Скопировано';
    } else {
        // Last resort: reveal the log so it can be selected manually.
        document.getElementById('logBox').classList.remove('hidden');
        document.getElementById('logBox').open = true;
        btn.textContent = '⚠️ Скопируйте вручную';
    }
    setTimeout(() => { btn.textContent = '📋 Копировать лог'; }, 2000);
}

document.addEventListener('DOMContentLoaded', () => {
    const btn = document.getElementById('copyLog');
    if (btn) btn.addEventListener('click', copyRunLog);
});
