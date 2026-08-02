/**
 * PianoMagic 7.1 — Frontend with Web Audio Synthesizer
 */

const API_BASE = 'https://pianomagic-api.onrender.com'; // <-- your backend
const FE_VERSION = '7.1';

// DOM
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const progressSection = document.getElementById('progressSection');
const progressBar = document.getElementById('progressBar');
const stageText = document.getElementById('stageText');
const errorBox = document.getElementById('errorBox');
const errorText = document.getElementById('errorText');
const results = document.getElementById('results');
const similarityValue = document.getElementById('similarityValue');
const similarityBar = document.getElementById('similarityBar');
const similarityLabel = document.getElementById('similarityLabel');
const specCanvas = document.getElementById('specCanvas');
const staffCanvas = document.getElementById('staffCanvas');
const chromaOrigCanvas = document.getElementById('chromaOrigCanvas');
const chromaSynthCanvas = document.getElementById('chromaSynthCanvas');
const pearsonCorr = document.getElementById('pearsonCorr');
const profileSim = document.getElementById('profileSim');
const playStaffBtn = document.getElementById('playStaffBtn');
const stopStaffBtn = document.getElementById('stopStaffBtn');
const staffTime = document.getElementById('staffTime');
const beVersion = document.getElementById('beVersion');
const apiStatus = document.getElementById('apiStatus');
const keyValue = document.getElementById('keyValue');
const tempoValue = document.getElementById('tempoValue');
const durationValue = document.getElementById('durationValue');

// State
let currentJobId = null;
let pollInterval = null;
let staffNotes = { rh: [], lh: [] };
let staffDuration = 0;
let staffPlaying = false;
let staffAnimFrame = null;
let staffStartTime = 0;

// ============================================================
// Web Audio API — Browser Synthesizer
// ============================================================
let audioCtx = null;
let activeOscillators = [];

function ensureAudioContext() {
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.state === 'suspended') {
        audioCtx.resume();
    }
}

function midiToFreq(midi) {
    return 440 * Math.pow(2, (midi - 69) / 12);
}

function stopAllSounds() {
    const now = audioCtx ? audioCtx.currentTime : 0;
    activeOscillators.forEach(o => {
        try { o.stop(now); } catch(e){}
    });
    activeOscillators = [];
}

function playTone(freq, startTime, duration, velocity = 0.5, pan = 0) {
    if (!audioCtx) return;
    const t = audioCtx.currentTime + startTime;
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    const panner = audioCtx.createStereoPanner();

    // Triangle wave + harmonics via waveshaper for richer tone
    osc.type = 'triangle';
    osc.frequency.value = freq;

    panner.pan.value = pan;

    osc.connect(gain);
    gain.connect(panner);
    panner.connect(audioCtx.destination);

    const attack = 0.008;
    const decay = 0.12;
    const sustain = velocity * 0.22;
    const release = Math.min(0.25, duration * 0.35);
    const total = duration + 0.05;

    gain.gain.setValueAtTime(0, t);
    gain.gain.linearRampToValueAtTime(sustain, t + attack);
    gain.gain.exponentialRampToValueAtTime(sustain * 0.65, t + attack + decay);
    gain.gain.setValueAtTime(sustain * 0.65, t + duration - release);
    gain.gain.exponentialRampToValueAtTime(0.001, t + duration);

    osc.start(t);
    osc.stop(t + total);
    activeOscillators.push(osc);

    // Auto-cleanup
    setTimeout(() => {
        const idx = activeOscillators.indexOf(osc);
        if (idx > -1) activeOscillators.splice(idx, 1);
    }, (startTime + total) * 1000 + 200);
}

function playMelody(notes, totalDuration) {
    ensureAudioContext();
    stopAllSounds();

    // Schedule RH notes (panned right)
    notes.rh.forEach(n => {
        playTone(midiToFreq(n.pitch), n.start, n.dur, (n.velocity || 80) / 127, 0.4);
    });
    // Schedule LH notes (panned left)
    notes.lh.forEach(n => {
        playTone(midiToFreq(n.pitch), n.start, n.dur, (n.velocity || 60) / 127, -0.4);
    });

    // Sync animation
    staffPlaying = true;
    staffStartTime = performance.now();
    playStaffBtn.style.display = 'none';
    stopStaffBtn.style.display = 'inline-block';
    animateStaff();

    // Auto-stop
    setTimeout(() => {
        if (staffPlaying) stopPlayback();
    }, totalDuration * 1000 + 500);
}

function stopPlayback() {
    staffPlaying = false;
    if (staffAnimFrame) cancelAnimationFrame(staffAnimFrame);
    stopAllSounds();
    playStaffBtn.style.display = 'inline-block';
    stopStaffBtn.style.display = 'none';
    drawStaff(staffNotes, staffDuration, 0);
    staffTime.textContent = `0.0s / ${staffDuration.toFixed(1)}s`;
}

playStaffBtn.addEventListener('click', () => {
    if (!staffDuration) return;
    playMelody(staffNotes, staffDuration);
});
stopStaffBtn.addEventListener('click', stopPlayback);

// ============================================================
// API Status
// ============================================================
async function checkApiStatus() {
    try {
        const ctrl = new AbortController();
        const to = setTimeout(() => ctrl.abort(), 8000);
        const res = await fetch(API_BASE + '/', { signal: ctrl.signal });
        clearTimeout(to);
        const data = await res.json();
        beVersion.textContent = data.version || '?';
        apiStatus.textContent = data.status === 'ok' ? 'API: online' : 'API: ошибка';
        apiStatus.className = 'api-status ' + (data.status === 'ok' ? 'online' : 'offline');
    } catch (e) {
        beVersion.textContent = '—';
        apiStatus.textContent = 'API: offline';
        apiStatus.className = 'api-status offline';
    }
}
checkApiStatus();
setInterval(checkApiStatus, 30000);

// ============================================================
// Upload
// ============================================================
function handleFile(file) {
    if (!file) return;
    const validExts = ['.mp3','.wav','.flac','.m4a','.ogg','.aac'];
    const ext = file.name.slice(file.name.lastIndexOf('.')).toLowerCase();
    const isValid = validExts.includes(ext) || file.type.startsWith('audio/');
    if (!isValid) { showError('Поддерживаются MP3, WAV, FLAC, M4A, OGG'); return; }
    if (file.size > 50 * 1024 * 1024) { showError('Файл слишком большой (макс. 50 МБ)'); return; }
    uploadFile(file);
}

uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('dragover'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
uploadZone.addEventListener('drop', e => {
    e.preventDefault(); uploadZone.classList.remove('dragover');
    handleFile(e.dataTransfer.files[0]);
});
uploadBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', e => handleFile(e.target.files[0]));
uploadZone.addEventListener('click', e => {
    if (e.target !== uploadBtn && e.target !== fileInput) fileInput.click();
});

async function uploadFile(file) {
    resetUI();
    progressSection.style.display = 'block';
    setProgress(5, 'upload');
    const formData = new FormData();
    formData.append('file', file);
    try {
        const res = await fetch(API_BASE + '/analyze', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.job_id) { currentJobId = data.job_id; pollStatus(data.job_id); }
        else showError('Ошибка сервера при создании задачи');
    } catch (e) { showError('Не удалось подключиться к API: ' + e.message); }
}

// ============================================================
// Polling
// ============================================================
function pollStatus(jobId) {
    if (pollInterval) clearInterval(pollInterval);
    pollInterval = setInterval(async () => {
        try {
            const res = await fetch(API_BASE + '/status/' + jobId);
            const data = await res.json();
            setProgress(data.progress, data.stage, data.stage_text);
            if (data.status === 'completed' || data.status === 'failed') {
                clearInterval(pollInterval);
                if (data.status === 'completed') await fetchResults(jobId);
                else showError(data.error || 'Неизвестная ошибка');
            }
        } catch (e) { clearInterval(pollInterval); showError('Ошибка связи с сервером'); }
    }, 1500);
}

function setProgress(pct, stage, text) {
    progressBar.style.width = pct + '%';
    if (text) stageText.textContent = text;
    const stages = ['upload','analyze','melody','fit','compare','render','done'];
    const idx = stages.indexOf(stage);
    document.querySelectorAll('.stage').forEach(el => {
        const s = el.dataset.stage;
        const si = stages.indexOf(s);
        el.classList.remove('active','done');
        if (si < idx) el.classList.add('done');
        if (si === idx) el.classList.add('active');
    });
}

// ============================================================
// Results
// ============================================================
async function fetchResults(jobId) {
    try {
        const res = await fetch(API_BASE + '/melody/' + jobId);
        const data = await res.json();
        results.style.display = 'block';
        staffNotes = { rh: data.melody_rh || [], lh: data.melody_lh || [] };
        staffDuration = data.duration || 0;

        const sim = data.similarity || 0;
        similarityValue.textContent = (sim * 100).toFixed(1) + '%';
        similarityBar.style.width = (sim * 100) + '%';
        if (sim >= 0.75) { similarityLabel.textContent = 'Отличное совпадение'; similarityBar.className = 'similarity-bar excellent'; }
        else if (sim >= 0.55) { similarityLabel.textContent = 'Хорошее совпадение'; similarityBar.className = 'similarity-bar good'; }
        else if (sim >= 0.35) { similarityLabel.textContent = 'Удовлетворительно'; similarityBar.className = 'similarity-bar fair'; }
        else { similarityLabel.textContent = 'Требуется доработка'; similarityBar.className = 'similarity-bar poor'; }

        keyValue.textContent = data.key_name || 'C major';
        tempoValue.textContent = data.tempo ? Math.round(data.tempo) + ' BPM' : '—';
        durationValue.textContent = data.duration ? data.duration.toFixed(1) + ' с' : '—';

        if (data.spec) drawSpectrogram(data.spec);
        if (data.chroma_orig && data.chroma_synth) {
            drawChroma(data.chroma_orig, chromaOrigCanvas);
            drawChroma(data.chroma_synth, chromaSynthCanvas);
            pearsonCorr.textContent = pearsonCorrelation(data.chroma_orig, data.chroma_synth).toFixed(3);
            profileSim.textContent = (sim * 100).toFixed(1) + '%';
        }

        drawStaff(staffNotes, staffDuration, 0);
        staffTime.textContent = `0.0s / ${staffDuration.toFixed(1)}s`;
        playStaffBtn.style.display = 'inline-block';
        stopStaffBtn.style.display = 'none';

        document.getElementById('downloadPdf').href = API_BASE + '/download/' + jobId + '.pdf';
        document.getElementById('downloadWav').href = API_BASE + '/download/' + jobId + '.wav';
    } catch (e) { showError('Ошибка загрузки результатов: ' + e.message); }
}

// ============================================================
// Visualizers
// ============================================================
function drawSpectrogram(specData) {
    const ctx = specCanvas.getContext('2d');
    const W = specCanvas.width, H = specCanvas.height;
    ctx.clearRect(0, 0, W, H);
    if (!specData || !specData.length) return;
    const rows = specData.length, cols = specData[0].length;
    const cw = Math.max(1, W / cols), ch = Math.max(1, H / rows);
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const v = specData[r][c] / 255;
            const hue = 240 - v * 240;
            ctx.fillStyle = `hsl(${hue}, 80%, ${20 + v * 60}%)`;
            ctx.fillRect(c * cw, H - (r + 1) * ch, cw + 0.5, ch + 0.5);
        }
    }
}

function drawChroma(values, canvas) {
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    const labels = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
    const maxV = Math.max(...values, 0.001);
    const bw = (W - 40) / 12;
    const colors = ['#e74c3c','#e67e22','#f1c40f','#2ecc71','#1abc9c','#3498db','#9b59b6','#e74c3c','#e67e22','#f1c40f','#2ecc71','#1abc9c'];
    for (let i = 0; i < 12; i++) {
        const h = (values[i] / maxV) * (H - 40);
        const x = 20 + i * bw, y = H - 25 - h;
        ctx.fillStyle = colors[i];
        ctx.fillRect(x, y, bw - 4, h);
        ctx.fillStyle = '#ccc';
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(labels[i], x + bw / 2 - 2, H - 5);
    }
}

function pearsonCorrelation(a, b) {
    const n = a.length;
    const sa = a.reduce((s, v) => s + v, 0);
    const sb = b.reduce((s, v) => s + v, 0);
    const sab = a.reduce((s, v, i) => s + v * b[i], 0);
    const sa2 = a.reduce((s, v) => s + v * v, 0);
    const sb2 = b.reduce((s, v) => s + v * v, 0);
    const num = n * sab - sa * sb;
    const den = Math.sqrt((n * sa2 - sa * sa) * (n * sb2 - sb * sb));
    return den === 0 ? 0 : num / den;
}

// ============================================================
// Staff Renderer
// ============================================================
const STAFF_TOP = 40;
const STAFF_GAP = 80;
const LINE_SPACING = 8;
const NOTE_RADIUS = 5;

function midiToStaffY(midi, clef) {
    if (clef === 'treble') {
        const middleC = STAFF_TOP + 4 * LINE_SPACING;
        return middleC - (midi - 60) * (LINE_SPACING / 2);
    } else {
        const middleC = STAFF_TOP + LINE_SPACING;
        return middleC - (midi - 60) * (LINE_SPACING / 2);
    }
}

function drawStaff(notes, duration, playheadTime) {
    const ctx = staffCanvas.getContext('2d');
    const W = staffCanvas.width, H = staffCanvas.height;
    ctx.clearRect(0, 0, W, H);
    const timeScale = duration > 0 ? (W - 60) / duration : 1;

    // Treble staff
    drawStaffLines(ctx, 30, STAFF_TOP, W - 30);
    ctx.fillStyle = '#555'; ctx.font = 'bold 20px serif';
    ctx.fillText('𝄞', 6, STAFF_TOP + 30);

    // Bass staff
    const bassTop = STAFF_TOP + STAFF_GAP;
    drawStaffLines(ctx, 30, bassTop, W - 30);
    ctx.fillStyle = '#555'; ctx.font = 'bold 18px serif';
    ctx.fillText('𝄢', 6, bassTop + 24);

    // Draw ledger lines helper
    function drawLedger(x, y, clefTop) {
        if (y < clefTop - 2) {
            for (let ly = clefTop - LINE_SPACING; ly >= y - 4; ly -= LINE_SPACING) {
                ctx.strokeStyle = '#555'; ctx.lineWidth = 1;
                ctx.beginPath(); ctx.moveTo(x - 8, ly); ctx.lineTo(x + 8, ly); ctx.stroke();
            }
        }
        if (y > clefTop + 4 * LINE_SPACING + 2) {
            for (let ly = clefTop + 5 * LINE_SPACING; ly <= y + 4; ly += LINE_SPACING) {
                ctx.strokeStyle = '#555'; ctx.lineWidth = 1;
                ctx.beginPath(); ctx.moveTo(x - 8, ly); ctx.lineTo(x + 8, ly); ctx.stroke();
            }
        }
    }

    // RH notes
    notes.rh.forEach(n => {
        const x = 30 + n.start * timeScale;
        const y = midiToStaffY(n.pitch, 'treble');
        const w = Math.max(5, n.dur * timeScale);
        drawLedger(x + w / 2, y, STAFF_TOP);
        ctx.fillStyle = '#ff6b6b';
        ctx.beginPath();
        ctx.ellipse(x + w / 2, y, w / 2, NOTE_RADIUS, 0, 0, Math.PI * 2);
        ctx.fill();
        // stem
        if (n.pitch >= 71) {
            ctx.strokeStyle = '#ff6b6b'; ctx.lineWidth = 1.5;
            ctx.beginPath(); ctx.moveTo(x + w / 2 + w / 2 - 1, y); ctx.lineTo(x + w / 2 + w / 2 - 1, y + 28); ctx.stroke();
        }
    });

    // LH notes
    notes.lh.forEach(n => {
        const x = 30 + n.start * timeScale;
        const y = midiToStaffY(n.pitch, 'bass');
        const w = Math.max(5, n.dur * timeScale);
        drawLedger(x + w / 2, y, bassTop);
        ctx.fillStyle = '#4ecdc4';
        ctx.beginPath();
        ctx.ellipse(x + w / 2, y, w / 2, NOTE_RADIUS, 0, 0, Math.PI * 2);
        ctx.fill();
    });

    // Playhead
    if (playheadTime >= 0 && playheadTime <= duration) {
        const px = 30 + playheadTime * timeScale;
        ctx.strokeStyle = '#2ecc71'; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(px, 8); ctx.lineTo(px, H - 8); ctx.stroke();
    }
}

function drawStaffLines(ctx, x1, y, x2) {
    ctx.strokeStyle = '#444'; ctx.lineWidth = 1;
    for (let i = 0; i < 5; i++) {
        const ly = y + i * LINE_SPACING;
        ctx.beginPath(); ctx.moveTo(x1, ly); ctx.lineTo(x2, ly); ctx.stroke();
    }
}

function animateStaff() {
    if (!staffPlaying) return;
    const elapsed = (performance.now() - staffStartTime) / 1000;
    if (elapsed >= staffDuration) { stopPlayback(); return; }
    drawStaff(staffNotes, staffDuration, elapsed);
    staffTime.textContent = `${elapsed.toFixed(1)}s / ${staffDuration.toFixed(1)}s`;
    staffAnimFrame = requestAnimationFrame(animateStaff);
}

// ============================================================
// UI Helpers
// ============================================================
function showError(msg) {
    errorBox.style.display = 'flex';
    errorText.textContent = msg;
    progressSection.style.display = 'none';
}

function resetUI() {
    errorBox.style.display = 'none';
    results.style.display = 'none';
    progressBar.style.width = '0%';
    stageText.textContent = 'Ожидание...';
    document.querySelectorAll('.stage').forEach(el => el.classList.remove('active', 'done'));
    if (pollInterval) clearInterval(pollInterval);
    stopPlayback();
}

document.addEventListener('touchmove', e => {
    if (e.target === document.body) e.preventDefault();
}, { passive: false });
