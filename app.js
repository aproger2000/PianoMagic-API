/**
 * PianoMagic 7.0 — Frontend Controller
 * Features: drag-drop, real-time staff, chroma comparison, spectrogram, iOS support
 */

const API_BASE = 'https://pianomagic-api.onrender.com';  // <-- Update to your backend URL
const FE_VERSION = '7.0';

// DOM Elements
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
// API Status Check
// ============================================================
async function checkApiStatus() {
    try {
        const ctrl = new AbortController();
        const timeout = setTimeout(() => ctrl.abort(), 8000);
        const res = await fetch(API_BASE + '/', { signal: ctrl.signal });
        clearTimeout(timeout);
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
// Upload Handling
// ============================================================
function handleFile(file) {
    if (!file) return;
    const validTypes = ['audio/mpeg','audio/wav','audio/flac','audio/mp4','audio/ogg','audio/x-wav','audio/aac'];
    const validExts = ['.mp3','.wav','.flac','.m4a','.ogg','.aac'];
    const ext = file.name.slice(file.name.lastIndexOf('.')).toLowerCase();
    const isValid = validTypes.includes(file.type) || validExts.includes(ext);
    if (!isValid) {
        showError('Поддерживаются только MP3, WAV, FLAC, M4A, OGG');
        return;
    }
    if (file.size > 50 * 1024 * 1024) {
        showError('Файл слишком большой (макс. 50 МБ)');
        return;
    }
    uploadFile(file);
}

uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
});
uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
});
uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    handleFile(file);
});

uploadBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));

// iOS: make the invisible input clickable across the whole zone
uploadZone.addEventListener('click', (e) => {
    if (e.target !== uploadBtn && e.target !== fileInput) {
        fileInput.click();
    }
});

async function uploadFile(file) {
    resetUI();
    progressSection.style.display = 'block';
    setProgress(5, 'upload');

    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch(API_BASE + '/analyze', {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        if (data.job_id) {
            currentJobId = data.job_id;
            pollStatus(data.job_id);
        } else {
            showError('Ошибка сервера при создании задачи');
        }
    } catch (e) {
        showError('Не удалось подключиться к API: ' + e.message);
    }
}

// ============================================================
// Status Polling
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
                if (data.status === 'completed') {
                    await fetchResults(jobId);
                } else {
                    showError(data.error || 'Неизвестная ошибка');
                }
            }
        } catch (e) {
            clearInterval(pollInterval);
            showError('Ошибка связи с сервером');
        }
    }, 1500);
}

function setProgress(percent, stage, text) {
    progressBar.style.width = percent + '%';
    if (text) stageText.textContent = text;

    document.querySelectorAll('.stage').forEach(el => {
        const s = el.dataset.stage;
        el.classList.remove('active', 'done');
        const stages = ['upload','analyze','melody','fit','compare','render','done'];
        const idx = stages.indexOf(stage);
        const elIdx = stages.indexOf(s);
        if (elIdx < idx) el.classList.add('done');
        if (elIdx === idx) el.classList.add('active');
    });
}

// ============================================================
// Fetch & Display Results
// ============================================================
async function fetchResults(jobId) {
    try {
        const res = await fetch(API_BASE + '/melody/' + jobId);
        const data = await res.json();

        results.style.display = 'block';
        staffNotes = { rh: data.melody_rh || [], lh: data.melody_lh || [] };
        staffDuration = data.duration || 0;

        // Similarity
        const sim = data.similarity || 0;
        similarityValue.textContent = (sim * 100).toFixed(1) + '%';
        similarityBar.style.width = (sim * 100) + '%';
        if (sim >= 0.75) {
            similarityLabel.textContent = 'Отличное совпадение';
            similarityBar.className = 'similarity-bar excellent';
        } else if (sim >= 0.55) {
            similarityLabel.textContent = 'Хорошее совпадение';
            similarityBar.className = 'similarity-bar good';
        } else if (sim >= 0.35) {
            similarityLabel.textContent = 'Удовлетворительно';
            similarityBar.className = 'similarity-bar fair';
        } else {
            similarityLabel.textContent = 'Требуется доработка';
            similarityBar.className = 'similarity-bar poor';
        }

        // Info cards
        keyValue.textContent = data.key_name || 'C major';
        tempoValue.textContent = data.tempo ? Math.round(data.tempo) + ' BPM' : '—';
        durationValue.textContent = data.duration ? data.duration.toFixed(1) + ' с' : '—';

        // Spectrogram
        if (data.spec) drawSpectrogram(data.spec);

        // Chroma
        if (data.chroma_orig && data.chroma_synth) {
            drawChroma(data.chroma_orig, chromaOrigCanvas);
            drawChroma(data.chroma_synth, chromaSynthCanvas);
            const corr = pearsonCorrelation(data.chroma_orig, data.chroma_synth);
            pearsonCorr.textContent = corr.toFixed(3);
            profileSim.textContent = (sim * 100).toFixed(1) + '%';
        }

        // Staff
        drawStaff(staffNotes, staffDuration, 0);
        staffTime.textContent = `0.0s / ${staffDuration.toFixed(1)}s`;

        // Downloads
        document.getElementById('downloadPdf').href = API_BASE + '/download/' + jobId + '.pdf';
        document.getElementById('downloadWav').href = API_BASE + '/download/' + jobId + '.wav';

    } catch (e) {
        showError('Ошибка загрузки результатов: ' + e.message);
    }
}

// ============================================================
// Spectrogram Renderer
// ============================================================
function drawSpectrogram(specData) {
    const ctx = specCanvas.getContext('2d');
    const W = specCanvas.width;
    const H = specCanvas.height;
    ctx.clearRect(0, 0, W, H);

    if (!specData || !specData.length) return;
    const rows = specData.length;
    const cols = specData[0].length;
    const cellW = Math.max(1, W / cols);
    const cellH = Math.max(1, H / rows);

    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const val = specData[r][c];
            const intensity = val / 255;
            const hue = 240 - intensity * 240; // blue to red
            ctx.fillStyle = `hsl(${hue}, 80%, ${20 + intensity * 60}%)`;
            ctx.fillRect(c * cellW, H - (r + 1) * cellH, cellW + 0.5, cellH + 0.5);
        }
    }
}

// ============================================================
// Chroma Histogram Renderer
// ============================================================
function drawChroma(values, canvas) {
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const labels = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
    const maxVal = Math.max(...values, 0.001);
    const barW = (W - 40) / 12;
    const colors = [
        '#e74c3c','#e67e22','#f1c40f','#2ecc71','#1abc9c','#3498db',
        '#9b59b6','#e74c3c','#e67e22','#f1c40f','#2ecc71','#1abc9c'
    ];

    for (let i = 0; i < 12; i++) {
        const h = (values[i] / maxVal) * (H - 40);
        const x = 20 + i * barW;
        const y = H - 25 - h;

        ctx.fillStyle = colors[i];
        ctx.fillRect(x, y, barW - 4, h);

        ctx.fillStyle = '#ccc';
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(labels[i], x + barW/2 - 2, H - 5);
    }
}

function pearsonCorrelation(a, b) {
    const n = a.length;
    const sumA = a.reduce((s, v) => s + v, 0);
    const sumB = b.reduce((s, v) => s + v, 0);
    const sumAB = a.reduce((s, v, i) => s + v * b[i], 0);
    const sumA2 = a.reduce((s, v) => s + v * v, 0);
    const sumB2 = b.reduce((s, v) => s + v * v, 0);
    const num = n * sumAB - sumA * sumB;
    const den = Math.sqrt((n * sumA2 - sumA * sumA) * (n * sumB2 - sumB * sumB));
    return den === 0 ? 0 : num / den;
}

// ============================================================
// Real-time Staff Renderer
// ============================================================
const STAFF_TOP = 40;
const STAFF_GAP = 70;
const LINE_SPACING = 8;
const NOTE_RADIUS = 5;

function midiToStaffY(midi, clef) {
    // Treble: middle C (60) = 5th line from bottom (y = STAFF_TOP + 4*LINE_SPACING)
    // Bass: middle C = first ledger line above (y = STAFF_TOP + LINE_SPACING)
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
    const W = staffCanvas.width;
    const H = staffCanvas.height;
    ctx.clearRect(0, 0, W, H);

    const timeScale = duration > 0 ? (W - 60) / duration : 1;

    // Draw treble staff
    drawStaffLines(ctx, 30, STAFF_TOP, W - 30);
    ctx.fillStyle = '#333';
    ctx.font = 'bold 18px serif';
    ctx.fillText('𝄞', 8, STAFF_TOP + 28);

    // Draw bass staff
    const bassTop = STAFF_TOP + STAFF_GAP;
    drawStaffLines(ctx, 30, bassTop, W - 30);
    ctx.fillStyle = '#333';
    ctx.font = 'bold 16px serif';
    ctx.fillText('𝄢', 8, bassTop + 22);

    // Draw notes
    notes.rh.forEach(n => {
        const x = 30 + n.start * timeScale;
        const y = midiToStaffY(n.pitch, 'treble');
        const w = Math.max(4, n.dur * timeScale);
        ctx.fillStyle = '#e74c3c';
        ctx.beginPath();
        ctx.ellipse(x + w/2, y, w/2, NOTE_RADIUS, 0, 0, Math.PI * 2);
        ctx.fill();
    });

    notes.lh.forEach(n => {
        const x = 30 + n.start * timeScale;
        const y = midiToStaffY(n.pitch, 'bass');
        const w = Math.max(4, n.dur * timeScale);
        ctx.fillStyle = '#3498db';
        ctx.beginPath();
        ctx.ellipse(x + w/2, y, w/2, NOTE_RADIUS, 0, 0, Math.PI * 2);
        ctx.fill();
    });

    // Playhead
    if (playheadTime >= 0 && playheadTime <= duration) {
        const px = 30 + playheadTime * timeScale;
        ctx.strokeStyle = '#2ecc71';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(px, 10);
        ctx.lineTo(px, H - 10);
        ctx.stroke();
    }
}

function drawStaffLines(ctx, x1, y, x2) {
    ctx.strokeStyle = '#444';
    ctx.lineWidth = 1;
    for (let i = 0; i < 5; i++) {
        const ly = y + i * LINE_SPACING;
        ctx.beginPath();
        ctx.moveTo(x1, ly);
        ctx.lineTo(x2, ly);
        ctx.stroke();
    }
}

// ============================================================
// Staff Playback Animation
// ============================================================
playStaffBtn.addEventListener('click', () => {
    if (staffPlaying) {
        stopStaffPlayback();
    } else {
        startStaffPlayback();
    }
});

function startStaffPlayback() {
    if (!staffDuration) return;
    staffPlaying = true;
    staffStartTime = performance.now();
    playStaffBtn.textContent = '⏹ Стоп';
    animateStaff();
}

function stopStaffPlayback() {
    staffPlaying = false;
    if (staffAnimFrame) cancelAnimationFrame(staffAnimFrame);
    playStaffBtn.textContent = '▶ Играть';
    drawStaff(staffNotes, staffDuration, 0);
    staffTime.textContent = `0.0s / ${staffDuration.toFixed(1)}s`;
}

function animateStaff() {
    if (!staffPlaying) return;
    const elapsed = (performance.now() - staffStartTime) / 1000;
    if (elapsed >= staffDuration) {
        stopStaffPlayback();
        return;
    }
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
    stopStaffPlayback();
}

// Prevent page scroll bounce on iOS
document.addEventListener('touchmove', (e) => {
    if (e.target === document.body) e.preventDefault();
}, { passive: false });
