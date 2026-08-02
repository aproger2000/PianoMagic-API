
import os
import uuid
import shutil
import subprocess
import logging
import struct
import math
import json
import tempfile
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# Пытаемся импортировать librosa — если нет, fallback на numpy
# ============================================================
try:
    import librosa
    import soundfile as sf
    HAS_LIBROSA = True
    logger.info("librosa доступен — используем продвинутый анализ")
except ImportError:
    HAS_LIBROSA = False
    logger.warning("librosa не найден — используем fallback-анализ")

app = FastAPI(title="PianoMagic API", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://aproger2000.github.io",
        "https://aproger2000.github.io/PianoMagic",
        "http://localhost:8080",
        "http://127.0.0.1:5500",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS_DIR = Path("/tmp/pianomagic_jobs")
JOBS_DIR.mkdir(exist_ok=True)
jobs = {}

SR = 44100  # Подняли с 22050 для качества
HOP_LENGTH = 512

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

STAGES = {
    "upload": {"text": "Загрузка файла...", "progress": 5},
    "analyze": {"text": "Анализ аудио...", "progress": 15},
    "melody": {"text": "Извлечение мелодии...", "progress": 30},
    "fit": {"text": "Подбор фортепианной партии...", "progress": 50},
    "compare": {"text": "Сличение отпечатков...", "progress": 70},
    "render": {"text": "Рендер WAV/PDF...", "progress": 90},
    "done": {"text": "Готово!", "progress": 100},
}

# ============================================================
# API ENDPOINTS (совместимость с фронтендом)
# ============================================================

@app.get("/")
async def root():
    return {"status": "ok", "version": "6.1", "engine": "librosa" if HAS_LIBROSA else "fallback"}

@app.post("/analyze")
async def analyze(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(file.filename or "input.mp3").suffix
    input_path = job_dir / f"input{ext}"

    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    jobs[job_id] = {
        "status": "processing",
        "stage": "upload",
        "input_path": str(input_path),
        "pdf_path": str(job_dir / "output.pdf"),
        "xml_path": str(job_dir / "output.xml"),
        "wav_path": str(job_dir / "output.wav"),
        "melody_rh": None,
        "melody_lh": None,
        "spec": None,
        "duration": 0,
        "similarity": 0.0,
        "chroma_orig": None,
        "chroma_synth": None,
        "error": None
    }
    background_tasks.add_task(process_audio_v5, job_id, input_path)
    return {"job_id": job_id, "status": "processing"}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    j = jobs[job_id]
    stage_info = STAGES.get(j.get("stage", "upload"), STAGES["upload"])
    return {
        "job_id": job_id,
        "status": j["status"],
        "stage": j.get("stage", "upload"),
        "stage_text": stage_info["text"],
        "progress": stage_info["progress"],
        "similarity": j.get("similarity", 0.0),
        "error": j.get("error"),
    }

@app.get("/melody/{job_id}")
async def get_melody(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    if jobs[job_id]["melody_rh"] is None:
        raise HTTPException(status_code=400, detail="Melody not ready")
    return JSONResponse({
        "melody_rh": jobs[job_id]["melody_rh"],
        "melody_lh": jobs[job_id]["melody_lh"],
        "spec": jobs[job_id]["spec"],
        "duration": jobs[job_id]["duration"],
        "similarity": jobs[job_id].get("similarity", 0.0),
        "chroma_orig": jobs[job_id].get("chroma_orig"),
        "chroma_synth": jobs[job_id].get("chroma_synth"),
    })

@app.post("/render/{job_id}")
async def render_pdf(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    if jobs[job_id]["melody_rh"] is None:
        raise HTTPException(status_code=400, detail="Melody not ready")

    xml_path = Path(jobs[job_id]["xml_path"])
    pdf_path = Path(jobs[job_id]["pdf_path"])
    wav_path = Path(jobs[job_id]["wav_path"])

    try:
        build_musicxml_v5(jobs[job_id]["melody_rh"], jobs[job_id]["melody_lh"], xml_path)
        generate_wav_v5(jobs[job_id]["melody_rh"], jobs[job_id]["melody_lh"], wav_path)

        mscore = find_musescore()
        if mscore:
            result = subprocess.run(
                [mscore, "-o", str(pdf_path), str(xml_path)],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode != 0:
                logger.warning(f"MuseScore error: {result.stderr or result.stdout}")
        else:
            logger.warning("MuseScore не найден, PDF не создан")

        jobs[job_id]["status"] = "completed"
        return {"status": "completed", "job_id": job_id}
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{job_id}.pdf")
async def download_pdf(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    pdf_path = Path(jobs[job_id]["pdf_path"])
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF not ready")
    return FileResponse(pdf_path, media_type="application/pdf", filename=f"PianoMagic_{job_id}.pdf")

@app.get("/download/{job_id}.wav")
async def download_wav(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    wav_path = Path(jobs[job_id]["wav_path"])
    if not wav_path.exists():
        raise HTTPException(status_code=404, detail="WAV not ready")
    return FileResponse(wav_path, media_type="audio/wav", filename=f"PianoMagic_{job_id}.wav")

# ============================================================
# УТИЛИТЫ
# ============================================================

def set_stage(job_id, stage):
    if job_id in jobs:
        jobs[job_id]["stage"] = stage
        logger.info(f"[{job_id}] Stage: {stage}")

def read_audio(path):
    """Читает аудио через ffmpeg или librosa"""
    if HAS_LIBROSA:
        y, sr = librosa.load(str(path), sr=SR, mono=True)
        return y, sr
    else:
        cmd = ['ffmpeg', '-y', '-i', str(path), '-ar', str(SR), '-ac', '1', '-f', 'f32le', '-']
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg error: {result.stderr.decode()[:200]}")
        return np.frombuffer(result.stdout, dtype=np.float32), SR

def midi_to_name(midi):
    name = NOTE_NAMES[midi % 12]
    octave = (midi // 12) - 1
    if '#' in name:
        return name[0], 1, octave
    return name, 0, octave

def dur_info(dur):
    """Возвращает тип ноты и точку для MusicXML (divisions=8)"""
    if dur >= 28: return 'whole', False
    if dur >= 20: return 'half', True
    if dur >= 12: return 'half', False
    if dur >= 10: return 'quarter', True
    if dur >= 6: return 'quarter', False
    if dur >= 5: return 'eighth', True
    if dur >= 3: return 'eighth', False
    return '16th', False

def find_musescore():
    for cmd in ["musescore3", "musescore", "mscore"]:
        if shutil.which(cmd):
            return cmd
    return None

def compute_spectrogram(audio, sr, n_fft=2048, hop=512):
    frames = []
    window = np.hanning(n_fft)
    for i in range(0, len(audio) - n_fft, hop):
        frame = audio[i:i + n_fft] * window
        spec = np.abs(np.fft.rfft(frame))
        frames.append(spec)
    if not frames:
        return np.zeros((n_fft // 2 + 1, 1))
    spec = np.array(frames).T
    spec_db = 20 * np.log10(spec + 1e-10)
    vmax = spec_db.max()
    spec_db = np.clip(spec_db, vmax - 60, vmax)
    spec_db = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min() + 1e-10)
    freq_step = max(1, spec_db.shape[0] // 128)
    time_step = max(1, spec_db.shape[1] // 200)
    spec_db = spec_db[::freq_step, ::time_step]
    return (spec_db * 255).astype(np.uint8)

# ============================================================
# ЯДРО v5.0: ПОДБОР НОТ ВМЕСТО РАСПОЗНАВАНИЯ
# ============================================================

def process_audio_v5(job_id, input_path):
    """Основной пайплайн обработки"""
    try:
        set_stage(job_id, "analyze")

        # 1. Читаем аудио
        audio, sr = read_audio(input_path)
        duration = len(audio) / sr
        jobs[job_id]["duration"] = round(duration, 2)

        # 2. Спектрограмма для фронтенда
        spec = compute_spectrogram(audio, sr)
        jobs[job_id]["spec"] = spec.tolist()

        if not HAS_LIBROSA:
            # Fallback на старый метод
            set_stage(job_id, "melody")
            rh, lh = extract_melody_fallback(audio, sr)
            jobs[job_id]["melody_rh"] = rh
            jobs[job_id]["melody_lh"] = lh
            set_stage(job_id, "done")
            jobs[job_id]["status"] = "completed"
            return

        # 3. Извлекаем мелодию через librosa
        set_stage(job_id, "melody")
        melody_data = extract_melody_librosa(audio, sr)

        # 4. Подбираем фортепианную партию
        set_stage(job_id, "fit")
        rh_notes, lh_notes = fit_piano_arrangement(melody_data)

        # 5. Сохраняем ноты
        jobs[job_id]["melody_rh"] = rh_notes
        jobs[job_id]["melody_lh"] = lh_notes

        # 6. Рендер WAV (теперь ДО сравнения — тот же файл, что получит пользователь)
        set_stage(job_id, "render")
        xml_path = Path(jobs[job_id]["xml_path"])
        wav_path = Path(jobs[job_id]["wav_path"])
        pdf_path = Path(jobs[job_id]["pdf_path"])

        build_musicxml_v5(rh_notes, lh_notes, xml_path)
        generate_wav_v5(rh_notes, lh_notes, wav_path)

        mscore = find_musescore()
        if mscore:
            subprocess.run([mscore, "-o", str(pdf_path), str(xml_path)], 
                          capture_output=True, timeout=120)

        # 7. Сличение отпечатков — сравниваем с ФИНАЛЬНЫМ WAV
        set_stage(job_id, "compare")
        similarity, chroma_orig_mean, chroma_synth_mean = compare_fingerprints(audio, sr, wav_path)
        jobs[job_id]["similarity"] = round(similarity, 4)
        jobs[job_id]["chroma_orig"] = chroma_orig_mean
        jobs[job_id]["chroma_synth"] = chroma_synth_mean

        set_stage(job_id, "done")
        jobs[job_id]["status"] = "completed"
        logger.info(f"[{job_id}] Готово! Сходство: {similarity:.3f}")

    except Exception as e:
        logger.error(f"[{job_id}] Ошибка: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


def extract_melody_librosa(audio, sr):
    """Извлекает мелодию с помощью librosa"""
    # Отделяем гармонику
    y_harm = librosa.effects.harmonic(audio, margin=4.0)

    # Темп
    tempo = float(librosa.beat.beat_track(y=audio, sr=sr)[0])
    if tempo < 40 or tempo > 200:
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        of = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        if len(of) > 1:
            tempo = 60.0 / (np.median(np.diff(of)) * 512 / sr)
        tempo = max(40, min(200, tempo))

    beat_dur = 60.0 / tempo

    # Onset detection
    onset_env = librosa.onset.onset_strength(y=y_harm, sr=sr, hop_length=HOP_LENGTH)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH,
        pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.05, wait=3
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=HOP_LENGTH)

    # Pitch tracking
    pitches, mags = librosa.piptrack(
        y=y_harm, sr=sr, hop_length=HOP_LENGTH,
        fmin=librosa.note_to_hz('C3'), fmax=librosa.note_to_hz('C7')
    )

    melody_events = []
    for i, of in enumerate(onset_frames):
        window = slice(of, min(of + 10, pitches.shape[1]))
        best_pitch, best_mag = 0, 0
        for t in range(window.start, window.stop):
            for f in range(pitches.shape[0]):
                if mags[f, t] > best_mag and pitches[f, t] > 0:
                    best_mag = mags[f, t]
                    best_pitch = pitches[f, t]

        if best_pitch > 0 and best_mag > 0.005:
            midi = int(round(librosa.hz_to_midi(best_pitch)))
            end_time = onset_times[i+1] if i+1 < len(onset_times) else len(audio)/sr
            note_dur = min(end_time - onset_times[i], beat_dur * 2)
            melody_events.append({
                'time': onset_times[i], 'midi': midi,
                'mag': best_mag, 'dur': note_dur
            })

    # Упрощаем
    melody_events = sorted(melody_events, key=lambda e: e['time'])
    simplified = []
    last_midi, last_time = -1, -1
    for e in melody_events:
        if e['midi'] != last_midi or (e['time'] - last_time) > 0.08:
            simplified.append(e)
            last_midi = e['midi']
            last_time = e['time']

    # Фильтруем верхний голос
    high_voice = [e for e in simplified if e['midi'] >= 65]

    # Квантизация по битам
    beat_dur = 60.0 / tempo
    quantized = {}
    for e in high_voice:
        beat = int(round(e['time'] / beat_dur))
        if beat not in quantized or e['mag'] > quantized[beat]['mag']:
            quantized[beat] = e

    melody_beats = [quantized[b] for b in sorted(quantized.keys())]

    # Убираем повторы
    final_melody = []
    last_midi = -1
    for e in melody_beats:
        if e['midi'] != last_midi:
            final_melody.append(e)
            last_midi = e['midi']

    # Тональность
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=HOP_LENGTH)
    key_idx = int(np.argmax(np.mean(chroma, axis=1)))

    return {
        'melody': final_melody,
        'tempo': tempo,
        'beat_dur': beat_dur,
        'key_idx': key_idx,
        'duration': len(audio) / sr,
        'audio': audio,
        'sr': sr,
    }


def fit_piano_arrangement(melody_data, strategy='adaptive'):
    """Подбирает фортепианную партию"""
    melody = melody_data['melody']
    tempo = melody_data['tempo']
    beat_dur = melody_data['beat_dur']
    key_idx = melody_data['key_idx']

    if not melody:
        return [], []

    rh_notes = []
    lh_notes = []

    # Транспозиция в комфортный диапазон
    melody_midis = [e['midi'] for e in melody]
    center = (min(melody_midis) + max(melody_midis)) / 2
    transpose = int(round(72 - center))
    transpose = max(-12, min(12, transpose))

    # Правая рука: мелодия
    for i, e in enumerate(melody):
        rh_pitch = max(60, min(96, e['midi'] + transpose))
        dur = max(0.1, min(e['dur'], beat_dur * 1.5))

        rh_notes.append({
            'start': round(e['time'], 3),
            'dur': round(dur * 0.9, 3),
            'pitch': rh_pitch,
            'velocity': min(127, int(80 + e['mag'] * 40))
        })

        # Квинта на сильных долях
        if i % 4 == 0 and rh_pitch < 90:
            rh_notes.append({
                'start': round(e['time'], 3),
                'dur': round(dur * 0.4, 3),
                'pitch': rh_pitch + 7,
                'velocity': 45
            })

    # Левая рука: аккомпанемент
    for i in range(0, len(melody), 2):
        e = melody[i]
        degree = (e['midi'] % 12 - key_idx) % 12

        if degree in [0, 2, 4]:
            intervals = [0, 4, 7]
        elif degree in [5, 7]:
            intervals = [7, 11, 2] if degree == 7 else [5, 9, 0]
        else:
            intervals = [0, 4, 7]

        bass_root = 36 + (key_idx + intervals[0]) % 12

        for j, (p, toff) in enumerate(zip(
            [bass_root, bass_root + intervals[1], bass_root + 12, bass_root + intervals[1]],
            [0, 0.25, 0.5, 0.75]
        )):
            lh_notes.append({
                'start': round(e['time'] + toff * beat_dur, 3),
                'dur': round(beat_dur * 0.22, 3),
                'pitch': min(72, p),
                'velocity': 50
            })

    return rh_notes, lh_notes


def compare_fingerprints(original_audio, sr, wav_path):
    """Сравнивает оригинал с уже сгенерированным WAV"""
    try:
        # Читаем финальный синтез
        candidate, sr_cand = librosa.load(wav_path, sr=sr, mono=True)

        # Приводим к одной длине
        min_len = min(len(original_audio), len(candidate))
        orig = original_audio[:min_len]
        cand = candidate[:min_len]

        # Хромаграммное сходство
        c_orig = librosa.feature.chroma_stft(y=orig, sr=sr, hop_length=HOP_LENGTH)
        c_cand = librosa.feature.chroma_stft(y=cand, sr=sr, hop_length=HOP_LENGTH)
        min_c = min(c_orig.shape[1], c_cand.shape[1])
        chroma_sims = []
        for i in range(min_c):
            no = np.linalg.norm(c_orig[:, i]) + 1e-8
            nc = np.linalg.norm(c_cand[:, i]) + 1e-8
            chroma_sims.append(np.dot(c_orig[:, i], c_cand[:, i]) / (no * nc))
        chroma_sim = float(np.mean(chroma_sims))

        # Onset correlation
        o_orig = librosa.onset.onset_strength(y=orig, sr=sr, hop_length=HOP_LENGTH)
        o_cand = librosa.onset.onset_strength(y=cand, sr=sr, hop_length=HOP_LENGTH)
        min_o = min(len(o_orig), len(o_cand))
        o1 = o_orig[:min_o] / (np.max(o_orig) + 1e-8)
        o2 = o_cand[:min_o] / (np.max(o_cand) + 1e-8)
        onset_corr = float(np.corrcoef(o1, o2)[0, 1])
        if np.isnan(onset_corr):
            onset_corr = 0.0

        # Pitch contour
        f0_orig, _, _ = librosa.pyin(orig, fmin=librosa.note_to_hz('C3'),
                                       fmax=librosa.note_to_hz('C7'), sr=sr)
        f0_cand, _, _ = librosa.pyin(cand, fmin=librosa.note_to_hz('C3'),
                                       fmax=librosa.note_to_hz('C7'), sr=sr)
        f0_orig = np.nan_to_num(f0_orig, nan=0.0)
        f0_cand = np.nan_to_num(f0_cand, nan=0.0)
        min_f0 = min(len(f0_orig), len(f0_cand))
        valid = (f0_orig[:min_f0] > 0) & (f0_cand[:min_f0] > 0)
        if np.sum(valid) > 10:
            m1 = librosa.hz_to_midi(f0_orig[:min_f0][valid])
            m2 = librosa.hz_to_midi(f0_cand[:min_f0][valid])
            diff = np.abs(m1 - m2)
            diff = np.minimum(diff, 12 - diff)
            pitch_sim = float(np.mean(np.exp(-diff / 2.0)))
        else:
            pitch_sim = 0.0

        # Итоговый score
        total = chroma_sim * 0.25 + max(0, onset_corr) * 0.25 + pitch_sim * 0.30

        # Mel similarity
        mel_orig = librosa.feature.melspectrogram(y=orig, sr=sr, hop_length=HOP_LENGTH, n_mels=40)
        mel_cand = librosa.feature.melspectrogram(y=cand, sr=sr, hop_length=HOP_LENGTH, n_mels=40)
        min_m = min(mel_orig.shape[1], mel_cand.shape[1])
        m1 = librosa.power_to_db(mel_orig[:, :min_m] + 1e-8)
        m2 = librosa.power_to_db(mel_cand[:, :min_m] + 1e-8)
        mel_sim = max(0, 1 - np.mean(np.abs(m1 - m2)) / 80.0)

        total += mel_sim * 0.20

        # Average chroma vectors for histograms
        chroma_orig_mean = c_orig.mean(axis=1).tolist() if c_orig is not None else [0]*12
        chroma_synth_mean = c_cand.mean(axis=1).tolist() if c_cand is not None else [0]*12

        return min(1.0, max(0.0, total)), chroma_orig_mean, chroma_synth_mean
    except Exception as e:
        logger.warning(f"Fingerprint comparison failed: {e}")
        return 0.5



def extract_melody_fallback(audio, sr):
    """Старый метод для fallback"""
    N_FFT = 4096
    HOP = int(sr * 0.05)
    freqs = np.fft.rfftfreq(N_FFT, 1 / sr)
    frames = []

    for i in range(0, len(audio) - N_FFT, HOP):
        frame = audio[i:i + N_FFT] * np.hanning(N_FFT)
        spectrum = np.abs(np.fft.rfft(frame))

        mask = (freqs >= 200) & (freqs <= 2000)
        band_freqs = freqs[mask]
        band_spec = spectrum[mask]

        if len(band_spec) == 0:
            continue

        idx = np.argmax(band_spec)
        peak_freq = band_freqs[idx]
        peak_amp = band_spec[idx]

        global_max = np.max(spectrum) + 1e-10
        amp_norm = min(1.0, peak_amp / global_max)
        if amp_norm < 0.02:
            continue

        time = i / sr
        midi = int(round(69 + 12 * np.log2(peak_freq / 440.0)))
        frames.append({'time': time, 'freq': peak_freq, 'midi': midi, 'amp': amp_norm})

    if not frames:
        return [], []

    # Median filter
    if len(frames) > 7:
        pitches = [f['midi'] for f in frames]
        from scipy.ndimage import median_filter
        smoothed = median_filter(pitches, size=7)
        for i in range(len(frames)):
            frames[i]['midi'] = int(smoothed[i])

    # Segment
    notes = []
    current = {'start': frames[0]['time'], 'pitch': frames[0]['midi'], 'amp': frames[0]['amp']}
    for f in frames[1:]:
        if f['midi'] != current['pitch']:
            dur = f['time'] - current['start']
            if dur >= 0.15:
                notes.append({
                    'start': round(current['start'], 2),
                    'dur': round(dur, 2),
                    'pitch': current['pitch'],
                    'velocity': min(127, int(current['amp'] * 127))
                })
            current = {'start': f['time'], 'pitch': f['midi'], 'amp': f['amp']}

    dur = len(audio) / sr - current['start']
    if dur >= 0.15:
        notes.append({
            'start': round(current['start'], 2),
            'dur': round(dur, 2),
            'pitch': current['pitch'],
            'velocity': min(127, int(current['amp'] * 127))
        })

    # Deduplicate
    if not notes:
        return [], []

    deduped = [notes[0]]
    for n in notes[1:]:
        if n['pitch'] == deduped[-1]['pitch'] and n['start'] - (deduped[-1]['start'] + deduped[-1]['dur']) < 0.15:
            deduped[-1]['dur'] = round(n['start'] + n['dur'] - deduped[-1]['start'], 2)
            deduped[-1]['velocity'] = max(deduped[-1]['velocity'], n['velocity'])
        elif n['pitch'] != deduped[-1]['pitch']:
            deduped.append(n)

    # Split RH/LH
    rh = [n for n in deduped if n['pitch'] >= 60]
    lh = [n for n in deduped if n['pitch'] < 60]

    return rh, lh


# ============================================================
# РЕНДЕРИНГ
# ============================================================

def generate_wav_v5(rh_notes, lh_notes, wav_path, sr=44100):
    """Улучшенный синтез"""
    all_notes = rh_notes + lh_notes
    if not all_notes:
        return

    total_dur = max(n["start"] + n["dur"] for n in all_notes)
    total_samples = int(total_dur * sr) + sr
    audio = np.zeros(total_samples, dtype=np.float32)

    for n in all_notes:
        freq = 440.0 * (2.0 ** ((n["pitch"] - 69) / 12.0))
        start_sample = int(n["start"] * sr)
        dur_samples = int(n["dur"] * sr)
        if dur_samples <= 0:
            continue

        t = np.linspace(0, n["dur"], dur_samples, endpoint=False)
        wave = 0.55 * np.sin(2 * np.pi * freq * t)
        wave += 0.25 * np.sin(2 * np.pi * freq * 2 * t)
        wave += 0.12 * np.sin(2 * np.pi * freq * 3 * t)
        wave += 0.05 * np.sin(2 * np.pi * freq * 4 * t)
        wave += 0.03 * np.sin(2 * np.pi * freq * 5 * t)

        attack = int(0.008 * sr)
        decay = int(0.15 * sr)
        release = int(0.08 * sr)
        sustain = 0.6

        env = np.ones(dur_samples) * sustain
        if attack > 0:
            env[:attack] = np.linspace(0, 1, attack)
        if decay > 0 and attack + decay < dur_samples:
            env[attack:attack+decay] = np.linspace(1, sustain, decay)
        if release > 0 and dur_samples > release:
            env[-release:] = np.linspace(sustain, 0, release)

        end = min(start_sample + dur_samples, total_samples)
        actual = end - start_sample
        audio[start_sample:end] += wave[:actual] * env[:actual] * 0.3

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95

    audio_int16 = (audio * 32767).astype(np.int16)
    with open(wav_path, 'wb') as f:
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + total_samples * 2))
        f.write(b'WAVEfmt ')
        f.write(struct.pack('<IHHIIHH', 16, 1, 1, sr, sr * 2, 2, 16))
        f.write(b'data')
        f.write(struct.pack('<I', total_samples * 2))
        f.write(audio_int16.tobytes())


def build_musicxml_v5(rh_notes, lh_notes, xml_path):
    """Улучшенный MusicXML генератор"""
    DIVISIONS = 8

    root = Element('score-partwise', {'version': '3.1'})
    part_list = SubElement(root, 'part-list')

    score_part = SubElement(part_list, 'score-part', {'id': 'P1'})
    SubElement(score_part, 'part-name').text = 'Piano'

    part1 = SubElement(root, 'part', {'id': 'P1'})

    # Группируем ноты RH по тактам
    measures_rh = {}
    for n in rh_notes:
        m = int(n["start"] // 4.0)
        off = n["start"] % 4.0
        if off >= 3.999:
            m += 1
            off = 0.0
        measures_rh.setdefault(m, []).append({
            "offset": int(round(off * DIVISIONS)),
            "dur": int(round(n["dur"] * DIVISIONS)),
            "pitch": n["pitch"]
        })

    # Группируем ноты LH по тактам
    measures_lh = {}
    for n in lh_notes:
        m = int(n["start"] // 4.0)
        off = n["start"] % 4.0
        if off >= 3.999:
            m += 1
            off = 0.0
        measures_lh.setdefault(m, []).append({
            "offset": int(round(off * DIVISIONS)),
            "dur": int(round(n["dur"] * DIVISIONS)),
            "pitch": n["pitch"]
        })

    all_measures = sorted(set(list(measures_rh.keys()) + list(measures_lh.keys())))

    for m_idx in all_measures:
        measure = SubElement(part1, 'measure', {'number': str(m_idx + 1)})

        if m_idx == 0:
            attr = SubElement(measure, 'attributes')
            SubElement(attr, 'divisions').text = str(DIVISIONS)

            # Два нотоносца
            staves = SubElement(attr, 'staves')
            staves.text = '2'

            # Скрипичный ключ
            clef1 = SubElement(attr, 'clef', {'number': '1'})
            SubElement(clef1, 'sign').text = 'G'
            SubElement(clef1, 'line').text = '2'

            # Басовый ключ
            clef2 = SubElement(attr, 'clef', {'number': '2'})
            SubElement(clef2, 'sign').text = 'F'
            SubElement(clef2, 'line').text = '4'

            # Тональность (C major по умолчанию)
            key = SubElement(attr, 'key')
            SubElement(key, 'fifths').text = '0'

            # Размер
            time = SubElement(attr, 'time')
            SubElement(time, 'beats').text = '4'
            SubElement(time, 'beat-type').text = '4'

        # RH events
        events = sorted(measures_rh.get(m_idx, []), key=lambda x: x['offset'])
        last_end = 0
        for evt in events:
            off = evt['offset']
            dur = evt['dur']
            pitch = evt['pitch']

            gap = off - last_end
            if gap >= 2:
                r = SubElement(measure, 'note')
                SubElement(r, 'rest')
                SubElement(r, 'duration').text = str(gap)
                staff_el = SubElement(r, 'staff')
                staff_el.text = '1'
                t, dot = dur_info(gap)
                SubElement(r, 'type').text = t
                if dot:
                    SubElement(r, 'dot')

            step, alter, octv = midi_to_name(pitch)
            n = SubElement(measure, 'note')
            p = SubElement(n, 'pitch')
            SubElement(p, 'step').text = step
            if alter != 0:
                SubElement(p, 'alter').text = str(alter)
            SubElement(p, 'octave').text = str(octv)
            SubElement(n, 'duration').text = str(dur)
            staff_el = SubElement(n, 'staff')
            staff_el.text = '1'
            t, dot = dur_info(dur)
            SubElement(n, 'type').text = t
            if dot:
                SubElement(n, 'dot')

            last_end = off + dur

        fill = 32 - last_end
        if fill >= 2:
            r = SubElement(measure, 'note')
            SubElement(r, 'rest')
            SubElement(r, 'duration').text = str(fill)
            staff_el = SubElement(r, 'staff')
            staff_el.text = '1'
            t, dot = dur_info(fill)
            SubElement(r, 'type').text = t
            if dot:
                SubElement(r, 'dot')

        # LH events
        events = sorted(measures_lh.get(m_idx, []), key=lambda x: x['offset'])
        last_end = 0
        for evt in events:
            off = evt['offset']
            dur = evt['dur']
            pitch = evt['pitch']

            gap = off - last_end
            if gap >= 2:
                r = SubElement(measure, 'note')
                SubElement(r, 'rest')
                SubElement(r, 'duration').text = str(gap)
                staff_el = SubElement(r, 'staff')
                staff_el.text = '2'
                t, dot = dur_info(gap)
                SubElement(r, 'type').text = t
                if dot:
                    SubElement(r, 'dot')

            step, alter, octv = midi_to_name(pitch)
            n = SubElement(measure, 'note')
            p = SubElement(n, 'pitch')
            SubElement(p, 'step').text = step
            if alter != 0:
                SubElement(p, 'alter').text = str(alter)
            SubElement(p, 'octave').text = str(octv)
            SubElement(n, 'duration').text = str(dur)
            staff_el = SubElement(n, 'staff')
            staff_el.text = '2'
            t, dot = dur_info(dur)
            SubElement(n, 'type').text = t
            if dot:
                SubElement(n, 'dot')

            last_end = off + dur

        fill = 32 - last_end
        if fill >= 2:
            r = SubElement(measure, 'note')
            SubElement(r, 'rest')
            SubElement(r, 'duration').text = str(fill)
            staff_el = SubElement(r, 'staff')
            staff_el.text = '2'
            t, dot = dur_info(fill)
            SubElement(r, 'type').text = t
            if dot:
                SubElement(r, 'dot')

    xml_str = tostring(root, encoding='unicode')
    doctype = '<?xml version="1.0" encoding="UTF-8"?>\n'
    with open(xml_path, 'w', encoding='utf-8') as f:
        f.write(doctype + xml_str)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
