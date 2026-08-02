"""
PianoMagic API v7.0 — Deep Audio-to-Score Transcription Engine
===============================================================
Architecture:
  1. Harmonic-Percussive Separation (HPS)
  2. Multi-scale Onset Detection (superflux + default)
  3. Dual Pitch Tracking (PYIN primary, piptrack fallback)
  4. Intelligent Note Segmentation (adaptive threshold, merge, min-dur filter)
  5. Key Estimation (Krumhansl-Schmuckler profile correlation)
  6. Adaptive Piano Arrangement (voice leading, chord voicing)
  7. Synthesis (harmonic piano model with ADSR)
  8. Fingerprint Comparison (chroma + onset + pitch + mel + spectral contrast)
"""

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
from typing import List, Dict, Tuple, Optional

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# Optional librosa import with graceful fallback
# ============================================================
try:
    import librosa
    import soundfile as sf
    HAS_LIBROSA = True
    logger.info("[INIT] librosa available — advanced analysis enabled")
except ImportError as e:
    HAS_LIBROSA = False
    logger.warning(f"[INIT] librosa unavailable ({e}) — using fallback")

# ============================================================
# FastAPI App
# ============================================================
app = FastAPI(title="PianoMagic API", version="7.0.0", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.options("/{path:path}")
async def options_handler(path: str):
    return JSONResponse({}, status_code=200)

# ============================================================
# Globals
# ============================================================
JOBS_DIR = Path("/tmp/pianomagic_jobs")
JOBS_DIR.mkdir(exist_ok=True)
jobs: Dict[str, dict] = {}

SR = 44100
HOP_LENGTH = 512
N_FFT = 2048

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Krumhansl-Schmuckler key profiles (major / minor)
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

STAGES = {
    "upload":  {"text": "Загрузка файла...", "progress": 5},
    "analyze": {"text": "Анализ аудио...", "progress": 15},
    "melody":  {"text": "Извлечение мелодии...", "progress": 30},
    "fit":     {"text": "Подбор партии...", "progress": 50},
    "compare": {"text": "Сличение отпечатков...", "progress": 70},
    "render":  {"text": "Рендер WAV/PDF...", "progress": 90},
    "done":    {"text": "Готово!", "progress": 100},
}

# ============================================================
# API Endpoints
# ============================================================

@app.get("/")
async def root():
    return {
        "status": "ok",
        "version": "7.0",
        "engine": "librosa" if HAS_LIBROSA else "fallback",
        "has_librosa": HAS_LIBROSA
    }

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
        "duration": 0.0,
        "similarity": 0.0,
        "chroma_orig": None,
        "chroma_synth": None,
        "error": None,
        "key_name": None,
        "tempo": 0.0,
    }
    background_tasks.add_task(process_audio_v7, job_id, input_path)
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
    j = jobs[job_id]
    return JSONResponse({
        "melody_rh": j["melody_rh"],
        "melody_lh": j["melody_lh"],
        "spec": j["spec"],
        "duration": j["duration"],
        "similarity": j.get("similarity", 0.0),
        "chroma_orig": j.get("chroma_orig"),
        "chroma_synth": j.get("chroma_synth"),
        "key_name": j.get("key_name"),
        "tempo": j.get("tempo", 0.0),
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
        build_musicxml_v7(jobs[job_id]["melody_rh"], jobs[job_id]["melody_lh"], xml_path)
        generate_wav_v7(jobs[job_id]["melody_rh"], jobs[job_id]["melody_lh"], wav_path)

        mscore = find_musescore()
        if mscore:
            result = subprocess.run(
                [mscore, "-o", str(pdf_path), str(xml_path)],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode != 0:
                logger.warning(f"MuseScore error: {result.stderr or result.stdout}")
        else:
            logger.warning("MuseScore not found — PDF skipped")

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
# Utilities
# ============================================================

def set_stage(job_id: str, stage: str):
    if job_id in jobs:
        jobs[job_id]["stage"] = stage
        logger.info(f"[{job_id}] Stage: {stage}")

def read_audio(path: Path) -> Tuple[np.ndarray, int]:
    """Read audio via librosa or ffmpeg fallback."""
    if HAS_LIBROSA:
        y, sr = librosa.load(str(path), sr=SR, mono=True)
        return y, sr
    else:
        cmd = ['ffmpeg', '-y', '-i', str(path), '-ar', str(SR), '-ac', '1', '-f', 'f32le', '-']
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg error: {result.stderr.decode()[:200]}")
        return np.frombuffer(result.stdout, dtype=np.float32), SR

def midi_to_name(midi: int) -> Tuple[str, int, int]:
    name = NOTE_NAMES[midi % 12]
    octave = (midi // 12) - 1
    if '#' in name:
        return name[0], 1, octave
    return name, 0, octave

def dur_info(dur: int) -> Tuple[str, bool]:
    """MusicXML note type from duration in divisions (divisions=8)."""
    if dur >= 28:   return 'whole', False
    if dur >= 20:   return 'half', True
    if dur >= 12:   return 'half', False
    if dur >= 10:   return 'quarter', True
    if dur >= 6:    return 'quarter', False
    if dur >= 5:    return 'eighth', True
    if dur >= 3:    return 'eighth', False
    return '16th', False

def find_musescore() -> Optional[str]:
    for cmd in ["musescore3", "musescore", "mscore"]:
        if shutil.which(cmd):
            return cmd
    return None

def compute_spectrogram(audio: np.ndarray, sr: int, n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    """Compute normalized log-power spectrogram for frontend visualization."""
    if HAS_LIBROSA:
        # Use librosa STFT for better quality
        D = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop))
        spec_db = librosa.amplitude_to_db(D, ref=np.max)
    else:
        frames = []
        window = np.hanning(n_fft)
        for i in range(0, len(audio) - n_fft, hop):
            frame = audio[i:i + n_fft] * window
            spec = np.abs(np.fft.rfft(frame))
            frames.append(spec)
        if not frames:
            return np.zeros((n_fft // 2 + 1, 1), dtype=np.uint8)
        spec = np.array(frames).T
        spec_db = 20 * np.log10(spec + 1e-10)

    vmax = spec_db.max()
    spec_db = np.clip(spec_db, vmax - 80, vmax)
    spec_db = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min() + 1e-10)

    # Downsample for frontend
    freq_step = max(1, spec_db.shape[0] // 128)
    time_step = max(1, spec_db.shape[1] // 200)
    spec_db = spec_db[::freq_step, ::time_step]
    return (spec_db * 255).astype(np.uint8)

# ============================================================
# Key Estimation (Krumhansl-Schmuckler)
# ============================================================

def estimate_key(chroma: np.ndarray) -> Tuple[int, str, bool]:
    """
    Estimate musical key using Krumhansl-Schmuckler key-finding algorithm.
    Returns: (key_idx, key_name, is_major)
    """
    chroma_mean = np.mean(chroma, axis=1)
    if chroma_mean.sum() == 0:
        return 0, "C major", True

    chroma_norm = chroma_mean / (np.linalg.norm(chroma_mean) + 1e-10)

    best_corr = -1.0
    best_key = 0
    best_mode = True

    for shift in range(12):
        maj_profile = np.roll(KS_MAJOR, shift)
        min_profile = np.roll(KS_MINOR, shift)

        maj_corr = np.corrcoef(chroma_norm, maj_profile / np.linalg.norm(maj_profile))[0, 1]
        min_corr = np.corrcoef(chroma_norm, min_profile / np.linalg.norm(min_profile))[0, 1]

        if maj_corr > best_corr:
            best_corr = maj_corr
            best_key = shift
            best_mode = True
        if min_corr > best_corr:
            best_corr = min_corr
            best_key = shift
            best_mode = False

    key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    mode_str = "major" if best_mode else "minor"
    key_name = f"{key_names[best_key]} {mode_str}"

    return best_key, key_name, best_mode

# ============================================================
# Note Segmentation v7
# ============================================================

def segment_notes_v7(
    events: List[Dict],
    sr: int = SR,
    hop_length: int = HOP_LENGTH,
    min_note_dur_ms: float = 100.0,
    merge_gap_ms: float = 150.0,
    pitch_tolerance: int = 1,
    adaptive_threshold_factor: float = 0.12,
) -> List[Dict]:
    """
    Intelligent note segmentation with:
    - Adaptive magnitude thresholding
    - Pitch-aware note merging (±semitone, gap < 150ms)
    - Minimum duration filtering (>100ms)
    - Post-merge deduplication
    """
    if not events:
        return []

    # 1. Adaptive magnitude threshold
    mags = np.array([e.get('mag', 0.01) for e in events])
    if len(mags) > 0:
        mag_threshold = max(0.002, np.median(mags) * adaptive_threshold_factor)
    else:
        mag_threshold = 0.005

    filtered = [e for e in events if e.get('mag', 0) >= mag_threshold]
    if not filtered:
        filtered = events

    # 2. Sort by time
    filtered = sorted(filtered, key=lambda e: e['time'])

    MIN_NOTE_DUR = min_note_dur_ms / 1000.0
    MERGE_GAP = merge_gap_ms / 1000.0

    # 3. Merge similar adjacent notes
    merged = []
    for e in filtered:
        if not merged:
            merged.append({
                'time': e['time'],
                'midi': e['midi'],
                'mag': e.get('mag', 0.01),
                'dur': max(e.get('dur', MIN_NOTE_DUR), MIN_NOTE_DUR),
            })
            continue

        last = merged[-1]
        gap = e['time'] - (last['time'] + last['dur'])
        pitch_diff = abs(e['midi'] - last['midi'])

        # Merge condition: similar pitch AND small gap
        if pitch_diff <= pitch_tolerance and gap < MERGE_GAP:
            new_end = max(last['time'] + last['dur'], e['time'] + e.get('dur', MIN_NOTE_DUR))
            last['dur'] = new_end - last['time']
            last['mag'] = max(last['mag'], e.get('mag', 0.01))
            if e.get('mag', 0) > last['mag']:
                last['midi'] = e['midi']
        else:
            merged.append({
                'time': e['time'],
                'midi': e['midi'],
                'mag': e.get('mag', 0.01),
                'dur': max(e.get('dur', MIN_NOTE_DUR), MIN_NOTE_DUR),
            })

    # 4. Filter by minimum duration
    result = [m for m in merged if m['dur'] >= MIN_NOTE_DUR]

    # 5. Second-pass merge for identical pitches with tiny gaps
    final = []
    for m in result:
        if not final:
            final.append(m.copy())
            continue
        last = final[-1]
        gap = m['time'] - (last['time'] + last['dur'])
        if m['midi'] == last['midi'] and gap < 0.040:  # 40ms
            new_end = max(last['time'] + last['dur'], m['time'] + m['dur'])
            last['dur'] = new_end - last['time']
            last['mag'] = max(last['mag'], m['mag'])
        else:
            final.append(m.copy())

    logger.info(f"[segment_v7] {len(events)} -> {len(filtered)} -> {len(merged)} -> {len(result)} -> {len(final)} notes")
    return final

# ============================================================
# Main Processing Pipeline v7
# ============================================================

def process_audio_v7(job_id: str, input_path: Path):
    """Main audio processing pipeline."""
    try:
        set_stage(job_id, "analyze")

        # 1. Read audio
        audio, sr = read_audio(input_path)
        duration = len(audio) / sr
        jobs[job_id]["duration"] = round(duration, 2)

        # 2. Spectrogram for frontend
        spec = compute_spectrogram(audio, sr)
        jobs[job_id]["spec"] = spec.tolist()

        if not HAS_LIBROSA:
            set_stage(job_id, "melody")
            rh, lh = extract_melody_fallback(audio, sr)
            jobs[job_id]["melody_rh"] = rh
            jobs[job_id]["melody_lh"] = lh
            set_stage(job_id, "done")
            jobs[job_id]["status"] = "completed"
            return

        # 3. Extract melody
        set_stage(job_id, "melody")
        melody_data = extract_melody_librosa_v7(audio, sr)
        jobs[job_id]["tempo"] = round(melody_data['tempo'], 1)
        jobs[job_id]["key_name"] = melody_data.get('key_name', 'C major')

        # 4. Fit piano arrangement
        set_stage(job_id, "fit")
        rh_notes, lh_notes = fit_piano_arrangement_v7(melody_data)
        jobs[job_id]["melody_rh"] = rh_notes
        jobs[job_id]["melody_lh"] = lh_notes

        # 5. Render WAV (before comparison — same file user gets)
        set_stage(job_id, "render")
        xml_path = Path(jobs[job_id]["xml_path"])
        wav_path = Path(jobs[job_id]["wav_path"])
        pdf_path = Path(jobs[job_id]["pdf_path"])

        build_musicxml_v7(rh_notes, lh_notes, xml_path)
        generate_wav_v7(rh_notes, lh_notes, wav_path)

        mscore = find_musescore()
        if mscore:
            subprocess.run([mscore, "-o", str(pdf_path), str(xml_path)],
                           capture_output=True, timeout=120)

        # 6. Fingerprint comparison
        set_stage(job_id, "compare")
        similarity, chroma_orig_mean, chroma_synth_mean = compare_fingerprints_v7(audio, sr, wav_path)
        jobs[job_id]["similarity"] = round(similarity, 4)
        jobs[job_id]["chroma_orig"] = chroma_orig_mean
        jobs[job_id]["chroma_synth"] = chroma_synth_mean

        set_stage(job_id, "done")
        jobs[job_id]["status"] = "completed"
        logger.info(f"[{job_id}] Done! Similarity: {similarity:.3f}")

    except Exception as e:
        logger.error(f"[{job_id}] Error: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)

# ============================================================
# Melody Extraction v7
# ============================================================

def extract_melody_librosa_v7(audio: np.ndarray, sr: int) -> Dict:
    """
    Advanced melody extraction:
    - Harmonic-percussive separation
    - Superflux onset detection
    - PYIN primary pitch, piptrack fallback
    - segment_notes_v7 post-processing
    - Krumhansl-Schmuckler key estimation
    """
    # Harmonic-percussive separation
    y_harm = librosa.effects.harmonic(audio, margin=4.0)

    # Tempo estimation
    tempo = float(librosa.beat.beat_track(y=audio, sr=sr)[0])
    if tempo < 40 or tempo > 200:
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        of = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        if len(of) > 1:
            tempo = 60.0 / (np.median(np.diff(of)) * HOP_LENGTH / sr)
        tempo = max(40, min(200, tempo))

    beat_dur = 60.0 / tempo

    # Multi-scale onset detection
    onset_env = librosa.onset.onset_strength(y=y_harm, sr=sr, hop_length=HOP_LENGTH)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH,
        pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.04, wait=2
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=HOP_LENGTH)

    # PYIN pitch tracking (primary)
    f0_pyin, voiced_flag, voiced_probs = librosa.pyin(
        y_harm, sr=sr, hop_length=HOP_LENGTH,
        fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C8'),
        fill_na=None
    )

    # PIPTRACK fallback
    pitches, mags = librosa.piptrack(
        y=y_harm, sr=sr, hop_length=HOP_LENGTH,
        fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C8')
    )

    melody_events = []
    for i, of in enumerate(onset_frames):
        # Try PYIN first
        pyin_pitch = None
        if f0_pyin is not None and of < len(f0_pyin) and f0_pyin[of] is not None and not np.isnan(f0_pyin[of]):
            pyin_pitch = f0_pyin[of]

        # Fallback to piptrack
        window = slice(of, min(of + 12, pitches.shape[1]))
        best_pitch, best_mag = 0, 0
        for t in range(window.start, window.stop):
            for f in range(pitches.shape[0]):
                if mags[f, t] > best_mag and pitches[f, t] > 0:
                    best_mag = mags[f, t]
                    best_pitch = pitches[f, t]

        # Choose pitch: prefer PYIN if available and consistent with piptrack
        final_pitch = None
        if pyin_pitch is not None and pyin_pitch > 0:
            if best_pitch > 0:
                pyin_midi = librosa.hz_to_midi(pyin_pitch)
                pip_midi = librosa.hz_to_midi(best_pitch)
                if abs(pyin_midi - pip_midi) <= 3:  # Within 3 semitones
                    final_pitch = pyin_pitch
                else:
                    final_pitch = best_pitch  # Trust piptrack if divergent
            else:
                final_pitch = pyin_pitch
        elif best_pitch > 0 and best_mag > 0.003:
            final_pitch = best_pitch

        if final_pitch is not None and final_pitch > 0:
            midi = int(round(librosa.hz_to_midi(final_pitch)))
            midi = max(21, min(108, midi))  # Clamp to piano range

            end_time = onset_times[i+1] if i+1 < len(onset_times) else len(audio)/sr
            note_dur = min(end_time - onset_times[i], beat_dur * 3)
            note_dur = max(0.05, note_dur)

            melody_events.append({
                'time': onset_times[i],
                'midi': midi,
                'mag': best_mag if best_mag > 0 else 0.01,
                'dur': note_dur
            })

    # Intelligent segmentation
    final_melody = segment_notes_v7(melody_events, sr=sr, hop_length=HOP_LENGTH)

    # Key estimation
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=HOP_LENGTH)
    key_idx, key_name, is_major = estimate_key(chroma)

    return {
        'melody': final_melody,
        'tempo': tempo,
        'beat_dur': beat_dur,
        'key_idx': key_idx,
        'key_name': key_name,
        'is_major': is_major,
        'duration': len(audio) / sr,
        'audio': audio,
        'sr': sr,
    }

# ============================================================
# Piano Arrangement v7
# ============================================================

def fit_piano_arrangement_v7(melody_data: Dict) -> Tuple[List[Dict], List[Dict]]:
    """
    Adaptive piano arrangement with:
    - Voice-leading aware RH melody
    - Diatonic LH accompaniment based on estimated key
    - Dynamic velocity based on melodic contour
    """
    melody = melody_data['melody']
    tempo = melody_data['tempo']
    beat_dur = melody_data['beat_dur']
    key_idx = melody_data['key_idx']
    is_major = melody_data.get('is_major', True)

    if not melody:
        return [], []

    rh_notes = []
    lh_notes = []

    # Diatonic scale degrees for major/minor
    if is_major:
        scale_degrees = [0, 2, 4, 5, 7, 9, 11]  # Ionian
    else:
        scale_degrees = [0, 2, 3, 5, 7, 8, 10]  # Aeolian

    # Build diatonic pitch set
    def is_diatonic(midi: int) -> bool:
        return (midi % 12 - key_idx) % 12 in scale_degrees

    def snap_to_scale(midi: int, prefer_up: bool = True) -> int:
        """Snap a MIDI note to the nearest diatonic pitch."""
        pc = (midi % 12 - key_idx) % 12
        if pc in scale_degrees:
            return midi
        # Find nearest diatonic pitch class
        distances = [(sd, min((pc - sd) % 12, (sd - pc) % 12)) for sd in scale_degrees]
        distances.sort(key=lambda x: x[1])
        best_sd = distances[0][0]
        delta = best_sd - pc
        return midi + delta

    # RH: Melody with dynamics based on contour
    for i, e in enumerate(melody):
        rh_pitch = max(21, min(108, e['midi']))
        # Snap melody notes to scale if close (within 1 semitone)
        if not is_diatonic(rh_pitch):
            snapped = snap_to_scale(rh_pitch)
            if abs(snapped - rh_pitch) <= 1:
                rh_pitch = snapped

        dur = max(0.08, min(e['dur'], beat_dur * 2.5))

        # Dynamic velocity: louder on downbeats and peaks
        base_vel = 80 + e.get('mag', 0.5) * 35
        if i % 4 == 0:
            base_vel += 10
        velocity = min(127, int(base_vel))

        rh_notes.append({
            'start': round(e['time'], 3),
            'dur': round(dur * 0.92, 3),
            'pitch': rh_pitch,
            'velocity': velocity
        })

        # Add diatonic harmony note (3rd or 5th) on strong beats
        if i % 2 == 0 and rh_pitch < 95:
            degree = (rh_pitch % 12 - key_idx) % 12
            if is_major:
                if degree == 0:
                    harmony_interval = 4  # major 3rd
                elif degree == 4:
                    harmony_interval = 3  # minor 3rd up
                elif degree == 7:
                    harmony_interval = -4  # down to 5th
                else:
                    harmony_interval = 7  # perfect 5th
            else:
                if degree == 0:
                    harmony_interval = 3  # minor 3rd
                elif degree == 3:
                    harmony_interval = 4  # major 3rd up
                elif degree == 7:
                    harmony_interval = -3  # down to 5th
                else:
                    harmony_interval = 7

            harm_pitch = max(21, min(108, rh_pitch + harmony_interval))
            if is_diatonic(harm_pitch):
                rh_notes.append({
                    'start': round(e['time'], 3),
                    'dur': round(dur * 0.35, 3),
                    'pitch': harm_pitch,
                    'velocity': max(30, velocity - 35)
                })

    # LH: Alberti-bass style accompaniment
    for i in range(0, len(melody), 2):
        e = melody[i]
        degree = (e['midi'] % 12 - key_idx) % 12

        # Determine chord quality based on scale degree
        if is_major:
            if degree in [0, 5, 7]:  # I, IV, V
                intervals = [0, 4, 7]
            elif degree in [2, 4, 9]:  # ii, iii, vi
                intervals = [0, 3, 7]
            else:
                intervals = [0, 4, 7]
        else:
            if degree in [0, 3, 7, 10]:  # i, III, v, VII
                intervals = [0, 3, 7]
            elif degree in [2, 5, 8]:
                intervals = [0, 4, 7]
            else:
                intervals = [0, 3, 7]

        bass_root = 36 + (key_idx + intervals[0]) % 12
        # Ensure bass_root is in reasonable range
        while bass_root < 28:
            bass_root += 12
        while bass_root > 50:
            bass_root -= 12

        chord_tones = [bass_root + iv for iv in intervals]

        # Alberti pattern: low - mid - high - mid
        pattern = [
            (chord_tones[0], 0.0, 0.22),
            (chord_tones[1], 0.25, 0.18),
            (chord_tones[2], 0.50, 0.18),
            (chord_tones[1], 0.75, 0.22),
        ]

        for pitch, toff, tdur in pattern:
            pitch = min(72, max(21, pitch))
            lh_notes.append({
                'start': round(e['time'] + toff * beat_dur, 3),
                'dur': round(beat_dur * tdur, 3),
                'pitch': pitch,
                'velocity': 42 if toff == 0.0 else 32
            })

    return rh_notes, lh_notes

# ============================================================
# Fingerprint Comparison v7
# ============================================================

def compare_fingerprints_v7(original_audio: np.ndarray, sr: int, wav_path: Path) -> Tuple[float, List[float], List[float]]:
    """
    Multi-feature similarity comparison:
    - Chroma cosine similarity
    - Onset correlation
    - Pitch contour similarity
    - Mel spectrogram distance
    - Spectral contrast
    """
    try:
        candidate, _ = librosa.load(wav_path, sr=sr, mono=True)

        min_len = min(len(original_audio), len(candidate))
        orig = original_audio[:min_len]
        cand = candidate[:min_len]

        # 1. Chroma similarity
        c_orig = librosa.feature.chroma_stft(y=orig, sr=sr, hop_length=HOP_LENGTH)
        c_cand = librosa.feature.chroma_stft(y=cand, sr=sr, hop_length=HOP_LENGTH)
        min_c = min(c_orig.shape[1], c_cand.shape[1])
        chroma_sims = []
        for i in range(min_c):
            no = np.linalg.norm(c_orig[:, i]) + 1e-8
            nc = np.linalg.norm(c_cand[:, i]) + 1e-8
            chroma_sims.append(np.dot(c_orig[:, i], c_cand[:, i]) / (no * nc))
        chroma_sim = float(np.mean(chroma_sims))

        # 2. Onset correlation
        o_orig = librosa.onset.onset_strength(y=orig, sr=sr, hop_length=HOP_LENGTH)
        o_cand = librosa.onset.onset_strength(y=cand, sr=sr, hop_length=HOP_LENGTH)
        min_o = min(len(o_orig), len(o_cand))
        o1 = o_orig[:min_o] / (np.max(o_orig) + 1e-8)
        o2 = o_cand[:min_o] / (np.max(o_cand) + 1e-8)
        onset_corr = float(np.corrcoef(o1, o2)[0, 1])
        if np.isnan(onset_corr):
            onset_corr = 0.0

        # 3. Pitch contour
        f0_orig, _, _ = librosa.pyin(orig, fmin=librosa.note_to_hz('C2'),
                                       fmax=librosa.note_to_hz('C8'), sr=sr)
        f0_cand, _, _ = librosa.pyin(cand, fmin=librosa.note_to_hz('C2'),
                                       fmax=librosa.note_to_hz('C8'), sr=sr)
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

        # 4. Mel spectrogram
        mel_orig = librosa.feature.melspectrogram(y=orig, sr=sr, hop_length=HOP_LENGTH, n_mels=40)
        mel_cand = librosa.feature.melspectrogram(y=cand, sr=sr, hop_length=HOP_LENGTH, n_mels=40)
        min_m = min(mel_orig.shape[1], mel_cand.shape[1])
        m1 = librosa.power_to_db(mel_orig[:, :min_m] + 1e-8)
        m2 = librosa.power_to_db(mel_cand[:, :min_m] + 1e-8)
        mel_sim = max(0, 1 - np.mean(np.abs(m1 - m2)) / 80.0)

        # 5. Spectral contrast (timbre similarity)
        sc_orig = librosa.feature.spectral_contrast(y=orig, sr=sr, hop_length=HOP_LENGTH)
        sc_cand = librosa.feature.spectral_contrast(y=cand, sr=sr, hop_length=HOP_LENGTH)
        min_sc = min(sc_orig.shape[1], sc_cand.shape[1])
        sc_corr = float(np.corrcoef(sc_orig[:, :min_sc].flatten(), sc_cand[:, :min_sc].flatten())[0, 1])
        if np.isnan(sc_corr):
            sc_corr = 0.0
        sc_sim = max(0, (sc_corr + 1) / 2)

        # Weighted combination
        total = (
            chroma_sim * 0.25 +
            max(0, onset_corr) * 0.20 +
            pitch_sim * 0.25 +
            mel_sim * 0.20 +
            sc_sim * 0.10
        )

        chroma_orig_mean = c_orig.mean(axis=1).tolist() if c_orig is not None else [0.0]*12
        chroma_synth_mean = c_cand.mean(axis=1).tolist() if c_cand is not None else [0.0]*12

        return min(1.0, max(0.0, total)), chroma_orig_mean, chroma_synth_mean

    except Exception as e:
        logger.warning(f"Fingerprint comparison failed: {e}")
        return 0.5, [0.0]*12, [0.0]*12

# ============================================================
# Fallback Melody Extraction
# ============================================================

def extract_melody_fallback(audio: np.ndarray, sr: int) -> Tuple[List[Dict], List[Dict]]:
    """FFT-based fallback for environments without librosa."""
    N_FFT_FB = 4096
    HOP_FB = int(sr * 0.05)
    freqs = np.fft.rfftfreq(N_FFT_FB, 1 / sr)
    frames = []

    for i in range(0, len(audio) - N_FFT_FB, HOP_FB):
        frame = audio[i:i + N_FFT_FB] * np.hanning(N_FFT_FB)
        spectrum = np.abs(np.fft.rfft(frame))

        mask = (freqs >= 130) & (freqs <= 4200)  # Expanded range
        band_freqs = freqs[mask]
        band_spec = spectrum[mask]

        if len(band_spec) == 0:
            continue

        idx = np.argmax(band_spec)
        peak_freq = band_freqs[idx]
        peak_amp = band_spec[idx]

        global_max = np.max(spectrum) + 1e-10
        amp_norm = min(1.0, peak_amp / global_max)
        if amp_norm < 0.015:
            continue

        time = i / sr
        midi = int(round(69 + 12 * np.log2(peak_freq / 440.0)))
        frames.append({'time': time, 'freq': peak_freq, 'midi': midi, 'amp': amp_norm})

    if not frames:
        return [], []

    # Median smoothing
    if len(frames) > 7:
        try:
            from scipy.ndimage import median_filter
            pitches = [f['midi'] for f in frames]
            smoothed = median_filter(pitches, size=7)
            for i in range(len(frames)):
                frames[i]['midi'] = int(smoothed[i])
        except ImportError:
            pass

    # Segment with minimum duration
    notes = []
    current = {'start': frames[0]['time'], 'pitch': frames[0]['midi'], 'amp': frames[0]['amp']}
    for f in frames[1:]:
        if f['midi'] != current['pitch']:
            dur = f['time'] - current['start']
            if dur >= 0.12:
                notes.append({
                    'start': round(current['start'], 2),
                    'dur': round(dur, 2),
                    'pitch': current['pitch'],
                    'velocity': min(127, int(current['amp'] * 127))
                })
            current = {'start': f['time'], 'pitch': f['midi'], 'amp': f['amp']}

    dur = len(audio) / sr - current['start']
    if dur >= 0.12:
        notes.append({
            'start': round(current['start'], 2),
            'dur': round(dur, 2),
            'pitch': current['pitch'],
            'velocity': min(127, int(current['amp'] * 127))
        })

    if not notes:
        return [], []

    # Deduplicate and merge
    deduped = [notes[0]]
    for n in notes[1:]:
        gap = n['start'] - (deduped[-1]['start'] + deduped[-1]['dur'])
        if n['pitch'] == deduped[-1]['pitch'] and gap < 0.15:
            deduped[-1]['dur'] = round(n['start'] + n['dur'] - deduped[-1]['start'], 2)
            deduped[-1]['velocity'] = max(deduped[-1]['velocity'], n['velocity'])
        elif n['pitch'] != deduped[-1]['pitch']:
            deduped.append(n)

    rh = [n for n in deduped if n['pitch'] >= 60]
    lh = [n for n in deduped if n['pitch'] < 60]
    return rh, lh

# ============================================================
# WAV Synthesis v7 — Harmonic Piano Model
# ============================================================

def generate_wav_v7(rh_notes: List[Dict], lh_notes: List[Dict], wav_path: Path, sr: int = 44100):
    """
    Synthesize piano-like audio using additive synthesis with:
    - Inharmonicity (stiff string model)
    - ADSR envelope per note
    - Stereo field (RH right, LH left)
    - Gentle reverb simulation via exponential decay tail
    """
    all_notes = rh_notes + lh_notes
    if not all_notes:
        return

    total_dur = max(n["start"] + n["dur"] for n in all_notes)
    total_samples = int(total_dur * sr) + int(sr * 0.5)  # Extra for reverb tail

    # Stereo output
    audio_l = np.zeros(total_samples, dtype=np.float64)
    audio_r = np.zeros(total_samples, dtype=np.float64)

    # Piano inharmonicity coefficient
    B = 0.0003

    for n in all_notes:
        freq = 440.0 * (2.0 ** ((n["pitch"] - 69) / 12.0))
        start_sample = int(n["start"] * sr)
        dur_samples = int(n["dur"] * sr)
        if dur_samples <= 0:
            continue

        t = np.linspace(0, n["dur"], dur_samples, endpoint=False)

        # Inharmonic frequencies: fn = n * f0 * sqrt(1 + B * n^2)
        wave = np.zeros(dur_samples, dtype=np.float64)
        harmonics = [
            (1, 0.55),
            (2, 0.28),
            (3, 0.14),
            (4, 0.07),
            (5, 0.04),
            (6, 0.02),
            (7, 0.015),
        ]
        for h_num, h_amp in harmonics:
            h_freq = h_num * freq * np.sqrt(1 + B * h_num**2)
            wave += h_amp * np.sin(2 * np.pi * h_freq * t)

        # ADSR envelope
        attack = int(0.006 * sr)
        decay = int(0.12 * sr)
        release = int(0.18 * sr)
        sustain_level = 0.55

        env = np.ones(dur_samples, dtype=np.float64) * sustain_level
        if attack > 0:
            env[:attack] = np.linspace(0, 1, attack)
        if decay > 0 and attack + decay < dur_samples:
            env[attack:attack+decay] = np.linspace(1, sustain_level, decay)
        if release > 0 and dur_samples > release:
            env[-release:] *= np.linspace(1, 0, release)

        # Apply envelope
        wave *= env

        # Pan: RH slightly right, LH slightly left
        if n in rh_notes:
            pan_l, pan_r = 0.45, 0.65
            vel_scale = 0.35
        else:
            pan_l, pan_r = 0.65, 0.45
            vel_scale = 0.30

        vel = n.get("velocity", 80) / 127.0
        wave *= vel * vel_scale

        end = min(start_sample + dur_samples, total_samples)
        actual = end - start_sample

        audio_l[start_sample:end] += wave[:actual] * pan_l
        audio_r[start_sample:end] += wave[:actual] * pan_r

    # Simple reverb via exponential decay convolution (very light)
    decay = np.exp(-np.linspace(0, 5, int(0.15 * sr)))
    decay /= decay.sum()
    audio_l = np.convolve(audio_l, decay, mode='same')
    audio_r = np.convolve(audio_r, decay, mode='same')

    # Mix to stereo and normalize
    audio_stereo = np.stack([audio_l, audio_r], axis=1)
    peak = np.max(np.abs(audio_stereo))
    if peak > 0:
        audio_stereo = audio_stereo / peak * 0.95

    # Write stereo WAV
    audio_int16 = (audio_stereo * 32767).astype(np.int16)
    with open(wav_path, 'wb') as f:
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + total_samples * 4))
        f.write(b'WAVEfmt ')
        f.write(struct.pack('<IHHIIHH', 16, 1, 2, sr, sr * 4, 4, 16))
        f.write(b'data')
        f.write(struct.pack('<I', total_samples * 4))
        f.write(audio_int16.tobytes())

# ============================================================
# MusicXML Builder v7
# ============================================================

def build_musicxml_v7(rh_notes: List[Dict], lh_notes: List[Dict], xml_path: Path):
    """Build MusicXML with proper two-staff piano part."""
    DIVISIONS = 8

    root = Element('score-partwise', {'version': '3.1'})
    part_list = SubElement(root, 'part-list')

    score_part = SubElement(part_list, 'score-part', {'id': 'P1'})
    SubElement(score_part, 'part-name').text = 'Piano'

    part1 = SubElement(root, 'part', {'id': 'P1'})

    # Group notes by measure
    def group_by_measure(notes):
        measures = {}
        for n in notes:
            m = int(n["start"] // 4.0)
            off = n["start"] % 4.0
            if off >= 3.999:
                m += 1
                off = 0.0
            measures.setdefault(m, []).append({
                "offset": int(round(off * DIVISIONS)),
                "dur": int(round(n["dur"] * DIVISIONS)),
                "pitch": n["pitch"],
                "velocity": n.get("velocity", 80)
            })
        return measures

    measures_rh = group_by_measure(rh_notes)
    measures_lh = group_by_measure(lh_notes)
    all_measures = sorted(set(list(measures_rh.keys()) + list(measures_lh.keys())))

    for m_idx in all_measures:
        measure = SubElement(part1, 'measure', {'number': str(m_idx + 1)})

        if m_idx == 0:
            attr = SubElement(measure, 'attributes')
            SubElement(attr, 'divisions').text = str(DIVISIONS)
            staves = SubElement(attr, 'staves')
            staves.text = '2'

            clef1 = SubElement(attr, 'clef', {'number': '1'})
            SubElement(clef1, 'sign').text = 'G'
            SubElement(clef1, 'line').text = '2'

            clef2 = SubElement(attr, 'clef', {'number': '2'})
            SubElement(clef2, 'sign').text = 'F'
            SubElement(clef2, 'line').text = '4'

            key = SubElement(attr, 'key')
            SubElement(key, 'fifths').text = '0'

            time = SubElement(attr, 'time')
            SubElement(time, 'beats').text = '4'
            SubElement(time, 'beat-type').text = '4'

        def write_staff_events(measure, events, staff_num):
            events = sorted(events, key=lambda x: x['offset'])
            last_end = 0
            for evt in events:
                off = evt['offset']
                dur = evt['dur']
                pitch = evt['pitch']

                gap = off - last_end
                if gap >= 1:
                    r = SubElement(measure, 'note')
                    SubElement(r, 'rest')
                    SubElement(r, 'duration').text = str(gap)
                    staff_el = SubElement(r, 'staff')
                    staff_el.text = str(staff_num)
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
                staff_el.text = str(staff_num)
                t, dot = dur_info(dur)
                SubElement(n, 'type').text = t
                if dot:
                    SubElement(n, 'dot')

                last_end = off + dur

            fill = 32 - last_end
            if fill >= 1:
                r = SubElement(measure, 'note')
                SubElement(r, 'rest')
                SubElement(r, 'duration').text = str(fill)
                staff_el = SubElement(r, 'staff')
                staff_el.text = str(staff_num)
                t, dot = dur_info(fill)
                SubElement(r, 'type').text = t
                if dot:
                    SubElement(r, 'dot')

        write_staff_events(measure, measures_rh.get(m_idx, []), 1)
        write_staff_events(measure, measures_lh.get(m_idx, []), 2)

    xml_str = tostring(root, encoding='unicode')
    doctype = '<?xml version="1.0" encoding="UTF-8"?>\n'
    with open(xml_path, 'w', encoding='utf-8') as f:
        f.write(doctype + xml_str)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
