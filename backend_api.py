import os
import uuid
import shutil
import subprocess
import logging
import struct
import math
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from scipy import signal
from scipy.ndimage import median_filter

# Try to import librosa; fallback to custom FFT if unavailable
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    logging.warning("librosa not available, falling back to custom FFT")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PianoMagic API", version="4.2.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://aproger2000.github.io",
        "https://aproger2000.github.io/PianoMagic",
        "http://localhost:8080",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS_DIR = Path("/tmp/pianomagic_jobs")
JOBS_DIR.mkdir(exist_ok=True)
jobs = {}

SR = 22050
HOP_MS = 50
HOP = int(SR * HOP_MS / 1000)
N_FFT = 4096
MIN_AMP = 0.03
NOTE_THRESHOLD = 0.15
MELODY_MIN_FREQ = 200
MELODY_MAX_FREQ = 1500
HPF_CUTOFF = 250
DIVISIONS = 8

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

STAGES = {
    "upload": {"text": "Загрузка файла...", "progress": 10},
    "read": {"text": "Чтение аудио (ffmpeg)...", "progress": 20},
    "fft": {"text": "FFT-анализ (50мс фреймы)...", "progress": 40},
    "harmonic": {"text": "Подавление гармоник...", "progress": 55},
    "smooth": {"text": "Сглаживание мелодии...", "progress": 70},
    "quantize": {"text": "Квантизация нот...", "progress": 85},
    "render": {"text": "Рендер WAV/PDF...", "progress": 95},
    "done": {"text": "Готово!", "progress": 100},
}


@app.get("/")
async def root():
    return {"status": "ok", "version": "4.2.3"}


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
        "melody": None,
        "fft": None,
        "spec": None,
        "duration": 0,
        "error": None
    }
    background_tasks.add_task(process_audio, job_id, input_path)
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
        "error": j.get("error"),
    }


@app.get("/melody/{job_id}")
async def get_melody(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    if jobs[job_id]["melody"] is None:
        raise HTTPException(status_code=400, detail="Melody not ready")
    return JSONResponse({
        "melody": jobs[job_id]["melody"],
        "fft": jobs[job_id]["fft"],
        "spec": jobs[job_id]["spec"],
        "duration": jobs[job_id]["duration"]
    })


@app.post("/render/{job_id}")
async def render_pdf(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    if jobs[job_id]["melody"] is None:
        raise HTTPException(status_code=400, detail="Melody not ready")

    xml_path = Path(jobs[job_id]["xml_path"])
    pdf_path = Path(jobs[job_id]["pdf_path"])
    wav_path = Path(jobs[job_id]["wav_path"])

    try:
        build_musicxml(jobs[job_id]["melody"], xml_path)
        generate_wav(jobs[job_id]["melody"], wav_path)

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


def set_stage(job_id, stage):
    if job_id in jobs:
        jobs[job_id]["stage"] = stage
        logger.info(f"[{job_id}] Stage: {stage}")


def read_audio(path):
    cmd = ['ffmpeg', '-y', '-i', str(path), '-ar', str(SR), '-ac', '1', '-f', 'f32le', '-']
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {result.stderr.decode()[:200]}")
    return np.frombuffer(result.stdout, dtype=np.float32)


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


def extract_melody_pyin(audio, sr):
    """Use librosa.pyin for robust pitch tracking, with HPSS pre-processing."""
    try:
        # HPSS: separate harmonic content
        y_harmonic, y_percussive = librosa.effects.hpss(audio)
        
        # pYIN on harmonic part
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y_harmonic,
            fmin=librosa.note_to_hz('C3'),
            fmax=librosa.note_to_hz('C6'),
            sr=sr,
            frame_length=2048,
            hop_length=int(sr * HOP_MS / 1000)
        )
        
        times = librosa.times_like(f0, sr=sr, hop_length=int(sr * HOP_MS / 1000))
        
        frames = []
        for t, freq, flag in zip(times, f0, voiced_flag):
            if flag and not np.isnan(freq) and freq > 0:
                midi = int(round(librosa.hz_to_midi(freq)))
                # Clamp to reasonable range
                if 48 <= midi <= 96:  # C3-C7
                    frames.append({
                        "time": round(float(t), 3),
                        "freq": round(float(freq), 1),
                        "midi": midi,
                        "amp": 0.5  # pyin doesn't give amplitude
                    })
        return frames
    except Exception as e:
        logger.error(f"pyin failed: {e}")
        return None


def extract_melody_fft(audio, sr):
    """Fallback custom FFT-based extraction."""
    nyq = sr / 2.0
    b, a = signal.butter(4, HPF_CUTOFF / nyq, btype='high', analog=False)
    y_hp = signal.filtfilt(b, a, audio)

    freqs = np.fft.rfftfreq(N_FFT, 1 / sr)
    frames = []

    for i in range(0, len(y_hp) - N_FFT, HOP):
        frame = y_hp[i:i + N_FFT] * np.hanning(N_FFT)
        spectrum = np.abs(np.fft.rfft(frame))

        mask = (freqs >= MELODY_MIN_FREQ) & (freqs <= MELODY_MAX_FREQ)
        mel_freqs = freqs[mask]
        mel_spec = spectrum[mask]

        if len(mel_spec) == 0:
            continue

        # Salience
        salience = np.zeros_like(mel_spec)
        for idx, f in enumerate(mel_freqs):
            s = mel_spec[idx]
            for h in [2, 3, 4]:
                hf = f * h
                if hf > freqs[-1]:
                    break
                h_idx = np.argmin(np.abs(freqs - hf))
                if h_idx < len(spectrum):
                    s += spectrum[h_idx] * (0.4 ** (h - 1))
            salience[idx] = s

        idx = np.argmax(salience)
        peak_freq = mel_freqs[idx]
        peak_amp = mel_spec[idx]

        global_max = np.max(spectrum) + 1e-10
        amp_norm = min(1.0, peak_amp / global_max)
        time = i / sr
        midi = int(round(69 + 12 * np.log2(peak_freq / 440.0)))

        frames.append({
            "time": round(time, 3),
            "freq": round(peak_freq, 1),
            "midi": midi,
            "amp": amp_norm
        })

    return frames


def post_process_melody(frames, duration):
    """Clean up melody: median filter, hysteresis, outlier removal, segmentation."""
    if not frames:
        return []

    # Median filter: 11 frames (~550ms)
    if len(frames) > 11:
        pitches = [f["midi"] for f in frames]
        smoothed = median_filter(pitches, size=11)
        for i, f in enumerate(frames):
            f["midi"] = int(smoothed[i])

    # Hysteresis: min 5 frames (~250ms)
    min_frames = 5
    filtered = []
    run_start = 0
    run_pitch = frames[0]["midi"]
    for i in range(1, len(frames)):
        if frames[i]["midi"] != run_pitch:
            run_len = i - run_start
            if run_len >= min_frames:
                for j in range(run_start, i):
                    filtered.append(frames[j])
            run_start = i
            run_pitch = frames[i]["midi"]
    run_len = len(frames) - run_start
    if run_len >= min_frames:
        for j in range(run_start, len(frames)):
            filtered.append(frames[j])
    frames = filtered

    # Segment into notes
    notes = []
    if frames:
        current = {"start": frames[0]["time"], "pitch": frames[0]["midi"], "amp": frames[0].get("amp", 0.5)}
        for f in frames[1:]:
            if f["midi"] != current["pitch"]:
                dur = f["time"] - current["start"]
                if dur >= NOTE_THRESHOLD:
                    notes.append({
                        "start": round(current["start"], 2),
                        "dur": round(dur, 2),
                        "pitch": current["pitch"],
                        "velocity": min(127, int(current.get("amp", 0.5) * 127))
                    })
                current = {"start": f["time"], "pitch": f["midi"], "amp": f.get("amp", 0.5)}
        dur = duration - current["start"]
        if dur >= NOTE_THRESHOLD:
            notes.append({
                "start": round(current["start"], 2),
                "dur": round(dur, 2),
                "pitch": current["pitch"],
                "velocity": min(127, int(current.get("amp", 0.5) * 127))
            })

    if not notes:
        return []

    # Clamp to piano range
    for n in notes:
        n["pitch"] = max(21, min(108, n["pitch"]))

    # Deduplicate adjacent same pitches
    deduped = [notes[0]]
    for n in notes[1:]:
        if n["pitch"] != deduped[-1]["pitch"]:
            deduped.append(n)

    # Remove outlier short notes surrounded by long same notes
    cleaned = []
    for i, n in enumerate(deduped):
        if i == 0 or i == len(deduped) - 1:
            cleaned.append(n)
            continue
        prev_dur = deduped[i-1]["dur"]
        next_dur = deduped[i+1]["dur"]
        curr_dur = n["dur"]
        # If this note is very short and neighbors are long and same pitch, skip
        if curr_dur < 0.2 and prev_dur > 0.5 and next_dur > 0.5 and abs(deduped[i-1]["pitch"] - deduped[i+1]["pitch"]) <= 1:
            continue
        cleaned.append(n)

    return cleaned


def quantize_notes(notes, bpm=120):
    spq = 60.0 / bpm
    grid = 0.125  # 1/8th note
    quantized = []
    for n in notes:
        qs = round((n["start"] / spq) / grid) * grid
        qdur = max(grid, round((n["dur"] / spq) / grid) * grid)
        qdur = min(4.0, qdur)
        quantized.append({
            "start": qs,
            "dur": qdur,
            "pitch": n["pitch"],
            "velocity": n["velocity"]
        })
    return quantized


def freq_to_midi(freq):
    if freq <= 0:
        return 0
    return 69 + 12 * np.log2(freq / 440.0)


def midi_to_name(midi):
    name = NOTE_NAMES[midi % 12]
    octave = (midi // 12) - 1
    if '#' in name:
        return name[0], 1, octave
    return name, 0, octave


def dur_info(dur):
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


def generate_wav(notes, wav_path, sr=22050):
    if not notes:
        return
    total_dur = max(n["start"] + n["dur"] for n in notes)
    total_samples = int(total_dur * sr) + sr
    audio = np.zeros(total_samples, dtype=np.float32)

    for n in notes:
        freq = 440.0 * (2.0 ** ((n["pitch"] - 69) / 12.0))
        start_sample = int(n["start"] * sr)
        dur_samples = int(n["dur"] * sr)
        if dur_samples <= 0:
            continue

        t = np.linspace(0, n["dur"], dur_samples, endpoint=False)
        wave = 0.6 * np.sin(2 * np.pi * freq * t)
        wave += 0.2 * np.sin(2 * np.pi * freq * 2 * t)
        wave += 0.1 * np.sin(2 * np.pi * freq * 3 * t)

        attack = int(0.02 * sr)
        release = int(0.08 * sr)
        env = np.ones(dur_samples)
        if attack > 0:
            env[:attack] = np.linspace(0, 1, attack)
        if release > 0 and dur_samples > release:
            env[-release:] = np.linspace(1, 0, release)

        end = min(start_sample + dur_samples, total_samples)
        actual = end - start_sample
        audio[start_sample:end] += wave[:actual] * env[:actual] * 0.3

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95

    audio_int16 = (audio * 32767).astype(np.int16)
    with open(wav_path, 'wb') as f:
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + len(audio_int16) * 2))
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))
        f.write(struct.pack('<H', 1))
        f.write(struct.pack('<H', 1))
        f.write(struct.pack('<I', sr))
        f.write(struct.pack('<I', sr * 2))
        f.write(struct.pack('<H', 2))
        f.write(struct.pack('<H', 16))
        f.write(b'data')
        f.write(struct.pack('<I', len(audio_int16) * 2))
        f.write(audio_int16.tobytes())


def process_audio(job_id: str, input_path: Path):
    try:
        set_stage(job_id, "read")
        audio = read_audio(input_path)
        duration = len(audio) / SR

        if len(audio) < N_FFT:
            raise ValueError("Аудио слишком короткое")

        # Spectrogram for visualization
        set_stage(job_id, "fft")
        spec_uint8 = compute_spectrogram(audio, SR)
        spec_list = spec_uint8.tolist()

        # Melody extraction
        set_stage(job_id, "harmonic")
        if HAS_LIBROSA:
            logger.info(f"[{job_id}] Using librosa.pyin + HPSS")
            frames = extract_melody_pyin(audio, SR)
        else:
            logger.info(f"[{job_id}] Using fallback FFT")
            frames = extract_melody_fft(audio, SR)

        if not frames:
            raise ValueError("Не удалось извлечь мелодию")

        set_stage(job_id, "smooth")
        notes = post_process_melody(frames, duration)

        if not notes:
            raise ValueError("Мелодия не выделена")

        set_stage(job_id, "quantize")
        quantized = quantize_notes(notes, bpm=120)

        set_stage(job_id, "render")
        wav_path = Path(jobs[job_id]["wav_path"])
        generate_wav(quantized, wav_path)

        set_stage(job_id, "done")
        jobs[job_id]["melody"] = quantized
        jobs[job_id]["fft"] = frames[::max(1, len(frames) // 300)]
        jobs[job_id]["spec"] = spec_list
        jobs[job_id]["duration"] = round(duration, 1)
        jobs[job_id]["status"] = "analyzed"
        logger.info(f"[{job_id}] Готово: {len(quantized)} notes")

    except Exception as e:
        logger.error(f"[{job_id}] Ошибка: {e}", exc_info=True)
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
    finally:
        if input_path.exists():
            input_path.unlink()


def build_musicxml(notes, xml_path):
    avg_pitch = sum(n["pitch"] for n in notes) / len(notes)
    clef_sign = "F" if avg_pitch < 55 else "G"
    clef_line = "4" if avg_pitch < 55 else "2"

    root = Element('score-partwise', {'version': '3.1'})
    plist = SubElement(root, 'part-list')
    sp = SubElement(plist, 'score-part', {'id': 'P1'})
    SubElement(sp, 'part-name').text = 'Melody'

    part = SubElement(root, 'part', {'id': 'P1'})
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
            "pitch": n["pitch"]
        })

    for m_idx in sorted(measures.keys()):
        measure = SubElement(part, 'measure', {'number': str(m_idx + 1)})
        if m_idx == 0:
            attr = SubElement(measure, 'attributes')
            SubElement(attr, 'divisions').text = str(DIVISIONS)
            k = SubElement(attr, 'key')
            SubElement(k, 'fifths').text = '0'
            t = SubElement(attr, 'time')
            SubElement(t, 'beats').text = '4'
            SubElement(t, 'beat-type').text = '4'
            c = SubElement(attr, 'clef')
            SubElement(c, 'sign').text = clef_sign
            SubElement(c, 'line').text = clef_line

        events = sorted(measures[m_idx], key=lambda x: x['offset'])
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
            t, dot = dur_info(fill)
            SubElement(r, 'type').text = t
            if dot:
                SubElement(r, 'dot')

    xml_str = tostring(root, encoding='unicode')
    doctype = '<?xml version="1.0" encoding="UTF-8"?>'
    with open(xml_path, 'w', encoding='utf-8') as f:
        f.write(doctype + chr(10) + xml_str)
