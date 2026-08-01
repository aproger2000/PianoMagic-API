import os
import uuid
import shutil
import subprocess
import logging
import json
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PianoMagic API", version="4.1.1")

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
MIN_AMP = 0.05
NOTE_THRESHOLD = 0.15
MELODY_MIN_FREQ = 50
MELODY_MAX_FREQ = 1500
DIVISIONS = 2

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

STAGES = {
    "upload":    {"text": "Загрузка файла...",          "progress": 10},
    "read":      {"text": "Чтение аудио (ffmpeg)...",   "progress": 20},
    "fft":       {"text": "FFT-анализ (50мс фреймы)...","progress": 40},
    "harmonic":  {"text": "Подавление гармоник...",     "progress": 55},
    "smooth":    {"text": "Сглаживание мелодии...",     "progress": 70},
    "quantize":  {"text": "Квантизация нот...",         "progress": 85},
    "done":      {"text": "Готово!",                    "progress": 100},
}

avg_pitch = sum(n["pitch"] for n in notes) / len(notes)
clef_sign = "F" if avg_pitch < 55 else "G"
clef_line = "4" if avg_pitch < 55 else "2"

@app.get("/")
async def root():
    return {"status": "ok", "version": "4.1.1"}


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
        "melody": None,
        "fft": None,
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

    mscore = find_musescore()
    if not mscore:
        raise RuntimeError("MuseScore не найден")
    result = subprocess.run(
        [mscore, "-o", str(pdf_path), str(xml_path)],
        capture_output=True, text=True, timeout=120
    )

    try:
        build_musicxml(jobs[job_id]["melody"], xml_path)
        result = subprocess.run(
            ["musescore3", "-o", str(pdf_path), str(xml_path)],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            raise RuntimeError(f"MuseScore3 error: {result.stderr or result.stdout}")
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


def set_stage(job_id, stage):
    if job_id in jobs:
        jobs[job_id]["stage"] = stage
        logger.info(f"[{job_id}] Stage: {stage}")

def find_musescore():
    for cmd in ["musescore3", "musescore", "mscore"]:
        if shutil.which(cmd):
            return cmd
    return None

def read_audio(path):
    cmd = ['ffmpeg', '-y', '-i', str(path), '-ar', str(SR), '-ac', '1', '-f', 'f32le', '-']
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {result.stderr.decode()[:200]}")
    return np.frombuffer(result.stdout, dtype=np.float32)


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
    if dur >= 8: return 'whole', False
    if dur == 6: return 'half', True
    if dur >= 4: return 'half', False
    if dur == 3: return 'quarter', True
    if dur >= 2: return 'quarter', False
    return 'eighth', False


def find_fundamental(spectrum, freqs, min_f, max_f):
    mask = (freqs >= min_f) & (freqs <= max_f)
    mel_freqs = freqs[mask]
    mel_spec = spectrum[mask]

    if len(mel_spec) == 0:
        return None, 0

    peak_indices = []
    spec_copy = mel_spec.copy()
    for _ in range(5):
        if len(spec_copy) == 0:
            break
        idx = np.argmax(spec_copy)
        peak_indices.append(idx)
        left = max(0, idx - 3)
        right = min(len(spec_copy), idx + 4)
        spec_copy[left:right] = 0

    candidates = []
    for idx in peak_indices:
        freq = mel_freqs[idx]
        amp = mel_spec[idx]
        is_harmonic = False
        for div in [2, 3, 4]:
            fundamental = freq / div
            if fundamental < 30:          # ← разрешаем искать фундаментал даже очень низко
                continue
            f_idx = np.argmin(np.abs(mel_freqs - fundamental))
            if f_idx < len(mel_spec) and mel_spec[f_idx] > amp * 0.15:
                is_harmonic = True
                candidates.append((mel_freqs[f_idx], mel_spec[f_idx]))
                break
        if not is_harmonic:
            candidates.append((freq, amp))

    if not candidates:
        return None, 0
    best = max(candidates, key=lambda x: x[1])
    return best[0], best[1]


def process_audio(job_id: str, input_path: Path):
    try:
        set_stage(job_id, "read")
        audio = read_audio(input_path)
        duration = len(audio) / SR

        if len(audio) < N_FFT:
            raise ValueError("Аудио слишком короткое")

        set_stage(job_id, "fft")
        freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
        frames = []

        for i in range(0, len(audio) - N_FFT, HOP):
            frame = audio[i:i + N_FFT] * np.hanning(N_FFT)
            spectrum = np.abs(np.fft.rfft(frame))
            peak_freq, peak_amp = find_fundamental(spectrum, freqs, MELODY_MIN_FREQ, MELODY_MAX_FREQ)
            if peak_freq is None:
                continue

            global_max = np.max(spectrum) + 1e-10
            amp_norm = min(1.0, peak_amp / global_max)
            time = i / SR
            frames.append({
                "time": round(time, 3),
                "freq": round(peak_freq, 1),
                "amp": round(amp_norm, 3),
                "midi": int(round(freq_to_midi(peak_freq)))
            })

        if not frames:
            raise ValueError("Не удалось извлечь мелодию")

        set_stage(job_id, "harmonic")
        melody_raw = [f for f in frames if f["amp"] >= MIN_AMP]

        # Медианный фильтр
        if len(melody_raw) > 7:
            pitches = [m["midi"] for m in melody_raw]
            smoothed = []
            for i in range(len(pitches)):
                window = pitches[max(0, i - 3):min(len(pitches), i + 4)]
                smoothed.append(int(np.median(window)))
            for i, m in enumerate(melody_raw):
                m["midi"] = smoothed[i]

        # Гистерезис
        HYSTERESIS_MS = 0.15
        min_frames = int(HYSTERESIS_MS / (HOP_MS / 1000))
        filtered = []
        if melody_raw:
            run_start = 0
            run_pitch = melody_raw[0]["midi"]
            for i in range(1, len(melody_raw)):
                if melody_raw[i]["midi"] != run_pitch:
                    run_len = i - run_start
                    if run_len >= min_frames:
                        for j in range(run_start, i):
                            filtered.append(melody_raw[j])
                    run_start = i
                    run_pitch = melody_raw[i]["midi"]
            run_len = len(melody_raw) - run_start
            if run_len >= min_frames:
                for j in range(run_start, len(melody_raw)):
                    filtered.append(melody_raw[j])
        melody_raw = filtered

        set_stage(job_id, "smooth")
        notes = []
        if melody_raw:
            current = {"start": melody_raw[0]["time"], "pitch": melody_raw[0]["midi"], "amp": melody_raw[0]["amp"]}
            for m in melody_raw[1:]:
                if m["midi"] != current["pitch"]:
                    dur = m["time"] - current["start"]
                    if dur >= NOTE_THRESHOLD:
                        notes.append({
                            "start": round(current["start"], 2),
                            "dur": round(dur, 2),
                            "pitch": current["pitch"],
                            "velocity": min(127, int(current["amp"] * 127))
                        })
                    current = {"start": m["time"], "pitch": m["midi"], "amp": m["amp"]}
            dur = duration - current["start"]
            if dur >= NOTE_THRESHOLD:
                notes.append({
                    "start": round(current["start"], 2),
                    "dur": round(dur, 2),
                    "pitch": current["pitch"],
                    "velocity": min(127, int(current["amp"] * 127))
                })

        if not notes:
            raise ValueError("Мелодия не выделена")
            
# Опционально: ограничить диапазон пианино A0–C8, но не сдвигать октавы
        for n in notes:
            n["pitch"] = max(21, min(108, n["pitch"]))

        deduped = [notes[0]]
        for n in notes[1:]:
            if n["pitch"] != deduped[-1]["pitch"]:
                deduped.append(n)
        notes = deduped

        set_stage(job_id, "quantize")
         spq = 60.0 / bpm  # bpm можно передавать с фронта, по умолчанию 120

        quantized = []
        for n in notes:
               # шаг 0.25 вместо 0.5 для более точной ритмики
            qs = round((n["start"] / spq) / 0.25) * 0.25
            qdur = max(0.25, round((n["dur"] / spq) / 0.25) * 0.25)
            qdur = min(4.0, qdur)
            quantized.append({
                "start": qs,
                "dur": qdur,
                "pitch": n["pitch"],
                "velocity": n["velocity"]
            })

        set_stage(job_id, "done")
        jobs[job_id]["melody"] = quantized
        jobs[job_id]["fft"] = frames[::max(1, len(frames) // 300)]
        jobs[job_id]["duration"] = round(duration, 1)
        jobs[job_id]["status"] = "analyzed"
        logger.info(f"[{job_id}] Готово")

    except Exception as e:
        logger.error(f"[{job_id}] Ошибка: {e}", exc_info=True)
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
    finally:
        if input_path.exists():
            input_path.unlink()


def build_musicxml(notes, xml_path):
    root = Element('score-partwise', {'version': '3.1'})
    plist = SubElement(root, 'part-list')
    sp = SubElement(plist, 'score-part', {'id': 'P1'})
    SubElement(sp, 'part-name').text = 'Melody'

    # в build_musicxml:
    c = SubElement(attr, 'clef')
    SubElement(c, 'sign').text = clef_sign
    SubElement(c, 'line').text = clef_line

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
            SubElement(c, 'sign').text = 'G'
            SubElement(c, 'line').text = '2'

        events = sorted(measures[m_idx], key=lambda x: x['offset'])
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

        fill = 8 - last_end
        if fill >= 1:
            r = SubElement(measure, 'note')
            SubElement(r, 'rest')
            SubElement(r, 'duration').text = str(fill)
            t, dot = dur_info(fill)
            SubElement(r, 'type').text = t
            if dot:
                SubElement(r, 'dot')

    xml_str = tostring(root, encoding='unicode')
    doctype = '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">\n'
    with open(xml_path, 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n' + doctype + xml_str)
