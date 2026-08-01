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

app = FastAPI(title="PianoMagic API", version="4.0.0")

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
N_FFT = 2048
MIN_AMP = 0.02
NOTE_THRESHOLD = 0.08
MELODY_MIN_FREQ = 250
MELODY_MAX_FREQ = 1050
DIVISIONS = 2

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


@app.get("/")
async def root():
    return {"status": "ok", "version": "4.0.0"}


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
    return {"job_id": job_id, "status": jobs[job_id]["status"], "error": jobs[job_id].get("error")}


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


def process_audio(job_id: str, input_path: Path):
    try:
        logger.info(f"[{job_id}] === FFT анализ v4.0.0 ===")
        audio = read_audio(input_path)
        duration = len(audio) / SR
        logger.info(f"[{job_id}] Аудио: {duration:.1f}s")

        if len(audio) < N_FFT:
            raise ValueError("Аудио слишком короткое")

        freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
        mask = (freqs >= MELODY_MIN_FREQ) & (freqs <= MELODY_MAX_FREQ)
        mel_freqs = freqs[mask]

        frames = []
        for i in range(0, len(audio) - N_FFT, HOP):
            frame = audio[i:i + N_FFT] * np.hanning(N_FFT)
            spectrum = np.abs(np.fft.rfft(frame))
            mel_spectrum = spectrum[mask]

            if len(mel_spectrum) == 0:
                continue

            peak_idx = np.argmax(mel_spectrum)
            peak_freq = float(mel_freqs[peak_idx])
            peak_amp = float(mel_spectrum[peak_idx])
            global_max = np.max(spectrum) + 1e-10
            amp_norm = min(1.0, peak_amp / global_max)

            time = i / SR
            frames.append({
                "time": round(time, 3),
                "freq": round(peak_freq, 1),
                "amp": round(amp_norm, 3),
                "midi": int(round(freq_to_midi(peak_freq)))
            })

        logger.info(f"[{job_id}] Фреймов: {len(frames)}")

        if not frames:
            raise ValueError("Не удалось извлечь мелодию")

        # Фильтр по амплитуде
        melody_raw = [f for f in frames if f["amp"] >= MIN_AMP]

        # Медианный фильтр (окно 5)
        if len(melody_raw) > 5:
            pitches = [m["midi"] for m in melody_raw]
            smoothed = []
            for i in range(len(pitches)):
                window = pitches[max(0, i - 2):min(len(pitches), i + 3)]
                smoothed.append(int(np.median(window)))
            for i, m in enumerate(melody_raw):
                m["midi"] = smoothed[i]

        # Группировка в ноты
        notes = []
        if melody_raw:
            current = {"start": melody_raw[0]["time"], "pitch": melody_raw[0]["midi"], "amp": melody_raw[0]["amp"]}
            for m in melody_raw[1:]:
                if m["midi"] == current["pitch"]:
                    current["amp"] = max(current["amp"], m["amp"])
                else:
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

        logger.info(f"[{job_id}] Нот: {len(notes)}")

        if not notes:
            raise ValueError("Мелодия не выделена")

        # Диапазон C4–C6
        for n in notes:
            while n["pitch"] < 60:
                n["pitch"] += 12
            while n["pitch"] > 84:
                n["pitch"] -= 12

        # Квантизация
        spq = 60.0 / 120
        quantized = []
        for n in notes:
            qs = round((n["start"] / spq) / 0.5) * 0.5
            qdur = max(0.5, round((n["dur"] / spq) / 0.5) * 0.5)
            qdur = min(4.0, qdur)
            quantized.append({
                "start": qs,
                "dur": qdur,
                "pitch": n["pitch"],
                "velocity": n["velocity"]
            })

        jobs[job_id]["melody"] = quantized
        jobs[job_id]["fft"] = frames[::max(1, len(frames) // 200)]
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
