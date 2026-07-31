import os
import uuid
import shutil
import subprocess
import logging
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PianoMagic API", version="3.4.0")

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

BPM = 120
SEC_PER_QUARTER = 60.0 / BPM
GRID_8TH = 0.5


@app.get("/")
async def root():
    return {"status": "ok", "version": "3.4.0"}


@app.post("/transcribe/file")
async def transcribe_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(file.filename or "input.mp3").suffix
    input_path = job_dir / f"input{ext}"
    xml_path = job_dir / "output.xml"
    pdf_path = job_dir / "output.pdf"

    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    jobs[job_id] = {"status": "processing", "pdf_path": str(pdf_path), "error": None}
    background_tasks.add_task(process_audio, job_id, input_path, xml_path, pdf_path)

    return {"job_id": job_id, "status": "processing"}


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job_id,
        "status": jobs[job_id]["status"],
        "error": jobs[job_id].get("error"),
    }


@app.get("/download/{job_id}.pdf")
async def download_pdf(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    pdf_path = Path(jobs[job_id]["pdf_path"])
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF not ready")
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"PianoMagic_{job_id}.pdf"
    )


def process_audio(job_id: str, input_path: Path, xml_path: Path, pdf_path: Path):
    try:
        logger.info(f"[{job_id}] === Начало v3.4.0 (pYIN + только скрипичный) ===")

        import librosa
        import numpy as np
        from music21 import (
            stream, instrument, note as m21_note, clef,
            tempo, meter, key, duration as m21_duration,
            tie, pitch as m21_pitch, articulations
        )

        # 1. Загрузка аудио
        y, sr = librosa.load(str(input_path), sr=22050, mono=True)
        logger.info(f"[{job_id}] Аудио загружено: {len(y)} samples @ {sr}Hz")

        # 2. pYIN — извлечение мелодии (f0)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C4'),   # не ниже C4
            fmax=librosa.note_to_hz('C7'),   # не выше C7
            sr=sr,
            frame_length=2048,
        )
        logger.info(f"[{job_id}] pYIN завершён")

        # 3. Конвертация f0 → MIDI-ноты
        times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=512)
        melody = []

        current_note = None
        current_start = 0
        min_note_len = 0.08  # сек

        for t, freq, voiced in zip(times, f0, voiced_flag):
            if not voiced or freq is None or np.isnan(freq):
                if current_note is not None:
                    dur = t - current_start
                    if dur >= min_note_len:
                        melody.append({**current_note, "end": t})
                    current_note = None
                continue

            midi_pitch = int(round(librosa.hz_to_midi(freq)))

            if current_note is None or current_note["pitch"] != midi_pitch:
                if current_note is not None:
                    dur = t - current_start
                    if dur >= min_note_len:
                        melody.append({**current_note, "end": t})
                current_note = {"pitch": midi_pitch, "start": t}
                current_start = t

        # Закрываем последнюю ноту
        if current_note is not None:
            melody.append({**current_note, "end": times[-1] if len(times) > 0 else current_start + 0.5})

        logger.info(f"[{job_id}] Мелодия: {len(melody)} нот")

        if not melody:
            raise ValueError("Мелодия не выделена")

        # 4. Тональность
        try:
            from music21.analysis import discrete
            s_tmp = stream.Stream()
            for m in melody:
                s_tmp.append(m21_note.Note(midi=m["pitch"], quarterLength=0.5))
            analyzer = discrete.KrumhanslSchmuckler()
            detected_key = analyzer.getSolution(s_tmp)
        except Exception:
            detected_key = key.Key("C")

        sharps = detected_key.sharps
        if abs(sharps) > 3:
            candidates = [0, 1, 2, 3, -1, -2, -3]
            best = min(candidates, key=lambda c: abs(c - sharps))
            detected_key = key.Key(m21_pitch.Pitch(sharps=best).name)

        logger.info(f"[{job_id}] Тональность: {detected_key.name}")

        # 5. Квантизация
        def quantize(t):
            ql = t / SEC_PER_QUARTER
            return round(ql / GRID_8TH) * GRID_8TH

        quantized = []
        for m in melody:
            qs = quantize(m["start"])
            qdur = max(GRID_8TH, round((m["end"] - m["start"]) / SEC_PER_QUARTER / GRID_8TH) * GRID_8TH)
            qdur = min(4.0, qdur)
            quantized.append({
                "start": qs,
                "dur": qdur,
                "pitch": m["pitch"],
            })

        # 6. Создание нотного текста — ТОЛЬКО скрипичный ключ, ТОЛЬКО одна линия
        score = stream.Score()

        part = stream.Part()
        part.insert(0, instrument.Piano())
        part.insert(0, clef.TrebleClef())  # ← ЯВНО скрипичный ключ
        part.insert(0, detected_key)
        part.insert(0, meter.TimeSignature("4/4"))
        part.insert(0, tempo.MetronomeMark(number=BPM))

        measures = {}
        for n in quantized:
            m_idx = int(n["start"] // 4.0)
            off = n["start"] % 4.0
            if off >= 3.999:
                m_idx += 1
                off = 0.0
            measures.setdefault(m_idx, []).append(n)

        for m_idx in sorted(measures.keys()):
            measure = stream.Measure(number=m_idx + 1)
            notes_list = sorted(measures[m_idx], key=lambda x: x["start"])

            last_end = 0.0
            prev_note = None

            for nd in notes_list:
                off = nd["start"] % 4.0
                dur = min(nd["dur"], 4.0 - off)
                if dur < GRID_8TH:
                    continue

                gap = off - last_end
                if gap >= GRID_8TH:
                    r = m21_note.Rest()
                    r.duration = m21_duration.Duration(quarterLength=round(gap / GRID_8TH) * GRID_8TH)
                    measure.insert(last_end, r)
                    last_end += r.duration.quarterLength

                n = m21_note.Note(midi=nd["pitch"])
                n.duration = m21_duration.Duration(quarterLength=dur)

                if prev_note and prev_note.pitch.midi == nd["pitch"]:
                    if prev_note.offset + prev_note.duration.quarterLength >= off - 0.01:
                        prev_note.tie = tie.Tie("start")
                        n.tie = tie.Tie("stop")

                measure.insert(off, n)
                prev_note = n
                last_end = off + dur

            part.append(measure)

        try:
            part.makeBeams(inPlace=True)
        except Exception:
            pass

        score.insert(0, part)

        # 7. Сохраняем MusicXML (clef сохраняется точно, в отличие от MIDI)
        score.write("musicxml", str(xml_path))
        logger.info(f"[{job_id}] MusicXML записан")

        # MuseScore: XML → PDF
        result = subprocess.run(
            ["musescore3", "-o", str(pdf_path), str(xml_path)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"MuseScore3 error: {result.stderr}")

        logger.info(f"[{job_id}] PDF создан")
        jobs[job_id]["status"] = "completed"

    except Exception as e:
        logger.error(f"[{job_id}] Ошибка: {e}", exc_info=True)
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
    finally:
        if input_path.exists():
            input_path.unlink()
        if xml_path.exists():
            xml_path.unlink()
