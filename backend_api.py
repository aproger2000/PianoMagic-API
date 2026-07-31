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

app = FastAPI(title="PianoMagic API", version="3.1.0")

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


@app.get("/")
async def root():
    return {"status": "ok", "version": "3.1.0"}


@app.post("/transcribe/file")
async def transcribe_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(file.filename or "input.mp3").suffix
    input_path = job_dir / f"input{ext}"
    midi_path = job_dir / "output.mid"
    pdf_path = job_dir / "output.pdf"

    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    jobs[job_id] = {"status": "processing", "pdf_path": str(pdf_path), "error": None}
    background_tasks.add_task(process_audio, job_id, input_path, midi_path, pdf_path)

    return {"job_id": job_id, "status": "processing"}


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "status": jobs[job_id]["status"]}


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


def process_audio(job_id: str, input_path: Path, midi_path: Path, pdf_path: Path):
    try:
        logger.info(f"[{job_id}] Начало обработки...")

        from basic_pitch.inference import predict
        from music21 import (
            stream, instrument, clef, note as m21_note, chord as m21_chord,
            tempo, meter
        )

        # 1. Basic Pitch -> note_events (start, end, pitch, amplitude, bends)
        _, _, note_events = predict(str(input_path))

        logger.info(f"[{job_id}] Найдено нот: {len(note_events)}")

        # 2. Фильтрация артефактов
        MIN_AMP = 0.25      # минимальная громкость (0-1)
        MIN_DUR = 0.08      # сек — слишком короткие ноты
        MAX_RIGHT_POLY = 4  # макс нот в аккорде (правая рука)
        MAX_LEFT_POLY = 3   # макс нот в аккорде (левая рука)
        BPM = 120
        SEC_PER_QUARTER = 60.0 / BPM
        GRID = 0.5          # 1/8 note в quarterLength (чистый ритм)

        filtered = []
        for start, end, pitch, amp, _ in note_events:
            dur = end - start
            if amp < MIN_AMP or dur < MIN_DUR:
                continue
            filtered.append({
                'start': start,
                'end': end,
                'pitch': int(pitch),
                'velocity': min(127, int(amp * 127)),
            })

        logger.info(f"[{job_id}] После фильтрации: {len(filtered)}")
        if not filtered:
            raise ValueError("Не найдено подходящих нот в аудио")

        # 3. Разделение по рукам (C4 = 60)
        right_raw = [n for n in filtered if n['pitch'] >= 60]
        left_raw = [n for n in filtered if n['pitch'] < 60]

        # 4. Квантизация + ограничение полифонии
        def quantize(t):
            ql = t / SEC_PER_QUARTER
            return round(ql / GRID) * GRID

        def build_hand(raw_notes, max_poly):
            buckets = {}
            for n in raw_notes:
                q_start = quantize(n['start'])
                q_end = quantize(n['end'])
                q_dur = max(GRID, q_end - q_start)
                buckets.setdefault(q_start, []).append({**n, 'q_dur': q_dur})

            items = []
            for offset in sorted(buckets.keys()):
                notes = buckets[offset]
                # Оставляем самые громкие
                notes.sort(key=lambda x: x['velocity'], reverse=True)
                notes = notes[:max_poly]

                # Уникальные pitch (убираем дубли)
                seen = set()
                unique = []
                for note in notes:
                    if note['pitch'] not in seen:
                        seen.add(note['pitch'])
                        unique.append(note)

                if unique:
                    items.append((offset, unique))
            return items

        right_hand = build_hand(right_raw, MAX_RIGHT_POLY)
        left_hand = build_hand(left_raw, MAX_LEFT_POLY)

        logger.info(f"[{job_id}] Правая: {len(right_hand)} акк., Левая: {len(left_hand)} акк.")

        # 5. Создаём нотный текст с нуля
        score = stream.Score()
        score.insert(0, tempo.MetronomeMark(number=BPM))
        score.insert(0, meter.TimeSignature('4/4'))

        # Правая рука
        right_part = stream.Part()
        right_part.insert(0, instrument.Piano())

        for offset, notes in right_hand:
            if len(notes) == 1:
                n = m21_note.Note(notes[0]['pitch'])
                n.duration.quarterLength = notes[0]['q_dur']
                n.volume.velocity = notes[0]['velocity']
                right_part.insert(offset, n)
            else:
                c = m21_chord.Chord([p['pitch'] for p in notes])
                c.duration.quarterLength = notes[0]['q_dur']
                c.volume.velocity = notes[0]['velocity']
                right_part.insert(offset, c)

        # Левая рука
        left_part = stream.Part()
        left_part.insert(0, instrument.Piano())
        left_part.insert(0, clef.BassClef())

        for offset, notes in left_hand:
            if len(notes) == 1:
                n = m21_note.Note(notes[0]['pitch'])
                n.duration.quarterLength = notes[0]['q_dur']
                n.volume.velocity = notes[0]['velocity']
                left_part.insert(offset, n)
            else:
                c = m21_chord.Chord([p['pitch'] for p in notes])
                c.duration.quarterLength = notes[0]['q_dur']
                c.volume.velocity = notes[0]['velocity']
                left_part.insert(offset, c)

        score.insert(0, right_part)
        score.insert(0, left_part)

        # 6. Сохраняем MIDI
        score.write("midi", str(midi_path))
        logger.info(f"[{job_id}] MIDI записан")

        # 7. MuseScore -> PDF
        result = subprocess.run(
            ["musescore3", "-o", str(pdf_path), str(midi_path)],
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
