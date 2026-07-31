"""
PianoMagic API — Piano Edition (разделение рук + квантизация)
"""

import asyncio
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from itertools import groupby

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from music21 import (
    converter, stream, note, chord, meter, key, instrument,
    clef, layout, metadata
)
from basic_pitch.inference import predict

app = FastAPI(title="PianoMagic API Piano", version="12.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

TEMP_DIR = Path("temp")
OUTPUT_DIR = Path("output")
TEMP_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_FILE_SIZE = 50 * 1024 * 1024
jobs = {}


def transcribe_sync(input_path: Path, midi_path: Path):
    print(f"[TRANSCRIBE] {input_path}")
    _, midi_data, _ = predict(str(input_path))
    midi_data.write(str(midi_path))
    print(f"[TRANSCRIBE] MIDI done")


def quantize_duration(d: float) -> float:
    std = [4.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.375, 0.25, 0.125]
    return min(std, key=lambda x: abs(x - d))


def process_midi_to_sheet(midi_path: Path, pdf_path: Path, title: str):
    print(f"[SHEET] {midi_path}")
    midi_stream = converter.parse(str(midi_path))

    score = stream.Score()
    meta = metadata.Metadata()
    meta.title = title
    meta.composer = "PianoMagic AI"
    score.insert(0, meta)

    right = stream.Part()
    right.insert(0, instrument.Piano())
    right.insert(0, clef.TrebleClef())
    right.insert(0, meter.TimeSignature("4/4"))
    right.insert(0, key.Key("C"))

    left = stream.Part()
    left.insert(0, instrument.Piano())
    left.insert(0, clef.BassClef())
    left.insert(0, meter.TimeSignature("4/4"))
    left.insert(0, key.Key("C"))

    all_notes = []
    for el in midi_stream.recurse():
        if isinstance(el, note.Note):
            all_notes.append({
                "pitch": el.pitch.midi,
                "offset": float(el.offset),
                "duration": float(el.duration.quarterLength)
            })
        elif isinstance(el, chord.Chord):
            for p in el.pitches:
                all_notes.append({
                    "pitch": p.midi,
                    "offset": float(el.offset),
                    "duration": float(el.duration.quarterLength)
                })

    all_notes.sort(key=lambda x: (round(x["offset"], 3), -x["pitch"]))

    right_notes = []
    left_notes = []

    for offset, group in groupby(all_notes, key=lambda x: round(x["offset"], 3)):
        grp = list(group)
        if len(grp) == 1:
            n = grp[0]
            (right_notes if n["pitch"] >= 60 else left_notes).append(n)
        else:
            grp.sort(key=lambda x: x["pitch"], reverse=True)
            mid = len(grp) // 2
            right_notes.extend(grp[:mid])
            left_notes.extend(grp[mid:])

    def add_to_part(part, notes, min_p, max_p):
        prev = -1
        for n in notes:
            p = max(min_p, min(max_p, n["pitch"]))
            off = round(n["offset"] * 4) / 4
            dur = quantize_duration(n["duration"])
            if dur < 0.125:
                continue
            if abs(off - prev) < 0.0625:
                off = prev + dur
            part.insert(off, note.Note(pitch=p, quarterLength=dur))
            prev = off

    add_to_part(right, right_notes, 60, 96)
    add_to_part(left, left_notes, 28, 64)

    score.insert(0, right)
    score.insert(0, left)
    score.insert(0, layout.SystemLayout(isNew=True))

    xml_path = midi_path.with_suffix(".musicxml")
    score.write("musicxml", fp=str(xml_path))

    mscore = (
        shutil.which("mscore") or shutil.which("mscore4") or
        shutil.which("musescore") or shutil.which("musescore3")
    )
    if mscore:
        env = os.environ.copy()
        env["QT_QPA_PLATFORM"] = "offscreen"
        subprocess.run(
            [mscore, str(xml_path), "-o", str(pdf_path)],
            check=True, capture_output=True, timeout=60, env=env
        )
        print(f"[SHEET] MuseScore done")
    else:
        score.write("lily.pdf", fp=str(pdf_path))
        print(f"[SHEET] LilyPond done")


async def process_audio_async(job_id: str, input_path: Path):
    try:
        jobs[job_id]["status"] = "transcribing"
        jobs[job_id]["message"] = "AI анализирует аудио..."

        job_dir = TEMP_DIR / job_id
        job_dir.mkdir(exist_ok=True)

        midi_path = job_dir / "transcription.mid"
        pdf_path = OUTPUT_DIR / f"{job_id}.pdf"

        await asyncio.to_thread(transcribe_sync, input_path, midi_path)

        jobs[job_id]["status"] = "generating_sheet"
        jobs[job_id]["message"] = "Создание нот (2 руки)..."

        title = input_path.stem.replace("-", " ").replace("_", " ").title()
        await asyncio.to_thread(process_midi_to_sheet, midi_path, pdf_path, title)

        try:
            shutil.rmtree(job_dir)
            os.remove(input_path)
        except:
            pass

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["pdf_url"] = f"/download/{job_id}.pdf"
        jobs[job_id]["message"] = "Готово!"
        print(f"[DONE] {job_id}")

    except Exception as e:
        print(f"[ERROR] {job_id}: {e}")
        import traceback
        traceback.print_exc()
        jobs[job_id]["status"] = "error"
        jobs[job_id]["message"] = str(e)


@app.post("/transcribe/file")
async def transcribe_file(file: UploadFile = File(...)):
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Файл слишком большой")

    allowed = (".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac")
    if not file.filename.lower().endswith(allowed):
        raise HTTPException(status_code=400, detail="Неподдерживаемый формат")

    job_id = str(uuid.uuid4())[:8]
    input_path = TEMP_DIR / job_id / file.filename
    input_path.parent.mkdir(exist_ok=True)

    with open(input_path, "wb") as f:
        f.write(content)

    jobs[job_id] = {
        "job_id": job_id,
        "status": "processing",
        "message": "Начинаем...",
        "filename": file.filename
    }

    asyncio.create_task(process_audio_async(job_id, input_path))

    return JSONResponse(content={
        "job_id": job_id,
        "status": "processing",
        "message": "Обработка начата"
    })


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Задание не найдено")
    return JSONResponse(content=jobs[job_id])


@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Файл не найден")
    return FileResponse(path=file_path, filename=filename, media_type="application/pdf")


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "PianoMagic API Piano"}


@app.get("/")
async def root():
    return {"service": "PianoMagic API Piano", "version": "12.0.0"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)
