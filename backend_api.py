"""
PianoMagic API v2 — с фоновой обработкой
FastAPI + Basic Pitch + Background Tasks

Паттерн: upload → get job_id → poll status → download PDF
"""

import asyncio
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from music21 import converter, instrument
from basic_pitch.inference import predict

app = FastAPI(
    title="PianoMagic API v2",
    description="Транскрипция аудио в ноты (async)",
    version="2.0.0"
)

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

# Хранилище статусов задач
jobs = {}

# Пул потоков для тяжёлых задач (TensorFlow)
executor = ThreadPoolExecutor(max_workers=1)


def process_audio_sync(job_id: str, input_path: Path):
    """Синхронная обработка в отдельном потоке."""
    try:
        jobs[job_id]["status"] = "transcribing"
        jobs[job_id]["message"] = "AI анализирует аудио..."

        job_dir = TEMP_DIR / job_id
        job_dir.mkdir(exist_ok=True)

        midi_path = job_dir / "transcription.mid"
        pdf_path = OUTPUT_DIR / f"{job_id}.pdf"

        # Шаг 1: Audio → MIDI (Basic Pitch)
        print(f"[{job_id}] Starting transcription...")
        _, midi_data, _ = predict(str(input_path))
        midi_data.write(str(midi_path))
        print(f"[{job_id}] MIDI created: {midi_path}")

        jobs[job_id]["status"] = "generating_pdf"
        jobs[job_id]["message"] = "Генерация нотного листа..."

        # Шаг 2: MIDI → PDF
        mscore = (
            shutil.which("mscore") or 
            shutil.which("mscore4") or 
            shutil.which("musescore") or 
            shutil.which("musescore3")
        )

        if mscore:
            subprocess.run(
                [mscore, str(midi_path), "-o", str(pdf_path)],
                check=True,
                capture_output=True,
                timeout=60
            )
        else:
            s = converter.parse(str(midi_path))
            for part in s.parts:
                part.insert(0, instrument.Piano())
            s = s.quantize()
            s.write("lily.pdf", fp=str(pdf_path))

        print(f"[{job_id}] PDF created: {pdf_path}")

        # Очистка
        try:
            shutil.rmtree(job_dir)
            os.remove(input_path)
        except:
            pass

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["pdf_url"] = f"/download/{job_id}.pdf"
        jobs[job_id]["message"] = "Готово!"

    except Exception as e:
        print(f"[{job_id}] ERROR: {str(e)}")
        jobs[job_id]["status"] = "error"
        jobs[job_id]["message"] = str(e)


@app.post("/transcribe/file")
async def transcribe_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Загрузка файла и запуск фоновой транскрипции.
    Возвращает job_id для polling статуса.
    """
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Файл слишком большой. Максимум 50 МБ.")

    allowed = (".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac")
    if not file.filename.lower().endswith(allowed):
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемый формат. Разрешены: {', '.join(allowed)}"
        )

    job_id = str(uuid.uuid4())[:8]
    input_path = TEMP_DIR / job_id / file.filename
    input_path.parent.mkdir(exist_ok=True)

    with open(input_path, "wb") as f:
        f.write(content)

    # Сохраняем статус
    jobs[job_id] = {
        "job_id": job_id,
        "status": "processing",
        "message": "Файл загружен, начинаем обработку...",
        "filename": file.filename
    }

    # Запускаем в фоновом потоке (чтобы не блокировать HTTP-ответ)
    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, process_audio_sync, job_id, input_path)

    return JSONResponse(content={
        "job_id": job_id,
        "status": "processing",
        "message": "Обработка начата. Используйте GET /jobs/{job_id} для проверки статуса."
    })


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Проверка статуса задания."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Задание не найдено")

    return JSONResponse(content=jobs[job_id])


@app.get("/download/{filename}")
async def download_file(filename: str):
    """Скачивание PDF."""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Файл не найден")

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/pdf"
    )


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "PianoMagic API v2"}


@app.get("/")
async def root():
    return {
        "service": "PianoMagic API v2",
        "version": "2.0.0",
        "endpoints": {
            "upload": "POST /transcribe/file → возвращает job_id",
            "status": "GET /jobs/{job_id}",
            "download": "GET /download/{filename}"
        }
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
