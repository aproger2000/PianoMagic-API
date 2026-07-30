"""
PianoMagic — Audio-to-Sheet Music Transcription API
FastAPI + Basic Pitch + MuseScore

Запуск локально:
    uvicorn backend_api:app --reload

Деплой:
    Render / Railway / Hugging Face Spaces
"""

import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Optional

import yt_dlp
from basic_pitch.inference import predict
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from music21 import converter, instrument

app = FastAPI(
    title="PianoMagic API",
    description="Транскрипция аудио в ноты для фортепиано",
    version="1.0.0"
)

# CORS — разрешаем запросы с GitHub Pages / Vercel / localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене замените на конкретный домен
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Директории
TEMP_DIR = Path("temp")
OUTPUT_DIR = Path("output")
TEMP_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Максимальный размер файла (50 МБ)
MAX_FILE_SIZE = 50 * 1024 * 1024


def download_audio(url: str, output_path: Path) -> Path:
    """Скачивание аудио с YouTube, SoundCloud, и т.д."""
    try:
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": str(output_path.with_suffix("")),
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }],
            "quiet": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return output_path
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка скачивания: {str(e)}")


def audio_to_midi(audio_path: Path, midi_path: Path) -> Path:
    """Транскрипция аудио → MIDI через Basic Pitch (Spotify)."""
    try:
        _, midi_data, _ = predict(str(audio_path))
        midi_data.write(str(midi_path))
        return midi_path
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка транскрипции: {str(e)}")


def midi_to_pdf(midi_path: Path, pdf_path: Path) -> Path:
    """MIDI → PDF через MuseScore CLI или LilyPond (fallback)."""
    try:
        # Ищем MuseScore
        mscore = shutil.which("mscore") or shutil.which("mscore4") or shutil.which("musescore") or shutil.which("musescore3")


        if mscore:
            subprocess.run(
                [mscore, str(midi_path), "-o", str(pdf_path)],
                check=True,
                capture_output=True,
                timeout=60
            )
        else:
            # Fallback: music21 + LilyPond
            s = converter.parse(str(midi_path))
            for part in s.parts:
                part.insert(0, instrument.Piano())
            s = s.quantize()
            s.write("lily.pdf", fp=str(pdf_path))

        return pdf_path
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Таймаут генерации PDF")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации PDF: {str(e)}")


def process_audio(input_path: Path, job_id: str) -> dict:
    """Полный пайплайн: аудио → MIDI → PDF."""
    job_dir = TEMP_DIR / job_id
    job_dir.mkdir(exist_ok=True)

    midi_path = job_dir / "transcription.mid"
    pdf_path = OUTPUT_DIR / f"{job_id}.pdf"

    # Шаг 1: Audio → MIDI
    audio_to_midi(input_path, midi_path)

    # Шаг 2: MIDI → PDF
    midi_to_pdf(midi_path, pdf_path)

    # Очистка временных файлов
    try:
        shutil.rmtree(job_dir)
    except:
        pass

    return {
        "job_id": job_id,
        "status": "completed",
        "pdf_url": f"/download/{job_id}.pdf",
        "midi_url": f"/download/{job_id}.mid",
        "message": "Транскрипция завершена успешно"
    }


@app.post("/transcribe/file")
async def transcribe_file(file: UploadFile = File(...)):
    """
    Загрузка аудиофайла и транскрипция в ноты.
    Поддерживаемые форматы: MP3, WAV, FLAC, M4A, OGG
    """
    # Проверка размера
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Файл слишком большой. Максимум 50 МБ.")

    # Проверка формата
    allowed = (".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac", ".wma")
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

    return JSONResponse(content=process_audio(input_path, job_id))


@app.post("/transcribe/url")
async def transcribe_url(url: str = Form(...)):
    """
    Транскрипция по ссылке (YouTube, SoundCloud, и др.)
    """
    job_id = str(uuid.uuid4())[:8]
    input_path = TEMP_DIR / job_id / "audio.wav"
    input_path.parent.mkdir(exist_ok=True)

    download_audio(url, input_path)
    return JSONResponse(content=process_audio(input_path, job_id))


@app.get("/download/{filename}")
async def download_file(filename: str):
    """Скачивание сгенерированного PDF или MIDI."""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Файл не найден или устарел")

    media_type = "application/pdf" if filename.endswith(".pdf") else "audio/midi"
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type=media_type
    )


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Проверка статуса задания."""
    pdf_path = OUTPUT_DIR / f"{job_id}.pdf"
    if pdf_path.exists():
        return {
            "job_id": job_id,
            "status": "completed",
            "pdf_url": f"/download/{job_id}.pdf",
        }
    return {"job_id": job_id, "status": "processing"}


@app.get("/health")
async def health_check():
    """Проверка работоспособности сервера."""
    return {"status": "ok", "service": "PianoMagic API"}


@app.get("/")
async def root():
    return {
        "service": "PianoMagic API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "POST /transcribe/file",
            "url": "POST /transcribe/url",
            "download": "GET /download/{filename}",
            "health": "GET /health"
        },
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
