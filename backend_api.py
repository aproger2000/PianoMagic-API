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

app = FastAPI(title="PianoMagic API", version="3.0.1")

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
    return {"status": "ok", "version": "3.0.1"}


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

        # ========== 1. Basic Pitch (ONNX) -> MIDI ==========
        from basic_pitch.inference import predict_and_save
        from basic_pitch import ICASSP_2022_MODEL_PATH

        # Явно указываем путь к ONNX-модели, чтобы не грузить TensorFlow
        onnx_model_path = Path(ICASSP_2022_MODEL_PATH).with_suffix(".onnx")
        if not onnx_model_path.exists():
            onnx_model_path = Path(ICASSP_2022_MODEL_PATH).parent / "nmp.onnx"
        
        logger.info(f"[{job_id}] ONNX модель: {onnx_model_path}")
        if not onnx_model_path.exists():
            raise FileNotFoundError(f"ONNX модель не найдена: {onnx_model_path}")

        predict_and_save(
            audio_path_list=[str(input_path)],
            output_directory=str(midi_path.parent),
            save_midi=True,
            sonify_midi=False,
            save_model_outputs=False,
            save_notes=False,
            model_path=str(onnx_model_path),
        )

        # Basic Pitch сохраняет файл как {input_name}_basic_pitch.mid
        bp_output = midi_path.parent / f"{input_path.stem}_basic_pitch.mid"
        if not bp_output.exists():
            candidates = list(midi_path.parent.glob("*_basic_pitch.mid"))
            if candidates:
                bp_output = candidates[0]
            else:
                raise FileNotFoundError("Basic Pitch не создал MIDI-файл")

        logger.info(f"[{job_id}] MIDI создан: {bp_output}")

        # ========== 2. Пост-обработка music21 ==========
        from music21 import converter, stream, instrument, clef, note as m21_note, chord as m21_chord

        score = converter.parse(str(bp_output))

        treble_notes = []
        bass_notes = []

        for element in score.recurse():
            if isinstance(element, m21_note.Note):
                (treble_notes if element.pitch.midi >= 60 else bass_notes).append(element)
            elif isinstance(element, m21_chord.Chord):
                for pitch in element.pitches:
                    n = m21_note.Note(pitch)
                    n.duration = element.duration
                    n.offset = element.offset
                    (treble_notes if pitch.midi >= 60 else bass_notes).append(n)

        new_score = stream.Score()

        right_hand = stream.Part()
        right_hand.insert(0, instrument.Piano())

        left_hand = stream.Part()
        left_hand.insert(0, instrument.Piano())
        left_hand.insert(0, clef.BassClef())

        for n in treble_notes:
            right_hand.insert(n.offset, n)
        for n in bass_notes:
            left_hand.insert(n.offset, n)

        new_score.insert(0, right_hand)
        new_score.insert(0, left_hand)

        try:
            new_score = new_score.quantize(quarterLengthDivisors=[4, 3], inPlace=False)
        except Exception as e:
            logger.warning(f"[{job_id}] Квантизация пропущена: {e}")

        new_score.write("midi", str(midi_path))
        logger.info(f"[{job_id}] Пост-обработка завершена")

        # ========== 3. MuseScore3 -> PDF ==========
        result = subprocess.run(
            ["musescore3", "-o", str(pdf_path), str(midi_path)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"MuseScore3 error: {result.stderr}")

        logger.info(f"[{job_id}] PDF создан: {pdf_path}")
        jobs[job_id]["status"] = "completed"

    except Exception as e:
        logger.error(f"[{job_id}] Ошибка: {e}", exc_info=True)
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
    finally:
        if input_path.exists():
            input_path.unlink()
        bp_cleanup = midi_path.parent / f"{input_path.stem}_basic_pitch.mid"
        if bp_cleanup.exists():
            bp_cleanup.unlink()
