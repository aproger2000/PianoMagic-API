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

app = FastAPI(title="PianoMagic API", version="3.3.0")

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
MIN_AMP = 0.25


@app.get("/")
async def root():
    return {"status": "ok", "version": "3.3.0"}


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


def process_audio(job_id: str, input_path: Path, midi_path: Path, pdf_path: Path):
    try:
        logger.info(f"[{job_id}] === Начало v3.3.0 (только мелодия, только скрипичный ключ) ===")

        from basic_pitch.inference import predict
        from music21 import (
            stream, instrument, note as m21_note,
            tempo, meter, key, duration as m21_duration,
            tie, pitch as m21_pitch, articulations
        )

        # 1. Basic Pitch -> note_events
        _, _, note_events = predict(str(input_path))
        logger.info(f"[{job_id}] Всего нот: {len(note_events)}")

        if not note_events:
            raise ValueError("Ноты не найдены")

        # 2. Фильтрация по громкости
        notes = []
        for start, end, pitch, amp, _ in note_events:
            if amp < MIN_AMP:
                continue
            notes.append({
                "start": float(start),
                "end": float(end),
                "pitch": int(pitch),
                "velocity": min(127, int(amp * 127)),
            })

        if not notes:
            raise ValueError("После фильтрации нот не осталось")

        # 3. === ИЗВЛЕЧЕНИЕ ТОЛЬКО МЕЛОДИИ (верхний голос, монофония) ===
        TIME_RES = 0.1
        timeline = {}

        for n in notes:
            t = n["start"]
            while t < n["end"]:
                slot = round(t / TIME_RES) * TIME_RES
                # В каждый момент времени оставляем только самую высокую ноту
                if slot not in timeline or n["pitch"] > timeline[slot]["pitch"]:
                    timeline[slot] = n
                t += TIME_RES

        # Собираем мелодическую линию
        slots = sorted(timeline.keys())
        melody = []
        for slot in slots:
            n = timeline[slot]
            if not melody or n["pitch"] != melody[-1]["pitch"]:
                melody.append({
                    "start": slot,
                    "pitch": n["pitch"],
                    "velocity": n["velocity"],
                })

        # Вычисляем длительности
        for i in range(len(melody)):
            if i + 1 < len(melody):
                melody[i]["dur"] = melody[i + 1]["start"] - melody[i]["start"]
            else:
                melody[i]["dur"] = 0.5

        # Убираем слишком короткие
        melody = [m for m in melody if m["dur"] >= 0.08]

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

        # 5. Квантизация до 1/8
        def quantize(t):
            ql = t / SEC_PER_QUARTER
            return round(ql / GRID_8TH) * GRID_8TH

        quantized = []
        for m in melody:
            qs = quantize(m["start"])
            qdur = max(GRID_8TH, round(m["dur"] / SEC_PER_QUARTER / GRID_8TH) * GRID_8TH)
            qdur = min(4.0, qdur)
            quantized.append({
                "start": qs,
                "dur": qdur,
                "pitch": m["pitch"],
                "velocity": m["velocity"],
            })

        # 6. === СОЗДАНИЕ НОТНОГО ТЕКСТА: ТОЛЬКО ОДНА ЛИНИЯ, ТОЛЬКО СКРИПИЧНЫЙ КЛЮЧ ===
        score = stream.Score()

        # ОДНА часть (одна рука)
        part = stream.Part()
        part.insert(0, instrument.Piano())
        part.insert(0, detected_key)
        part.insert(0, meter.TimeSignature("4/4"))
        part.insert(0, tempo.MetronomeMark(number=BPM))
        # Скрипичный ключ по умолчанию — ничего не добавляем, bass clef НЕ ставим

        # Раскладываем по тактам
        measures = {}
        for n in quantized:
            m_idx = int(n["start"] // 4.0)
            off = n["start"] % 4.0
            if off >= 3.999:
                m_idx += 1
                off = 0.0
            measures.setdefault(m_idx, []).append(n)

        avg_vel = sum(m["velocity"] for m in quantized) / len(quantized)

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

                # Пауза
                gap = off - last_end
                if gap >= GRID_8TH:
                    r = m21_note.Rest()
                    r.duration = m21_duration.Duration(quarterLength=round(gap / GRID_8TH) * GRID_8TH)
                    measure.insert(last_end, r)
                    last_end += r.duration.quarterLength

                # Нота
                n = m21_note.Note(midi=nd["pitch"])
                n.duration = m21_duration.Duration(quarterLength=dur)
                n.volume.velocity = nd["velocity"]

                # Лига
                if prev_note and prev_note.pitch.midi == nd["pitch"]:
                    if prev_note.offset + prev_note.duration.quarterLength >= off - 0.01:
                        prev_note.tie = tie.Tie("start")
                        n.tie = tie.Tie("stop")

                # Акцент
                if nd["velocity"] > avg_vel * 1.3:
                    n.articulations.append(articulations.Accent())

                measure.insert(off, n)
                prev_note = n
                last_end = off + dur

            part.append(measure)

        try:
            part.makeBeams(inPlace=True)
        except Exception:
            pass

        score.insert(0, part)

        # 7. Сохранение
        score.write("midi", str(midi_path))
        logger.info(f"[{job_id}] MIDI записан")

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
