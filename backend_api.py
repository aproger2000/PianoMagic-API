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

app = FastAPI(title="PianoMagic API", version="3.5.0")

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
MIN_AMP = 0.30


@app.get("/")
async def root():
    return {"status": "ok", "version": "3.5.0"}


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
        logger.info(f"[{job_id}] === Начало v3.5.0 (две руки) ===")

        from basic_pitch.inference import predict
        from music21 import (
            stream, instrument, note as m21_note, clef,
            tempo, meter, key, duration as m21_duration,
            tie, pitch as m21_pitch, articulations
        )

        # 1. Basic Pitch
        _, _, note_events = predict(str(input_path))
        logger.info(f"[{job_id}] Всего нот: {len(note_events)}")

        if not note_events:
            raise ValueError("Ноты не найдены")

        # 2. Фильтрация
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

        # 3. === РАЗДЕЛЕНИЕ НА ДВА ГОЛОСА ===
        TIME_RES = 0.1
        timeline = {}

        for n in notes:
            t = n["start"]
            while t < n["end"]:
                slot = round(t / TIME_RES) * TIME_RES
                if slot not in timeline:
                    timeline[slot] = []
                timeline[slot].append(n)
                t += TIME_RES

        right_voice = []
        left_voice = []

        for slot in sorted(timeline.keys()):
            slot_notes = timeline[slot]
            slot_notes.sort(key=lambda x: x["pitch"])

            # Верхняя нота → правая рука (мелодия)
            top = slot_notes[-1]
            if not right_voice or top["pitch"] != right_voice[-1]["pitch"]:
                right_voice.append({
                    "start": slot,
                    "pitch": top["pitch"],
                    "velocity": top["velocity"],
                })

            # Нижняя нота → левая рука (бас), если отличается значимо
            if len(slot_notes) > 1:
                bottom = slot_notes[0]
                interval = top["pitch"] - bottom["pitch"]
                # Берём бас только если он достаточно далеко (≥ 5 полутонов) и низкий
                if interval >= 5 and bottom["pitch"] < 60:
                    if not left_voice or bottom["pitch"] != left_voice[-1]["pitch"]:
                        left_voice.append({
                            "start": slot,
                            "pitch": bottom["pitch"],
                            "velocity": bottom["velocity"],
                        })

        # Длительности
        for voice in [right_voice, left_voice]:
            for i in range(len(voice)):
                if i + 1 < len(voice):
                    voice[i]["dur"] = voice[i + 1]["start"] - voice[i]["start"]
                else:
                    voice[i]["dur"] = 0.5
            voice[:] = [n for n in voice if n["dur"] >= 0.08]

        logger.info(f"[{job_id}] Правая: {len(right_voice)}, Левая: {len(left_voice)}")

        if not right_voice:
            raise ValueError("Мелодия не выделена")

        # 4. Диапазоны
        for n in right_voice:
            while n["pitch"] < 60:   # C4
                n["pitch"] += 12
            while n["pitch"] > 84:   # C6
                n["pitch"] -= 12

        for n in left_voice:
            while n["pitch"] > 60:   # C4
                n["pitch"] -= 12
            while n["pitch"] < 36:   # C2
                n["pitch"] += 12

        # 5. Тональность (по правой руке)
        try:
            from music21.analysis import discrete
            s_tmp = stream.Stream()
            for m in right_voice:
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

        # 6. Квантизация
        def quantize(t):
            ql = t / SEC_PER_QUARTER
            return round(ql / GRID_8TH) * GRID_8TH

        def quantize_voice(voice):
            out = []
            for m in voice:
                qs = quantize(m["start"])
                qdur = max(GRID_8TH, round(m["dur"] / SEC_PER_QUARTER / GRID_8TH) * GRID_8TH)
                qdur = min(4.0, qdur)
                out.append({"start": qs, "dur": qdur, "pitch": m["pitch"], "velocity": m["velocity"]})
            return out

        right_q = quantize_voice(right_voice)
        left_q = quantize_voice(left_voice)

        # 7. === СОЗДАНИЕ НОТНОГО ТЕКСТА ===
        score = stream.Score()

        # --- Правая рука (TrebleClef) ---
        right_part = stream.Part()
        right_part.insert(0, instrument.Piano())
        right_part.insert(0, clef.TrebleClef())
        right_part.insert(0, detected_key)
        right_part.insert(0, meter.TimeSignature("4/4"))
        right_part.insert(0, tempo.MetronomeMark(number=BPM))

        # --- Левая рука (BassClef) ---
        left_part = stream.Part()
        left_part.insert(0, instrument.Piano())
        left_part.insert(0, clef.BassClef())
        left_part.insert(0, detected_key)
        left_part.insert(0, meter.TimeSignature("4/4"))
        left_part.insert(0, tempo.MetronomeMark(number=BPM))

        def build_measures(q_voice):
            measures = {}
            for n in q_voice:
                m_idx = int(n["start"] // 4.0)
                off = n["start"] % 4.0
                if off >= 3.999:
                    m_idx += 1
                    off = 0.0
                measures.setdefault(m_idx, []).append(n)
            return measures

        def fill_part(part, q_voice, avg_vel):
            measures = build_measures(q_voice)
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
                    n.volume.velocity = nd["velocity"]

                    if prev_note and prev_note.pitch.midi == nd["pitch"]:
                        if prev_note.offset + prev_note.duration.quarterLength >= off - 0.01:
                            prev_note.tie = tie.Tie("start")
                            n.tie = tie.Tie("stop")

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

        avg_vel_right = sum(n["velocity"] for n in right_q) / len(right_q) if right_q else 64
        avg_vel_left = sum(n["velocity"] for n in left_q) / len(left_q) if left_q else 64

        fill_part(right_part, right_q, avg_vel_right)
        fill_part(left_part, left_q, avg_vel_left)

        score.insert(0, right_part)
        score.insert(0, left_part)

        # 8. MusicXML → PDF
        score.write("musicxml", str(xml_path))
        logger.info(f"[{job_id}] MusicXML записан")

        result = subprocess.run(
            ["musescore3", "-o", str(pdf_path), str(xml_path)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"MuseScore3 error: {result.stderr or result.stdout}")

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
