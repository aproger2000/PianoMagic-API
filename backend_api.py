import os
import uuid
import shutil
import subprocess
import logging
from pathlib import Path
from collections import defaultdict

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PianoMagic API", version="3.2.4")

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
MIN_AMP = 0.40
MIN_DUR_SEC = 0.15
MAX_DUR_QL = 4.0
RIGHT_POLY_MAX = 4
LEFT_POLY_MAX = 3
MAX_KEY_SHARPS = 3
MAX_CHORD_SPAN = 12  # макс 1 октава в аккорде


@app.get("/")
async def root():
    return {"status": "ok", "version": "3.2.4"}


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


def limit_chord_span(notes, max_span):
    """Удаляет крайние ноты, пока размах аккорда <= max_span полутонов.
    Удаляет тишее из двух крайних."""
    if len(notes) <= 1:
        return notes
    notes = sorted(notes, key=lambda x: x["pitch"])
    while len(notes) > 1 and notes[-1]["pitch"] - notes[0]["pitch"] > max_span:
        if notes[0]["velocity"] <= notes[-1]["velocity"]:
            notes.pop(0)
        else:
            notes.pop()
    return notes


def process_audio(job_id: str, input_path: Path, midi_path: Path, pdf_path: Path):
    try:
        logger.info(f"[{job_id}] === Начало v3.2.4 ===")

        from basic_pitch.inference import predict
        from basic_pitch import ICASSP_2022_MODEL_PATH

        _, _, note_events = predict(str(input_path))
        logger.info(f"[{job_id}] Найдено нот: {len(note_events)}")

        if not note_events:
            raise ValueError("Ноты не найдены")

        from music21 import (
            stream, instrument, clef, note as m21_note, chord as m21_chord,
            tempo, meter, key, duration as m21_duration, articulations,
            tie, pitch as m21_pitch
        )

        # 2. Фильтрация
        notes_raw = []
        for start, end, pitch_val, amp, _ in note_events:
            dur = end - start
            if amp < MIN_AMP or dur < MIN_DUR_SEC:
                continue
            p = int(pitch_val)
            while p < 24:
                p += 12
            while p > 96:
                p -= 12
            notes_raw.append({
                "start": start,
                "end": end,
                "pitch": p,
                "velocity": min(127, int(amp * 127)),
            })

        if not notes_raw:
            raise ValueError("После фильтрации нот не осталось")

        avg_vel = sum(n["velocity"] for n in notes_raw) / len(notes_raw)
        logger.info(f"[{job_id}] После фильтрации: {len(notes_raw)}, avg_vel={avg_vel:.1f}")

        # 3. Тональность
        try:
            from music21.analysis import discrete
            s_tmp = stream.Stream()
            for n in notes_raw:
                s_tmp.append(m21_note.Note(midi=n["pitch"], quarterLength=0.5))
            analyzer = discrete.KrumhanslSchmuckler()
            detected_key = analyzer.getSolution(s_tmp)
        except Exception as e:
            logger.warning(f"[{job_id}] Krumhansl failed: {e}")
            detected_key = key.Key("C")

        sharps = detected_key.sharps
        if abs(sharps) > MAX_KEY_SHARPS:
            candidates = [0, 1, 2, 3, -1, -2, -3]
            best = min(candidates, key=lambda c: abs(c - sharps))
            detected_key = key.Key(m21_pitch.Pitch(sharps=best).name)
            sharps = best

        logger.info(f"[{job_id}] Тональность: {detected_key.name} ({sharps} знаков)")

        # 4. Разделение по рукам с адаптивной зоной пересечения
        right_raw = []
        left_raw = []
        for n in notes_raw:
            p = n["pitch"]
            if p >= 60:  # C4 и выше → правая
                right_raw.append(n)
            elif p < 48:  # ниже C3 → левая
                left_raw.append(n)
            else:  # C3–B3 (48–59) → зона пересечения
                # Назначаем руку, где меньше нот (балансировка)
                if len(right_raw) <= len(left_raw):
                    right_raw.append(n)
                else:
                    left_raw.append(n)

        # 5. Квантизация
        def quantize_ql(t):
            ql = t / SEC_PER_QUARTER
            q = round(ql / GRID_8TH) * GRID_8TH
            return max(0.0, q)

        def prepare_hand(raw, poly_max, is_right):
            quantized = []
            for n in raw:
                qs = quantize_ql(n["start"])
                qe = quantize_ql(n["end"])
                qdur = max(GRID_8TH, round((qe - qs) / GRID_8TH) * GRID_8TH)
                qdur = min(MAX_DUR_QL, qdur)
                quantized.append({**n, "q_start": qs, "q_dur": qdur})

            slots = defaultdict(list)
            for n in quantized:
                slot = round(n["q_start"] / GRID_8TH) * GRID_8TH
                slots[slot].append(n)

            result = []
            for slot in sorted(slots.keys()):
                notes = slots[slot]
                notes.sort(key=lambda x: x["velocity"], reverse=True)

                # Уникальные pitch
                seen = set()
                unique = []
                for note in notes:
                    if note["pitch"] not in seen:
                        seen.add(note["pitch"])
                        unique.append(note)
                notes = unique[:poly_max]

                # === ОГРАНИЧЕНИЕ ШИРИНЫ АККОРДА ===
                notes = limit_chord_span(notes, MAX_CHORD_SPAN)

                # Фильтр для левой руки
                if not is_right:
                    pitches_sorted = sorted(notes, key=lambda x: x["pitch"])
                    skip = set()
                    for i, a in enumerate(pitches_sorted):
                        if i in skip:
                            continue
                        for j, b in enumerate(pitches_sorted):
                            if i >= j or j in skip:
                                continue
                            if abs(a["pitch"] - b["pitch"]) == 1:
                                if a["velocity"] < b["velocity"]:
                                    skip.add(i)
                                else:
                                    skip.add(j)
                    filtered = [pitches_sorted[i] for i in range(len(pitches_sorted)) if i not in skip]
                    notes = filtered[:poly_max]

                    # M7 (11 полутонов) → октава
                    for n in notes:
                        for other in notes:
                            if n is not other and abs(n["pitch"] - other["pitch"]) == 11:
                                if n["pitch"] < other["pitch"]:
                                    n["pitch"] += 12

                    # Повторное ограничение ширины после транспозиции
                    notes = limit_chord_span(notes, MAX_CHORD_SPAN)

                result.extend(notes)
            return result

        right_events = prepare_hand(right_raw, RIGHT_POLY_MAX, True)
        left_events = prepare_hand(left_raw, LEFT_POLY_MAX, False)
        logger.info(f"[{job_id}] Правая: {len(right_events)}, Левая: {len(left_events)}")

        # 6. Создание партий
        def build_part(events, is_right):
            part = stream.Part()
            part.insert(0, instrument.Piano())
            part.insert(0, detected_key)
            part.insert(0, meter.TimeSignature("4/4"))
            part.insert(0, tempo.MetronomeMark(number=BPM))
            if not is_right:
                part.insert(0, clef.BassClef())

            events.sort(key=lambda x: x["q_start"])

            measures = defaultdict(list)
            for evt in events:
                start_ql = evt["q_start"]
                dur_ql = evt["q_dur"]
                pitch = evt["pitch"]
                vel = evt["velocity"]

                m_idx = int(start_ql // 4.0)
                m_offset = start_ql % 4.0

                if m_offset >= 3.999:
                    m_idx += 1
                    m_offset = 0.0

                remaining = dur_ql
                current_m = m_idx
                current_off = m_offset

                while remaining > 0.001 and current_m < 500:
                    space = 4.0 - current_off
                    chunk = min(remaining, space)
                    chunk = round(chunk / GRID_8TH) * GRID_8TH
                    chunk = max(GRID_8TH, chunk)

                    if chunk >= GRID_8TH and current_off < 4.0:
                        measures[current_m].append({
                            "offset": current_off,
                            "dur": chunk,
                            "pitch": pitch,
                            "velocity": vel,
                        })

                    remaining -= chunk
                    current_m += 1
                    current_off = 0.0

            for m_idx in sorted(measures.keys()):
                measure = stream.Measure(number=m_idx + 1)
                slot_notes = measures[m_idx]
                slot_notes.sort(key=lambda x: x["offset"])

                last_end = 0.0
                prev_notes = {}

                for nd in slot_notes:
                    offset = nd["offset"]
                    dur = nd["dur"]
                    pitch = nd["pitch"]
                    vel = nd["velocity"]

                    gap = offset - last_end
                    if gap >= GRID_8TH:
                        rest_ql = round(gap / GRID_8TH) * GRID_8TH
                        rest_ql = min(rest_ql, 4.0 - last_end)
                        if rest_ql >= GRID_8TH:
                            r = m21_note.Rest()
                            r.duration = m21_duration.Duration(quarterLength=rest_ql)
                            measure.insert(last_end, r)
                            last_end += rest_ql

                    dur = min(dur, 4.0 - offset)
                    if dur < GRID_8TH or offset >= 4.0:
                        continue

                    n = m21_note.Note(midi=pitch)
                    n.duration = m21_duration.Duration(quarterLength=dur)
                    n.volume.velocity = vel

                    if vel > avg_vel * 1.3:
                        n.articulations.append(articulations.Accent())
                    if dur <= 0.5:
                        n.articulations.append(articulations.Staccato())

                    if pitch in prev_notes:
                        prev = prev_notes[pitch]
                        if prev.offset + prev.duration.quarterLength >= offset - 0.01:
                            prev.tie = tie.Tie("start")
                            n.tie = tie.Tie("stop")

                    measure.insert(offset, n)
                    prev_notes[pitch] = n
                    last_end = max(last_end, offset + dur)

                part.append(measure)

            try:
                part.makeBeams(inPlace=True)
            except Exception as e:
                logger.warning(f"[{job_id}] makeBeams пропущен: {e}")
            return part

        right_part = build_part(right_events, True)
        left_part = build_part(left_events, False)

        score = stream.Score()
        score.insert(0, right_part)
        score.insert(0, left_part)

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
