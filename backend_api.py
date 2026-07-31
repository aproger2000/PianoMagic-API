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

app = FastAPI(title="PianoMagic API", version="3.6.0")

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
MIN_DUR_SEC = 0.10
MAX_DUR_QL = 4.0
RIGHT_POLY_MAX = 3
LEFT_POLY_MAX = 3
MAX_KEY_SHARPS = 3
MAX_CHORD_SPAN = 12  # 1 октава


@app.get("/")
async def root():
    return {"status": "ok", "version": "3.6.0"}


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


def limit_chord_span(notes, max_span):
    """Удаляет крайние ноты, пока размах <= max_span. Удаляет тишее из крайних."""
    if len(notes) <= 1:
        return notes
    notes = sorted(notes, key=lambda x: x["pitch"])
    while len(notes) > 1 and notes[-1]["pitch"] - notes[0]["pitch"] > max_span:
        if notes[0]["velocity"] <= notes[-1]["velocity"]:
            notes.pop(0)
        else:
            notes.pop()
    return notes


def process_audio(job_id: str, input_path: Path, xml_path: Path, pdf_path: Path):
    try:
        logger.info(f"[{job_id}] === Начало v3.6.0 (полифония, 2 руки) ===")

        from basic_pitch.inference import predict
        from music21 import (
            stream, instrument, note as m21_note, chord as m21_chord, clef,
            tempo, meter, key, duration as m21_duration, articulations,
            tie, pitch as m21_pitch
        )

        # 1. Basic Pitch
        _, _, note_events = predict(str(input_path))
        logger.info(f"[{job_id}] Всего нот: {len(note_events)}")

        if not note_events:
            raise ValueError("Ноты не найдены")

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
                "start": float(start),
                "end": float(end),
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
        except Exception:
            detected_key = key.Key("C")

        sharps = detected_key.sharps
        if abs(sharps) > MAX_KEY_SHARPS:
            candidates = [0, 1, 2, 3, -1, -2, -3]
            best = min(candidates, key=lambda c: abs(c - sharps))
            detected_key = key.Key(m21_pitch.Pitch(sharps=best).name)
            sharps = best

        logger.info(f"[{job_id}] Тональность: {detected_key.name} ({sharps} знаков)")

        # 4. Квантизация
        def quantize_ql(t):
            ql = t / SEC_PER_QUARTER
            return round(ql / GRID_8TH) * GRID_8TH

        quantized = []
        for n in notes_raw:
            qs = quantize_ql(n["start"])
            qe = quantize_ql(n["end"])
            qdur = max(GRID_8TH, round((qe - qs) / GRID_8TH) * GRID_8TH)
            qdur = min(MAX_DUR_QL, qdur)
            quantized.append({
                "q_start": qs,
                "q_dur": qdur,
                "pitch": n["pitch"],
                "velocity": n["velocity"],
            })

        # 5. Группировка по слотам и разделение на руки
        slots = defaultdict(list)
        for n in quantized:
            slot = round(n["q_start"] / GRID_8TH) * GRID_8TH
            slots[slot].append(n)

        right_events = []
        left_events = []

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
            notes = unique

            # Разделение: ≥ C4 → правая, < C4 → левая
            right_candidates = [n for n in notes if n["pitch"] >= 60]
            left_candidates = [n for n in notes if n["pitch"] < 60]

            # === ПРАВАЯ РУКА ===
            right = right_candidates[:RIGHT_POLY_MAX]
            right = limit_chord_span(right, MAX_CHORD_SPAN)
            for n in right:
                right_events.append({**n, "q_start": slot})

            # === ЛЕВАЯ РУКА ===
            left = left_candidates[:LEFT_POLY_MAX]

            # Фильтр m2 в левой руке
            left_sorted = sorted(left, key=lambda x: x["pitch"])
            skip = set()
            for i, a in enumerate(left_sorted):
                if i in skip:
                    continue
                for j, b in enumerate(left_sorted):
                    if i >= j or j in skip:
                        continue
                    if abs(a["pitch"] - b["pitch"]) == 1:
                        if a["velocity"] < b["velocity"]:
                            skip.add(i)
                        else:
                            skip.add(j)
            left = [left_sorted[i] for i in range(len(left_sorted)) if i not in skip]
            left = left[:LEFT_POLY_MAX]

            # M7 (11 полутонов) → октава
            for n in left:
                for other in left:
                    if n is not other and abs(n["pitch"] - other["pitch"]) == 11:
                        if n["pitch"] < other["pitch"]:
                            n["pitch"] += 12

            left = limit_chord_span(left, MAX_CHORD_SPAN)
            for n in left:
                left_events.append({**n, "q_start": slot})

        logger.info(f"[{job_id}] Правая: {len(right_events)} нот/акк, Левая: {len(left_events)} нот/акк")

        # 6. Создание нотного текста
        def build_part(events, is_right):
            part = stream.Part()
            part.insert(0, instrument.Piano())
            part.insert(0, detected_key)
            part.insert(0, meter.TimeSignature("4/4"))
            part.insert(0, tempo.MetronomeMark(number=BPM))
            if is_right:
                part.insert(0, clef.TrebleClef())
            else:
                part.insert(0, clef.BassClef())

            # Группировка по тактам
            measures = defaultdict(list)
            for evt in events:
                m_idx = int(evt["q_start"] // 4.0)
                off = evt["q_start"] % 4.0
                if off >= 3.999:
                    m_idx += 1
                    off = 0.0
                measures[m_idx].append({**evt, "offset": off})

            for m_idx in sorted(measures.keys()):
                measure = stream.Measure(number=m_idx + 1)
                slot_notes = sorted(measures[m_idx], key=lambda x: x["offset"])

                last_end = 0.0
                prev_pitches = {}  # pitch -> note object for tie

                for nd in slot_notes:
                    off = nd["offset"]
                    dur = min(nd["q_dur"], 4.0 - off)
                    if dur < GRID_8TH:
                        continue

                    # Пауза
                    gap = off - last_end
                    if gap >= GRID_8TH:
                        rest_ql = round(gap / GRID_8TH) * GRID_8TH
                        rest_ql = min(rest_ql, 4.0 - last_end)
                        if rest_ql >= GRID_8TH:
                            r = m21_note.Rest()
                            r.duration = m21_duration.Duration(quarterLength=rest_ql)
                            measure.insert(last_end, r)
                            last_end += rest_ql

                    # Создаём ноту или аккорд
                    # Собираем все ноты на этом offset (они уже сгруппированы по слотам,
                    # но могут быть разные pitch на один offset)
                    # На самом деле events уже по одной ноте, но мы можем собрать аккорд
                    # если несколько нот на один offset
                    pass  # будет ниже

                # Пересобираем: группируем по offset внутри такта
                offset_groups = defaultdict(list)
                for nd in slot_notes:
                    offset_groups[nd["offset"]].append(nd)

                last_end = 0.0
                prev_objects = {}  # pitch -> last note/chord for tie

                for off in sorted(offset_groups.keys()):
                    group = offset_groups[off]
                    dur = min(group[0]["q_dur"], 4.0 - off)
                    if dur < GRID_8TH:
                        continue

                    # Пауза
                    gap = off - last_end
                    if gap >= GRID_8TH:
                        rest_ql = round(gap / GRID_8TH) * GRID_8TH
                        rest_ql = min(rest_ql, 4.0 - last_end)
                        if rest_ql >= GRID_8TH:
                            r = m21_note.Rest()
                            r.duration = m21_duration.Duration(quarterLength=rest_ql)
                            measure.insert(last_end, r)
                            last_end += rest_ql

                    # Создаём элемент
                    pitches = [g["pitch"] for g in group]
                    velocities = [g["velocity"] for g in group]
                    vel = max(velocities)

                    if len(pitches) == 1:
                        n = m21_note.Note(midi=pitches[0])
                    else:
                        n = m21_chord.Chord(pitches)

                    n.duration = m21_duration.Duration(quarterLength=dur)
                    n.volume.velocity = vel

                    # Акцент
                    if vel > avg_vel * 1.3:
                        n.articulations.append(articulations.Accent())

                    # Лиги для каждого pitch в аккорде
                    for p in pitches:
                        if p in prev_objects:
                            prev = prev_objects[p]
                            if prev.offset + prev.duration.quarterLength >= off - 0.01:
                                if hasattr(prev, 'tie') and prev.tie:
                                    if prev.tie.type == 'start':
                                        pass  # уже есть
                                    else:
                                        prev.tie = tie.Tie("start")
                                else:
                                    prev.tie = tie.Tie("start")
                                if hasattr(n, 'tie') and n.tie:
                                    pass
                                else:
                                    n.tie = tie.Tie("stop")  # для аккорда — сложнее
                                # music21 chord tie — применяем к pitch
                                # Проще: не делаем лиги для аккордов, только моно
                                pass

                    # Упрощение: лиги только для монофонических переходов
                    # (в полифонии лиги редко нужны для одного pitch)

                    measure.insert(off, n)
                    for p in pitches:
                        prev_objects[p] = n
                    last_end = off + dur

                part.append(measure)

            try:
                part.makeBeams(inPlace=True)
            except Exception:
                pass
            return part

        right_part = build_part(right_events, True)
        left_part = build_part(left_events, False)

        score = stream.Score()
        score.insert(0, right_part)
        score.insert(0, left_part)

        # 7. MusicXML → PDF
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
