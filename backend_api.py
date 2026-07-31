import os
import uuid
import shutil
import subprocess
import logging
from pathlib import Path
from collections import defaultdict
from xml.etree.ElementTree import Element, SubElement, tostring

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PianoMagic API", version="3.7.0")

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
RIGHT_POLY_MAX = 3
LEFT_POLY_MAX = 3
MAX_CHORD_SPAN = 12
DIVISIONS = 2  # quarter = 2, eighth = 1, half = 4, whole = 8

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


@app.get("/")
async def root():
    return {"status": "ok", "version": "3.7.0"}


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


def midi_to_pitch(midi):
    name = NOTE_NAMES[midi % 12]
    octave = (midi // 12) - 1
    if '#' in name:
        return name[0], 1, octave
    return name, 0, octave


def dur_to_type(dur):
    mapping = {8: 'whole', 6: 'half', 4: 'half', 3: 'quarter', 2: 'quarter', 1: 'eighth'}
    return mapping.get(dur, 'quarter')


def limit_chord_span(notes, max_span):
    if len(notes) <= 1:
        return notes
    notes = sorted(notes, key=lambda x: x["pitch"])
    while len(notes) > 1 and notes[-1]["pitch"] - notes[0]["pitch"] > max_span:
        if notes[0]["velocity"] <= notes[-1]["velocity"]:
            notes.pop(0)
        else:
            notes.pop()
    return notes


def build_musicxml(right_measures, left_measures, key_fifths=0):
    """Generate MusicXML 3.1 from measure data."""
    root = Element('score-partwise', {'version': '3.1'})

    # Part list
    plist = SubElement(root, 'part-list')
    for pid, pname, abbrev in [('P1', 'Right Hand', 'R.H.'), ('P2', 'Left Hand', 'L.H.')]:
        sp = SubElement(plist, 'score-part', {'id': pid})
        SubElement(sp, 'part-name').text = pname
        SubElement(sp, 'part-abbreviation').text = abbrev

    def add_part(pid, measures, clef_sign, clef_line):
        part = SubElement(root, 'part', {'id': pid})
        for m_idx, events in enumerate(measures):
            measure = SubElement(part, 'measure', {'number': str(m_idx + 1)})

            if m_idx == 0:
                attr = SubElement(measure, 'attributes')
                SubElement(attr, 'divisions').text = str(DIVISIONS)
                k = SubElement(attr, 'key')
                SubElement(k, 'fifths').text = str(key_fifths)
                t = SubElement(attr, 'time')
                SubElement(t, 'beats').text = '4'
                SubElement(t, 'beat-type').text = '4'
                c = SubElement(attr, 'clef')
                SubElement(c, 'sign').text = clef_sign
                SubElement(c, 'line').text = str(clef_line)

            events = sorted(events, key=lambda x: x['offset'])
            last_end = 0

            for evt in events:
                off = evt['offset']
                dur = evt['dur']
                pitches = evt['pitches']

                # Rest for gap
                gap = off - last_end
                if gap >= 1:
                    r = SubElement(measure, 'note')
                    SubElement(r, 'rest')
                    SubElement(r, 'duration').text = str(gap)
                    SubElement(r, 'type').text = dur_to_type(gap)

                # Note / Chord
                for i, p in enumerate(pitches):
                    n = SubElement(measure, 'note')
                    if i > 0:
                        SubElement(n, 'chord')

                    step, alter, octv = midi_to_pitch(p)
                    pitch_el = SubElement(n, 'pitch')
                    SubElement(pitch_el, 'step').text = step
                    if alter != 0:
                        SubElement(pitch_el, 'alter').text = str(alter)
                    SubElement(pitch_el, 'octave').text = str(octv)

                    SubElement(n, 'duration').text = str(dur)
                    SubElement(n, 'type').text = dur_to_type(dur)

                last_end = off + dur

            # Fill measure to 8 divisions
            fill = 8 - last_end
            if fill >= 1:
                r = SubElement(measure, 'note')
                SubElement(r, 'rest')
                SubElement(r, 'duration').text = str(fill)
                SubElement(r, 'type').text = dur_to_type(fill)

    add_part('P1', right_measures, 'G', 2)
    add_part('P2', left_measures, 'F', 4)

    xml_str = tostring(root, encoding='unicode')
    doctype = '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">\n'
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + doctype + xml_str


def process_audio(job_id: str, input_path: Path, xml_path: Path, pdf_path: Path):
    try:
        logger.info(f"[{job_id}] === Начало v3.7.0 (ручной MusicXML) ===")

        from basic_pitch.inference import predict
        _, _, note_events = predict(str(input_path))
        logger.info(f"[{job_id}] Всего нот: {len(note_events)}")

        if not note_events:
            raise ValueError("Ноты не найдены")

        # Фильтрация
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

        logger.info(f"[{job_id}] После фильтрации: {len(notes_raw)}")

        # Квантизация
        def q(t):
            return round((t / SEC_PER_QUARTER) / GRID_8TH) * GRID_8TH

        quantized = []
        for n in notes_raw:
            qs = q(n["start"])
            qe = q(n["end"])
            qdur = max(GRID_8TH, round((qe - qs) / GRID_8TH) * GRID_8TH)
            qdur = min(4.0, qdur)
            quantized.append({
                "q_start": qs,
                "dur_ql": qdur,
                "pitch": n["pitch"],
                "velocity": n["velocity"],
            })

        # Слоты (1/8)
        slots = defaultdict(list)
        for n in quantized:
            slot = round(n["q_start"] / GRID_8TH) * GRID_8TH
            slots[slot].append(n)

        right_events = []
        left_events = []

        for slot in sorted(slots.keys()):
            notes = slots[slot]
            notes.sort(key=lambda x: x["velocity"], reverse=True)

            seen = set()
            unique = []
            for note in notes:
                if note["pitch"] not in seen:
                    seen.add(note["pitch"])
                    unique.append(note)
            notes = unique

            right_cand = [n for n in notes if n["pitch"] >= 60]
            left_cand = [n for n in notes if n["pitch"] < 60]

            # Правая рука
            right = right_cand[:RIGHT_POLY_MAX]
            right = limit_chord_span(right, MAX_CHORD_SPAN)
            if right:
                right_events.append({
                    "start": slot,
                    "dur": right[0]["dur_ql"],
                    "pitches": [n["pitch"] for n in right]
                })

            # Левая рука
            left = left_cand[:LEFT_POLY_MAX]
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

            for n in left:
                for other in left:
                    if n is not other and abs(n["pitch"] - other["pitch"]) == 11:
                        if n["pitch"] < other["pitch"]:
                            n["pitch"] += 12

            left = limit_chord_span(left, MAX_CHORD_SPAN)
            if left:
                left_events.append({
                    "start": slot,
                    "dur": left[0]["dur_ql"],
                    "pitches": [n["pitch"] for n in left]
                })

        logger.info(f"[{job_id}] Правая: {len(right_events)}, Левая: {len(left_events)}")

        # В такты (4/4 = 8 divisions)
        def to_measures(events):
            measures = defaultdict(list)
            for evt in events:
                m = int(evt["start"] // 4.0)
                off = evt["start"] % 4.0
                if off >= 3.999:
                    m += 1
                    off = 0.0
                dur_div = max(1, int(round(evt["dur"] * DIVISIONS)))
                measures[m].append({
                    "offset": int(round(off * DIVISIONS)),
                    "dur": dur_div,
                    "pitches": evt["pitches"],
                })
            return [measures[i] for i in sorted(measures.keys())]

        right_m = to_measures(right_events)
        left_m = to_measures(left_events)

        # MusicXML
        xml_content = build_musicxml(right_m, left_m, key_fifths=0)
        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write(xml_content)

        logger.info(f"[{job_id}] MusicXML записан")

        # MuseScore3 → PDF
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
