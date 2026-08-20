"""
PianoMagic Backend API — v7.6.2
Audio-to-piano-score transcription
"""

import os
import uuid
import asyncio
import tempfile
import traceback
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import librosa
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom

import soundfile as sf
from scipy.ndimage import median_filter

# v7.6.0: Basic Pitch (Spotify's ICASSP-2022 neural AMT model) is the
# primary transcription engine. It is imported defensively so that a
# deploy where the wheel failed to install still boots and serves - it
# just silently falls back to the old librosa/PYIN path instead of
# 500-ing on every upload. Check /version at runtime to see which
# engine a given deploy actually ended up with.
BASIC_PITCH_AVAILABLE = False
_BASIC_PITCH_IMPORT_ERROR = None
# v7.6.1: importing basic-pitch successfully does NOT mean it can actually
# run - loading the ONNX/TFLite graph and doing inference happen later, per
# request. v7.6.0 caught those runtime failures and silently fell back to
# PYIN, so /version reported engine="basic-pitch" while every transcription
# was really produced by the old monophonic path; the output was byte-for-byte
# identical to v7.5.0 and there was no way to tell from outside. The last
# runtime failure is now recorded here and surfaced on /version and on the
# task result, so a silent fallback can't masquerade as a working engine.
LAST_ENGINE_ERROR = None
_BP_MODEL = None            # cached loaded model (loading it is not cheap)
_BP_MODEL_PATH_USED = None  # which serialized graph actually loaded
_BP_MODEL_LOAD_ERROR = None
_BP_MODEL_CANDIDATES = []
try:
    from basic_pitch.inference import predict as _bp_predict, Model as _BPModel
    from basic_pitch import ICASSP_2022_MODEL_PATH as _BP_MODEL_PATH
    BASIC_PITCH_AVAILABLE = True
    print("[INIT] basic-pitch available - using neural polyphonic transcription")
except Exception as _e:  # ImportError, or a backend/runtime that failed to load
    _BASIC_PITCH_IMPORT_ERROR = repr(_e)
    print(f"[INIT] basic-pitch NOT available ({_e!r}) - falling back to librosa PYIN")


def _bp_model_candidates() -> list:
    """
    v7.6.2: choose the serialized model ourselves instead of taking
    basic-pitch's ICASSP_2022_MODEL_PATH at face value.

    basic-pitch picks a graph by runtime priority TF > CoreML > TFLite >
    ONNX. This image has BOTH tflite-runtime and onnxruntime installed
    (tflite-runtime arrives as a default dependency on Linux), so it
    selected nmp.tflite - and that graph then refused to load:

        ValueError: File .../nmp.tflite cannot be loaded into either
        TensorFlow, CoreML, TFLite or ONNX. On this system,
        ['TensorFlowLite', 'ONNX'] is installed.

    tflite-runtime 2.14 predates NumPy 2.x, which this image has (2.2.6),
    so its interpreter can't initialise. Because the selection happens
    before any load is attempted, basic-pitch never fell through to the
    ONNX graph that would have worked, and every request died there.

    Preferring ONNX explicitly - and then actually trying each candidate
    until one loads - makes the choice depend on what works rather than
    on which packages merely happen to be importable.
    """
    out, seen = [], set()

    def add(p):
        if p is None:
            return
        s = str(p)
        if s not in seen:
            seen.add(s)
            out.append(p)

    try:
        from basic_pitch import build_icassp_2022_model_path, FilenameSuffix
        # onnx first: onnxruntime is the runtime we install and verify.
        for name in ("onnx", "tf", "tflite", "coreml"):
            suffix = getattr(FilenameSuffix, name, None)
            if suffix is None:
                continue
            try:
                add(build_icassp_2022_model_path(suffix))
            except Exception:
                pass
    except Exception:
        pass

    # Whatever basic-pitch itself would have used, as a last resort.
    try:
        add(_BP_MODEL_PATH)
    except Exception:
        pass
    return out


def _get_bp_model():
    """Load (once) the first candidate graph that actually initialises."""
    global _BP_MODEL, _BP_MODEL_PATH_USED, _BP_MODEL_LOAD_ERROR, _BP_MODEL_CANDIDATES
    if _BP_MODEL is not None:
        return _BP_MODEL

    _BP_MODEL_CANDIDATES = _bp_model_candidates()
    attempts = []
    for path in _BP_MODEL_CANDIDATES:
        if not Path(str(path)).exists():
            attempts.append(f"  {path} -> not present in the installed package")
            continue
        try:
            model = _BPModel(path)
        except Exception as e:
            attempts.append(f"  {path} -> {e!r}")
            continue
        _BP_MODEL = model
        _BP_MODEL_PATH_USED = str(path)
        _BP_MODEL_LOAD_ERROR = None
        print(f"[INIT] basic-pitch model loaded: {path}")
        return _BP_MODEL

    _BP_MODEL_LOAD_ERROR = ("No basic-pitch model graph could be loaded.\n"
                            + "\n".join(attempts))
    raise RuntimeError(_BP_MODEL_LOAD_ERROR)

# ───────────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────────
VERSION = "7.6.2"
UPLOAD_DIR = Path(tempfile.gettempdir()) / "pianomagic_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
print(f"[INIT] UPLOAD_DIR={UPLOAD_DIR}, exists={UPLOAD_DIR.exists()}")

tasks: Dict[str, dict] = {}

# ───────────────────────────────────────────────────────────────
# Diagnostics / per-task logging (v7.6.1)
# ───────────────────────────────────────────────────────────────
# Every stage of the pipeline already print()s what it's doing, but those
# lines only ever reached the Render console - invisible from the browser.
# When v7.6.0's neural engine failed at runtime and silently fell back,
# the UI looked completely normal and the regression was only detectable
# by byte-comparing two downloads. These helpers tee the same output into
# a per-task buffer that the frontend can show and copy, so what actually
# happened during a run is visible without server access.

def _env_report() -> List[str]:
    """Versions and engine state - the first thing worth knowing when a
    run misbehaves, and the thing that's hardest to guess from outside."""
    import platform
    lines = [
        f"PianoMagic backend      : v{VERSION}",
        f"Python                  : {platform.python_version()} ({platform.platform()})",
    ]
    for mod in ("numpy", "scipy", "librosa", "soundfile", "basic_pitch", "onnxruntime",
                "tensorflow", "tflite_runtime"):
        try:
            m = __import__(mod)
            ver = getattr(m, '__version__', None)
            if ver is None:
                # basic_pitch exposes no __version__; ask the package metadata
                # instead of reporting a useless "unknown".
                try:
                    from importlib.metadata import version as _pkgver
                    ver = _pkgver(mod.replace('_', '-'))
                except Exception:
                    ver = 'installed (version unknown)'
            lines.append(f"{mod:<24}: {ver}")
        except Exception:
            lines.append(f"{mod:<24}: NOT INSTALLED")
    lines.append(f"basic-pitch importable  : {BASIC_PITCH_AVAILABLE}")
    if _BASIC_PITCH_IMPORT_ERROR:
        lines.append(f"basic-pitch import error: {_BASIC_PITCH_IMPORT_ERROR}")
    if BASIC_PITCH_AVAILABLE:
        # Which graphs actually ship in this install, and which one we
        # selected. When the engine falls over, this is the difference
        # between "the model file isn't there" and "the runtime can't
        # read it" - two problems with completely different fixes.
        try:
            lines.append(f"basic-pitch default path: {_BP_MODEL_PATH}")
        except Exception:
            pass
        try:
            for cand in (_BP_MODEL_CANDIDATES or _bp_model_candidates()):
                mark = "present" if Path(str(cand)).exists() else "MISSING"
                lines.append(f"  candidate [{mark:<7}]   : {cand}")
        except Exception:
            pass
        lines.append(f"model loaded from       : {_BP_MODEL_PATH_USED or '(not loaded yet)'}")
        if _BP_MODEL_LOAD_ERROR:
            lines.append(f"model load error        : {_BP_MODEL_LOAD_ERROR}")
    return lines

class _TeeLog:
    """Writes to the real stdout AND into a task's log buffer.

    Used with contextlib.redirect_stdout so the pipeline's existing
    print() calls are captured without having to rewrite every one of
    them - which also means any future print() is logged automatically
    rather than being forgotten.
    """
    def __init__(self, buf: List[str], stream):
        self._buf = buf
        self._stream = stream
        self._partial = ''

    def write(self, text):
        try:
            self._stream.write(text)
        except Exception:
            pass
        self._partial += text
        while '\n' in self._partial:
            line, self._partial = self._partial.split('\n', 1)
            self._buf.append(line)
        return len(text)

    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass

# ───────────────────────────────────────────────────────────────
# FastAPI App
# ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="PianoMagic API",
    description="Audio-to-piano-score transcription API",
    version=VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ───────────────────────────────────────────────────────────────
# Data Models
# ───────────────────────────────────────────────────────────────
@dataclass
class Note:
    start: float
    end: float
    pitch_midi: int
    velocity: int = 80
    hand: str = "RH"

@dataclass
class TranscriptionResult:
    notes: List[Note] = field(default_factory=list)
    tempo: float = 120.0
    key: str = "C major"
    duration: float = 0.0
    sr: int = 22050

# ───────────────────────────────────────────────────────────────
# Utility Functions
# ───────────────────────────────────────────────────────────────
def midi_to_ly_step(midi: int) -> tuple:
    names = ['C', 'C', 'D', 'D', 'E', 'F', 'F', 'G', 'G', 'A', 'A', 'B']
    alters = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
    octave = (midi // 12) - 1
    idx = midi % 12
    return names[idx], alters[idx], octave

def estimate_key(chroma: np.ndarray) -> str:
    profiles = {
        'C major': [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        'C minor': [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    }
    all_profiles = {}
    for i in range(12):
        for mode, prof in profiles.items():
            key_name = librosa.midi_to_note(i + 60)[:-1]
            if mode == 'C major':
                all_profiles[f"{key_name} major"] = np.roll(prof, i)
            else:
                all_profiles[f"{key_name} minor"] = np.roll(prof, i)

    chroma_mean = chroma.mean(axis=1)
    correlations = {k: np.corrcoef(chroma_mean, v)[0, 1] for k, v in all_profiles.items()}
    return max(correlations, key=correlations.get)

# ───────────────────────────────────────────────────────────────
# Core Algorithm: v7.6.0 (Basic Pitch neural polyphonic + PYIN fallback)
# ───────────────────────────────────────────────────────────────
def smooth_pitch(voice: np.ndarray, window: int = 7) -> np.ndarray:
    """Median filter to remove vibrato artifacts."""
    valid = ~np.isnan(voice)
    if np.sum(valid) < window:
        return voice
    result = voice.copy()
    valid_indices = np.where(valid)[0]
    if len(valid_indices) >= window:
        smoothed = median_filter(voice[valid_indices], size=window, mode='nearest')
        result[valid_indices] = smoothed
    return result

def fill_short_gaps(voice: np.ndarray, times: np.ndarray, max_gap_ms: float = 100) -> np.ndarray:
    result = voice.copy()
    valid = ~np.isnan(voice)
    if np.sum(valid) < 2:
        return result

    gaps = np.diff(valid.astype(int), prepend=0, append=0)
    gap_starts = np.where(gaps == -1)[0]
    gap_ends = np.where(gaps == 1)[0]

    sr = 22050
    hop_length = 512
    for gs, ge in zip(gap_starts, gap_ends):
        if ge <= gs:
            continue
        gap_dur_ms = (ge - gs) * hop_length / sr * 1000
        if gap_dur_ms <= max_gap_ms and gs > 0 and ge < len(voice):
            result[gs:ge] = np.linspace(voice[gs-1], voice[ge], ge - gs)
    return result

def segment_pitch_contour(
    filled_voice: np.ndarray,
    times: np.ndarray,
    hand: str,
    min_dur_ms: float = 120,
    pause_thresh_ms: float = 250,
    pitch_jump_st: float = 1.5
) -> List[Note]:
    """
    v7.2.1: More relaxed thresholds to avoid fragmentation.
    """
    notes = []
    in_note = False
    note_start = 0.0
    current_pitch = 0.0

    for i in range(len(filled_voice)):
        if not np.isnan(filled_voice[i]):
            pitch = filled_voice[i]
            if not in_note:
                in_note = True
                note_start = times[i]
                current_pitch = pitch
            else:
                semitone_diff = abs(librosa.hz_to_midi(pitch) - librosa.hz_to_midi(current_pitch))
                if semitone_diff > pitch_jump_st:
                    note_end = times[i]
                    dur_ms = (note_end - note_start) * 1000
                    if dur_ms >= min_dur_ms and note_end > note_start:
                        notes.append(Note(
                            start=note_start,
                            end=note_end,
                            pitch_midi=int(round(librosa.hz_to_midi(current_pitch))),
                            hand=hand
                        ))
                    note_start = times[i]
                    current_pitch = pitch
        else:
            if in_note:
                j = i
                while j < len(filled_voice) and np.isnan(filled_voice[j]):
                    j += 1
                pause_ms = (times[min(j, len(times)-1)] - times[i]) * 1000 if j < len(times) else 9999
                if pause_ms > pause_thresh_ms or j >= len(times):
                    note_end = times[i]
                    dur_ms = (note_end - note_start) * 1000
                    if dur_ms >= min_dur_ms and note_end > note_start:
                        notes.append(Note(
                            start=note_start,
                            end=note_end,
                            pitch_midi=int(round(librosa.hz_to_midi(current_pitch))),
                            hand=hand
                        ))
                    in_note = False

    if in_note:
        note_end = times[-1]
        dur_ms = (note_end - note_start) * 1000
        if dur_ms >= min_dur_ms and note_end > note_start:
            notes.append(Note(
                start=note_start,
                end=note_end,
                pitch_midi=int(round(librosa.hz_to_midi(current_pitch))),
                hand=hand
            ))

    return notes

def merge_close_notes(notes: List[Note], max_gap_ms: float = 150, max_pitch_diff_st: float = 1.5) -> List[Note]:
    if not notes:
        return notes
    notes_sorted = sorted(notes, key=lambda n: n.start)
    merged = [notes_sorted[0]]
    for note in notes_sorted[1:]:
        last = merged[-1]
        gap_ms = (note.start - last.end) * 1000
        pitch_diff = abs(note.pitch_midi - last.pitch_midi)
        if gap_ms < max_gap_ms and pitch_diff <= max_pitch_diff_st and last.hand == note.hand:
            last.end = note.end
            # Keep the longer note's pitch (more stable)
            last_dur = last.end - last.start
            new_dur = note.end - note.start
            if new_dur > last_dur:
                last.pitch_midi = note.pitch_midi
        else:
            merged.append(note)
    return merged

def quantize_notes(notes: List[Note], time_grid_ms: float = 50) -> List[Note]:
    """Quantize note boundaries to reduce micro-timing jitter."""
    for n in notes:
        n.start = round(n.start * 1000 / time_grid_ms) * time_grid_ms / 1000
        n.end = round(n.end * 1000 / time_grid_ms) * time_grid_ms / 1000
        if n.end <= n.start:
            n.end = n.start + 0.12  # minimum 120ms
    return notes

def _two_means_split_midi(midi_vals: np.ndarray, fallback: int = 60) -> int:
    """
    Find a natural boundary between the two dominant registers present in
    a set of MIDI pitch values, via a simple 1D 2-means. Used to decide
    where this particular track's RH/LH divide sits instead of always
    assuming middle C. Clamped to C3..C6 so one odd track can't push the
    boundary somewhere that leaves a hand empty by construction.
    """
    midi_vals = np.asarray(midi_vals, dtype=float)
    midi_vals = midi_vals[np.isfinite(midi_vals)]
    if len(midi_vals) < 40:
        return fallback

    c1, c2 = np.percentile(midi_vals, 25), np.percentile(midi_vals, 75)
    for _ in range(25):
        d1 = np.abs(midi_vals - c1)
        d2 = np.abs(midi_vals - c2)
        g1 = midi_vals[d1 <= d2]
        g2 = midi_vals[d1 > d2]
        if len(g1) == 0 or len(g2) == 0:
            break
        n1, n2 = g1.mean(), g2.mean()
        done = abs(n1 - c1) < 0.01 and abs(n2 - c2) < 0.01
        c1, c2 = n1, n2
        if done:
            break

    lo, hi = sorted([c1, c2])
    return max(48, min(72, int(round((lo + hi) / 2))))

def _reduce_polyphony_to_two_voices(note_events, min_dur_ms: float = 80.0,
                                    amp_frac_of_peak: float = 0.10):
    """
    v7.6.0: turn Basic Pitch's polyphonic note list into the two
    monophonic voices a two-staff piano score can actually represent.

    Basic Pitch returns EVERY note it hears - on a full arrangement
    (vocal + accompaniment + percussion) that's hundreds of overlapping
    events spanning the whole keyboard. Two things force a reduction:
    musically, a dump of every detected partial is unreadable as sheet
    music; mechanically, generate_musicxml_v72's emit_voice() walks a
    single forward-only cursor per voice, so it cannot represent two
    notes overlapping *within* one voice at all.

    So we take the standard melody/bass skeleton: on a fine time grid,
    the top sounding pitch becomes the right hand and the bottom
    sounding pitch becomes the left hand. That's the reduction a person
    arranging a song for piano would start from, and it's what makes the
    result recognisable as the tune rather than as a cloud of notes.

    Quiet detections are dropped first, since those are mostly percussion
    smear and reverb tails rather than real notes. The threshold is a
    fraction of the LOUDEST detection, deliberately not a percentile of
    the amplitude distribution: percussion-heavy input can produce more
    spurious notes than real ones, and a percentile gate silently moves
    with them, so the junk survives its own gate. That failure mode is
    especially damaging here because the right hand is taken as the top
    sounding pitch - a swarm of weak high-register artefacts would
    become "the melody" and reproduce exactly the noise this engine is
    meant to eliminate. A peak-relative floor can't drift that way.
    """
    if not note_events:
        return []

    amps = np.array([ev[3] for ev in note_events], dtype=float)
    amp_peak = float(amps.max()) if len(amps) else 0.0
    amp_floor = amp_frac_of_peak * amp_peak
    events = [ev for ev in note_events if ev[3] >= amp_floor and ev[1] > ev[0]]
    if not events:
        events = [ev for ev in note_events if ev[1] > ev[0]]
    if not events:
        return []

    pitches = np.array([ev[2] for ev in events], dtype=float)
    split_midi = _two_means_split_midi(pitches)
    print(f"[EXTRACT] basic-pitch: {len(note_events)} raw events, {len(events)} after "
          f"amplitude gate, RH/LH split at MIDI {split_midi}")

    grid = 0.01  # 10 ms
    t_end = max(ev[1] for ev in events)
    n_frames = int(np.ceil(t_end / grid)) + 1
    if n_frames <= 0 or n_frames > 2_000_000:
        return []

    top = np.full(n_frames, -1, dtype=int)
    bottom = np.full(n_frames, -1, dtype=int)

    for start, end, pitch, _amp, *_rest in events:
        i0 = max(0, int(np.floor(start / grid)))
        i1 = min(n_frames, int(np.ceil(end / grid)))
        if i1 <= i0:
            continue
        seg_top = top[i0:i1]
        seg_bot = bottom[i0:i1]
        np.maximum(seg_top, pitch, out=seg_top)
        empty = seg_bot < 0
        seg_bot[empty] = pitch
        np.minimum(seg_bot, pitch, out=seg_bot)

    # Where only ONE pitch sounds, top == bottom. Give that note to the
    # hand its register belongs to rather than duplicating it on both
    # staves (which would read as a phantom unison).
    same = (top == bottom) & (top >= 0)
    rh_line = top.copy()
    lh_line = bottom.copy()
    rh_line[same & (top < split_midi)] = -1
    lh_line[same & (top >= split_midi)] = -1

    def runs_to_notes(line, hand):
        notes = []
        i = 0
        n = len(line)
        while i < n:
            if line[i] < 0:
                i += 1
                continue
            j = i
            while j < n and line[j] == line[i]:
                j += 1
            dur_ms = (j - i) * grid * 1000
            if dur_ms >= min_dur_ms:
                notes.append(Note(start=i * grid, end=j * grid,
                                  pitch_midi=int(line[i]), hand=hand))
            i = j
        return notes

    rh = runs_to_notes(rh_line, "RH")
    lh = runs_to_notes(lh_line, "LH")
    print(f"[EXTRACT] basic-pitch reduction: RH={len(rh)} notes, LH={len(lh)} notes")
    return sorted(rh + lh, key=lambda n: n.start)

def extract_melody_basic_pitch(file_path, y: np.ndarray, sr: int) -> TranscriptionResult:
    """
    v7.6.0: primary engine. Basic Pitch is a neural polyphonic
    transcription model, which is the actual fix for this service's
    long-standing "output is just noise" problem.

    Every previous version tracked pitch with librosa.pyin, which is
    monophonic *by construction*: it estimates at most one f0 per frame.
    Measuring the real test file showed ~10 simultaneous fundamentals in
    99% of frames - i.e. a full arrangement, not a solo line. Handed
    that, PYIN cannot report the melody plus the accompaniment; it
    reports one pitch per instant and, where instruments compete, that
    pitch lands on whichever partial happens to dominate. That is the
    root cause the earlier fixes (register-restricted passes, adaptive
    split, single broadband pass) were all working around rather than
    addressing - none of them could have succeeded, because the
    information was being discarded inside PYIN itself.

    Tempo and key still come from librosa, which is fine - those are
    global spectral/rhythmic statistics, not per-note decisions.
    """
    duration = librosa.get_duration(y=y, sr=sr)

    # Hand Basic Pitch a plain 22.05 kHz mono WAV decoded by us, rather
    # than the raw upload. The upload may be .mp3/.m4a/.ogg, and letting
    # the library re-open and re-decode it adds a failure mode we've
    # already cleared - librosa decoded this same file moments ago in
    # process_audio. 22050 Hz mono is exactly what the model consumes
    # internally, so this also skips a redundant resample.
    # Load the model before writing anything, so a model problem is
    # reported as a model problem rather than after a pile of I/O.
    model = _get_bp_model()
    print(f"[EXTRACT] Basic Pitch model: {_BP_MODEL_PATH_USED}")

    tmp_wav = UPLOAD_DIR / f"_bp_input_{uuid.uuid4().hex}.wav"
    try:
        sf.write(str(tmp_wav), y, sr, subtype='PCM_16')
        print(f"[EXTRACT] Running Basic Pitch on {tmp_wav.name} ({duration:.1f}s)...")
        _model_out, _midi, note_events = _bp_predict(
            str(tmp_wav),
            model,
            # A0-C7: the practical piano range. Bounding it keeps sub-bass
            # rumble and cymbal-region artefacts out of the note list.
            minimum_frequency=float(librosa.note_to_hz('A0')),
            maximum_frequency=float(librosa.note_to_hz('C7')),
            minimum_note_length=90.0,
            melodia_trick=True,
        )
    finally:
        try:
            tmp_wav.unlink(missing_ok=True)
        except Exception:
            pass
    print(f"[EXTRACT] Basic Pitch returned {len(note_events)} note events")

    notes = _reduce_polyphony_to_two_voices(note_events)
    notes = merge_close_notes(notes, max_gap_ms=90, max_pitch_diff_st=0.0)
    notes = quantize_notes(notes)

    tempo, key = _estimate_tempo_and_key(y, sr)

    clean = [n for n in notes
             if n.end > n.start and np.isfinite(n.start) and np.isfinite(n.end)
             and 0 <= n.pitch_midi <= 127]
    print(f"[EXTRACT] {len(clean)} clean notes returned (basic-pitch engine)")

    return TranscriptionResult(notes=clean, tempo=tempo, key=key,
                               duration=duration, sr=sr)

def _estimate_tempo_and_key(y: np.ndarray, sr: int):
    tempo = 120.0
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        try:
            # librosa >= 0.10 moved this to feature.rhythm.tempo;
            # librosa.beat.tempo still exists but is deprecated/removed
            # in some versions, so fall back if needed.
            tempo_est = librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr)
        except AttributeError:
            tempo_est = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        if isinstance(tempo_est, np.ndarray):
            tempo_est = tempo_est[0]
        tempo = float(tempo_est) if tempo_est > 40 else 120.0
    except Exception:
        pass

    key = "C major"
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        key = estimate_key(chroma)
    except Exception:
        pass

    return tempo, key

def _estimate_hand_split(y: np.ndarray, sr: int, hop_length: int) -> int:
    """
    v7.4.3: the RH/LH register boundary was hardcoded at middle C (C4).
    That silently assumes the source is a two-handed piano performance
    with melody in the treble and accompaniment in the bass. Real-world
    test audio (a single melody instrument/vocal recording, not a piano
    performance) broke that assumption: its actual tune sits mostly
    *below* C4, so the fixed C3-C6 "RH" pass only ever caught sparse
    fragments of it, while the real line got picked up by the "LH"
    pass and mislabeled as bass accompaniment. Confirmed by analyzing
    a real test file: ~69% of its confidently-voiced pitch content
    fell below C4, and the resulting RH staff was empty rests for most
    of the piece while LH carried what was clearly the tune.

    Fix: run one broadband PYIN pass first to see where THIS track's
    pitched content actually sits, then find a natural boundary between
    its two dominant registers via a simple 1D 2-means split (in MIDI/
    semitone space) instead of assuming C4 always separates them. Falls
    back to a fixed C4 if there isn't enough confidently-voiced content
    to cluster (e.g. very short or very quiet input).
    """
    fallback_midi = 60  # C4
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr,
            hop_length=hop_length,
            frame_length=2048
        )
        mask = voiced_flag & (voiced_probs >= 0.1)
        hz = f0[mask]
        hz = hz[~np.isnan(hz)]
        if len(hz) < 40:
            print(f"[EXTRACT] Split estimation: only {len(hz)} confident frames, falling back to fixed C4 split")
            return fallback_midi

        split_midi = _two_means_split_midi(librosa.hz_to_midi(hz), fallback=fallback_midi)
        print(f"[EXTRACT] Adaptive hand-split at MIDI {split_midi} "
              f"(vs fixed-C4/MIDI60 fallback)")
        return split_midi
    except Exception as e:
        print(f"[EXTRACT WARN] Split estimation failed ({e}), falling back to fixed C4 split")
        return fallback_midi

def extract_melody_librosa_v75(y: np.ndarray, sr: int) -> TranscriptionResult:
    """
    v7.5.0: replaces the "two independent register-restricted PYIN
    passes" architecture from v7.3-v7.4.3.

    That design assumed the source always contains two simultaneous
    voices (RH melody + LH bass) that each stay inside their own
    register the whole time - true for a genuine two-handed piano
    recording. It breaks for a single melodic line (solo instrument,
    voice, simple song) whose real pitch legitimately wanders both
    above and below the RH/LH boundary: at any instant where the true
    note sits outside a given pass's allowed register, that pass has
    nothing real to find - but PYIN is still forced to report its best
    IN-BAND candidate for every frame it considers voiced, so it can
    lock onto a harmonic/subharmonic echo of the real note instead of
    correctly reporting silence. Confirmed on a real test file: with
    both the fixed-C4 split (v7.4.2) and the adaptive split (v7.4.3),
    the RH pass stayed almost entirely empty rests while LH carried
    what was clearly the actual tune, because that track's real melody
    is a single line, not two simultaneous voices.

    Fix: track ONE continuous line across the whole practical range in
    a single PYIN pass, so its own internal Viterbi smoothing always
    sees the true note wherever it is - no register can starve it.
    Hand/staff assignment then happens per completed NOTE afterwards
    (via _estimate_hand_split), not per frame during detection, which
    also avoids the original pre-v7.3 bug where a per-frame threshold
    split chopped single sustained notes into alternating RH/LH
    fragments near the boundary.

    Trade-off worth knowing: because this is one monophonic pitch
    track, two *genuinely* simultaneous notes (e.g. an actual chord,
    or true two-handed piano polyphony) can't both be captured at the
    same instant - only the stronger one will. That was already true
    of the original pre-v7.3 implementation this restores the shape
    of; real polyphonic transcription would need a different class of
    model entirely (see the note left in extract_melody's caller).
    """
    duration = librosa.get_duration(y=y, sr=sr)
    if len(y) < 2048:
        print(f"[EXTRACT WARN] Audio too short: {len(y)} samples")
        return TranscriptionResult(notes=[], tempo=120.0, key="C major", duration=duration, sr=sr)

    hop_length = 512
    prob_thresh = 0.1

    # 1. Single broadband PYIN pass across the practical piano range.
    print("[EXTRACT] Running single broadband PYIN pass (A0-C7)...")
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('A0'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr,
        hop_length=hop_length,
        frame_length=2048
    )
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
    n_voiced_flag_only = int(np.sum(voiced_flag))
    mask = voiced_flag & (voiced_probs >= prob_thresh)
    voice = np.where(mask, f0, np.nan)
    print(f"[EXTRACT] voiced_flag frames={n_voiced_flag_only}, after prob>={prob_thresh} gate={int(np.sum(mask))}")

    # 2. Fill short gaps
    v_filled = fill_short_gaps(voice, times, max_gap_ms=100)

    # 3. Smooth pitch contour (remove vibrato)
    v_smooth = smooth_pitch(v_filled, window=7)

    # 4. Segment into whole notes (hand is a placeholder here - assigned
    # for real in step 6, per note rather than per frame).
    print("[EXTRACT] Segmenting melody line...")
    notes_all = segment_pitch_contour(v_smooth, times, hand="RH", min_dur_ms=120, pause_thresh_ms=250, pitch_jump_st=1.5)
    print(f"[EXTRACT] Raw segments: {len(notes_all)}")

    # 5. Merge close notes (aggressive)
    notes_all = merge_close_notes(notes_all, max_gap_ms=150, max_pitch_diff_st=1.5)
    print(f"[EXTRACT] After merge: {len(notes_all)}")

    # 6. Assign RH/LH per completed note, using the same adaptive split
    # point _estimate_hand_split computes (falls back to fixed C4 if
    # there's too little confidently-voiced content to cluster).
    split_midi = _estimate_hand_split(y, sr, hop_length)
    for n in notes_all:
        n.hand = "RH" if n.pitch_midi >= split_midi else "LH"
    n_rh = sum(1 for n in notes_all if n.hand == "RH")
    n_lh = len(notes_all) - n_rh
    print(f"[EXTRACT] Hand split at MIDI {split_midi}: RH={n_rh} notes, LH={n_lh} notes")

    # 7. Quantize timing
    notes_all = quantize_notes(notes_all)

    # 8. Sort
    all_notes = sorted(notes_all, key=lambda n: n.start)

    # 9. Estimate tempo and key
    tempo = 120.0
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        try:
            # librosa >= 0.10 moved this to feature.rhythm.tempo;
            # librosa.beat.tempo still exists but is deprecated/removed
            # in some versions, so fall back if needed.
            tempo_est = librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr)
        except AttributeError:
            tempo_est = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        if isinstance(tempo_est, np.ndarray):
            tempo_est = tempo_est[0]
        tempo = float(tempo_est) if tempo_est > 40 else 120.0
    except Exception:
        pass

    key = "C major"
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        key = estimate_key(chroma)
    except Exception:
        pass

    # Final validation
    clean_notes = []
    for n in all_notes:
        if n.end > n.start and np.isfinite(n.start) and np.isfinite(n.end) and 0 <= n.pitch_midi <= 127:
            clean_notes.append(n)
    print(f"[EXTRACT] {len(clean_notes)} clean notes returned")

    return TranscriptionResult(
        notes=clean_notes,
        tempo=tempo,
        key=key,
        duration=duration,
        sr=sr
    )

# ───────────────────────────────────────────────────────────────
# Synthesis
# ───────────────────────────────────────────────────────────────
def synthesize_piano_v72(notes: List[Note], sr: int = 22050, duration: float = None) -> np.ndarray:
    if duration is None:
        duration = max((n.end for n in notes), default=1.0) + 1.0

    if not np.isfinite(duration) or duration <= 0:
        duration = 2.0
    duration = max(duration, 0.5)

    total_samples = int(duration * sr)
    if total_samples <= 0:
        return np.zeros((int(0.5 * sr), 2), dtype=np.float32)

    print(f"[SYNTH] duration={duration:.3f}s, samples={total_samples}, notes={len(notes)}")
    audio = np.zeros((total_samples, 2), dtype=np.float64)

    valid_notes = []
    for note in notes:
        if note.pitch_midi < 21 or note.pitch_midi > 108:
            continue
        if note.end <= note.start:
            continue
        if not np.isfinite(note.start) or not np.isfinite(note.end):
            continue
        valid_notes.append(note)

    print(f"[SYNTH] valid_notes={len(valid_notes)}")

    if not valid_notes:
        print("[SYNTH] No valid notes, returning silence")
        return np.zeros((total_samples, 2), dtype=np.float32)

    B = 0.0003
    harmonic_amps = [1.0, 0.5, 0.25, 0.125, 0.06, 0.03, 0.015]

    for note in valid_notes:
        freq = librosa.midi_to_hz(note.pitch_midi)
        start_sample = int(note.start * sr)
        end_sample = min(int(note.end * sr), total_samples)
        note_samples = end_sample - start_sample

        if note_samples <= 0:
            continue

        t = np.arange(note_samples) / sr
        note_audio = np.zeros(note_samples)

        for h, amp in enumerate(harmonic_amps, 1):
            f_h = h * freq * np.sqrt(1 + B * h**2)
            phase = np.cumsum(2 * np.pi * f_h / sr * np.ones(note_samples))
            note_audio += amp * np.sin(phase)

        # ADSR
        attack = min(int(0.01 * sr), note_samples // 4)
        decay = min(int(0.1 * sr), note_samples // 3)
        sustain_level = 0.7
        release = min(int(0.05 * sr), note_samples // 4)

        envelope = np.ones(note_samples) * sustain_level
        if attack > 0:
            envelope[:attack] = np.linspace(0, 1, attack)
        if decay > 0 and attack + decay < note_samples:
            envelope[attack:attack+decay] = np.linspace(1, sustain_level, decay)
        if release > 0 and note_samples - release > 0:
            envelope[-release:] = np.linspace(sustain_level, 0, release)

        note_audio *= envelope

        pan = 0.7 if note.hand == "RH" else 0.3
        left_amp = np.sqrt(1 - pan)
        right_amp = np.sqrt(pan)

        audio[start_sample:end_sample, 0] += note_audio * left_amp
        audio[start_sample:end_sample, 1] += note_audio * right_amp

    max_amp = np.max(np.abs(audio))
    if max_amp > 0:
        audio = audio / max_amp * 0.9

    # Reverb
    reverb = np.zeros_like(audio)
    delay_samples = int(0.05 * sr)
    if delay_samples < len(audio):
        reverb[delay_samples:] = audio[:-delay_samples] * 0.3
    audio = audio + reverb * 0.3

    max_amp = np.max(np.abs(audio))
    if max_amp > 0:
        audio = audio / max_amp * 0.95

    return audio.astype(np.float32)

# ───────────────────────────────────────────────────────────────
# MusicXML
# ───────────────────────────────────────────────────────────────
def _duration_divs_to_type(dur_divs: int) -> str:
    if dur_divs >= 16:
        return 'whole'
    elif dur_divs >= 8:
        return 'half'
    elif dur_divs >= 4:
        return 'quarter'
    elif dur_divs >= 2:
        return 'eighth'
    else:
        return '16th'

def generate_musicxml_v72(result: TranscriptionResult, title: str = "PianoMagic Transcription") -> str:
    """
    v7.4.2: fixed two structural bugs that made every export unreadable
    as real sheet music, independent of how good the underlying note
    detection is:

    1. Every note ended up inside a single <measure number="1">. The old
       loop tracked a `current_measure` counter correctly but only ever
       created a new <measure> element for the very first note (the
       `measure is None` branch) - the "start a new measure" condition
       right after it could never be true because the preceding `while`
       loop had already advanced measure_start past every note that
       would trigger it. Every subsequent note landed inside measure 1.

    2. RH and LH notes were interleaved into ONE forward-only timeline
       with no <backup> between them. MusicXML requires an explicit
       <backup> to rewind the time cursor when moving from one voice/
       staff to another in the same measure; without it, a reader has
       no way to know the two hands play *simultaneously* rather than
       one after another, so the two staves come out as a single
       garbled sequential line instead of a real two-handed score.
    """
    score = Element('score-partwise', version='3.1')

    work = SubElement(score, 'work')
    work_title = SubElement(work, 'work-title')
    work_title.text = title

    ident = SubElement(score, 'identification')
    creator = SubElement(ident, 'creator', type='software')
    creator.text = f'PianoMagic v{VERSION}'

    part_list = SubElement(score, 'part-list')
    score_part = SubElement(part_list, 'score-part', id='P1')
    part_name = SubElement(score_part, 'part-name')
    part_name.text = 'Piano'

    midi_inst = SubElement(score_part, 'midi-instrument', id='P1-I1')
    SubElement(midi_inst, 'midi-channel').text = '1'
    SubElement(midi_inst, 'midi-program').text = '1'

    part = SubElement(score, 'part', id='P1')

    beats_per_measure = 4
    divisions = 4
    divisions_per_measure = beats_per_measure * divisions
    seconds_per_beat = 60.0 / result.tempo
    measure_duration = beats_per_measure * seconds_per_beat

    notes_sorted = sorted(result.notes, key=lambda n: n.start)

    if not notes_sorted:
        measure = SubElement(part, 'measure', number='1')
        attr = SubElement(measure, 'attributes')
        SubElement(attr, 'divisions').text = '4'
        time = SubElement(attr, 'time')
        SubElement(time, 'beats').text = '4'
        SubElement(time, 'beat-type').text = '4'
        key_el = SubElement(attr, 'key')
        SubElement(key_el, 'fifths').text = '0'
        SubElement(attr, 'staves').text = '2'
        clef1 = SubElement(attr, 'clef', number='1')
        SubElement(clef1, 'sign').text = 'G'
        SubElement(clef1, 'line').text = '2'
        clef2 = SubElement(attr, 'clef', number='2')
        SubElement(clef2, 'sign').text = 'F'
        SubElement(clef2, 'line').text = '4'

        note_el = SubElement(measure, 'note')
        SubElement(note_el, 'rest')
        SubElement(note_el, 'duration').text = '16'
        SubElement(note_el, 'type').text = 'whole'
        SubElement(note_el, 'staff').text = '1'
    else:
        rh_notes = [n for n in notes_sorted if n.hand == 'RH']
        lh_notes = [n for n in notes_sorted if n.hand != 'RH']

        last_end = max(n.end for n in notes_sorted)
        total_measures = min(2000, max(1, int(last_end / measure_duration) + 1))

        def emit_voice(measure_el, voice_notes, measure_start, staff_num, voice_num):
            """Write one voice's notes/rests for this measure, padding
            gaps with rests so the voice always sums to exactly
            divisions_per_measure - that's what makes the fixed-size
            <backup> below correct for realigning to the other voice."""
            cursor = 0
            for note in voice_notes:
                start_divs = int(round((note.start - measure_start) / seconds_per_beat * divisions))
                start_divs = max(0, min(start_divs, divisions_per_measure))
                end_divs = int(round((note.end - measure_start) / seconds_per_beat * divisions))
                end_divs = max(start_divs + 1, min(end_divs, divisions_per_measure))

                if start_divs > cursor:
                    gap = start_divs - cursor
                    rest_el = SubElement(measure_el, 'note')
                    SubElement(rest_el, 'rest')
                    SubElement(rest_el, 'duration').text = str(gap)
                    SubElement(rest_el, 'voice').text = str(voice_num)
                    SubElement(rest_el, 'staff').text = str(staff_num)
                    cursor += gap
                elif start_divs < cursor:
                    # overlaps the previous note in this voice (quantization
                    # rounding, or two notes merged very close together) -
                    # never move the cursor backwards within one voice
                    start_divs = cursor

                dur_divs = min(max(1, end_divs - start_divs), divisions_per_measure - cursor)
                if dur_divs <= 0:
                    continue

                note_el = SubElement(measure_el, 'note')
                step, alter, octave = midi_to_ly_step(note.pitch_midi)
                pitch_el = SubElement(note_el, 'pitch')
                SubElement(pitch_el, 'step').text = step
                if alter != 0:
                    SubElement(pitch_el, 'alter').text = str(alter)
                SubElement(pitch_el, 'octave').text = str(octave)
                SubElement(note_el, 'duration').text = str(dur_divs)
                SubElement(note_el, 'type').text = _duration_divs_to_type(dur_divs)
                SubElement(note_el, 'staff').text = str(staff_num)
                SubElement(note_el, 'voice').text = str(voice_num)
                cursor += dur_divs

            if cursor < divisions_per_measure:
                rest_el = SubElement(measure_el, 'note')
                SubElement(rest_el, 'rest')
                SubElement(rest_el, 'duration').text = str(divisions_per_measure - cursor)
                SubElement(rest_el, 'voice').text = str(voice_num)
                SubElement(rest_el, 'staff').text = str(staff_num)

        for m in range(total_measures):
            measure_start = m * measure_duration
            measure_end = measure_start + measure_duration
            measure = SubElement(part, 'measure', number=str(m + 1))

            if m == 0:
                attr = SubElement(measure, 'attributes')
                SubElement(attr, 'divisions').text = str(divisions)

                time = SubElement(attr, 'time')
                SubElement(time, 'beats').text = '4'
                SubElement(time, 'beat-type').text = '4'

                key = SubElement(attr, 'key')
                SubElement(key, 'fifths').text = '0'

                SubElement(attr, 'staves').text = '2'

                clef1 = SubElement(attr, 'clef', number='1')
                SubElement(clef1, 'sign').text = 'G'
                SubElement(clef1, 'line').text = '2'

                clef2 = SubElement(attr, 'clef', number='2')
                SubElement(clef2, 'sign').text = 'F'
                SubElement(clef2, 'line').text = '4'

                direction = SubElement(measure, 'direction', placement='above')
                dt = SubElement(direction, 'direction-type')
                metro = SubElement(dt, 'metronome')
                SubElement(metro, 'beat-unit').text = 'quarter'
                SubElement(metro, 'per-minute').text = str(int(result.tempo))

            rh_in_measure = [n for n in rh_notes if measure_start <= n.start < measure_end]
            lh_in_measure = [n for n in lh_notes if measure_start <= n.start < measure_end]

            emit_voice(measure, rh_in_measure, measure_start, staff_num=1, voice_num=1)

            backup_el = SubElement(measure, 'backup')
            SubElement(backup_el, 'duration').text = str(divisions_per_measure)

            emit_voice(measure, lh_in_measure, measure_start, staff_num=2, voice_num=2)

    rough = tostring(score, encoding='unicode')
    reparsed = minidom.parseString(rough)
    return reparsed.toprettyxml(indent='  ')

# ───────────────────────────────────────────────────────────────
# Comparison (with chroma data for frontend display)
# ───────────────────────────────────────────────────────────────
def compare_audio_features(original_y: np.ndarray, synth_y: np.ndarray, sr: int) -> dict:
    min_len = min(len(original_y), len(synth_y))
    if min_len < 512:
        return {
            'chroma_correlation': 0.0,
            'spectral_contrast_correlation': 0.0,
            'onset_correlation': 0.0,
            'overall_similarity': 0.0,
            'chroma_orig': [],
            'chroma_synth': []
        }
    orig = original_y[:min_len]
    synth = synth_y[:min_len]

    try:
        chroma_orig = librosa.feature.chroma_stft(y=orig, sr=sr)
        chroma_synth = librosa.feature.chroma_stft(y=synth, sr=sr)
        chroma_corr = np.corrcoef(chroma_orig.mean(axis=1), chroma_synth.mean(axis=1))[0, 1]

        sc_orig = librosa.feature.spectral_contrast(y=orig, sr=sr)
        sc_synth = librosa.feature.spectral_contrast(y=synth, sr=sr)
        sc_corr = np.corrcoef(sc_orig.mean(axis=1), sc_synth.mean(axis=1))[0, 1]

        onset_orig = librosa.onset.onset_strength(y=orig, sr=sr)
        onset_synth = librosa.onset.onset_strength(y=synth, sr=sr)
        onset_corr = np.corrcoef(onset_orig, onset_synth)[0, 1]

        overall = np.mean([chroma_corr, sc_corr, onset_corr])

        return {
            'chroma_correlation': round(float(chroma_corr), 3),
            'spectral_contrast_correlation': round(float(sc_corr), 3),
            'onset_correlation': round(float(onset_corr), 3),
            'overall_similarity': round(float(overall), 3),
            'chroma_orig': chroma_orig.mean(axis=1).tolist(),
            'chroma_synth': chroma_synth.mean(axis=1).tolist()
        }
    except Exception:
        return {
            'chroma_correlation': 0.0,
            'spectral_contrast_correlation': 0.0,
            'onset_correlation': 0.0,
            'overall_similarity': 0.0,
            'chroma_orig': [],
            'chroma_synth': []
        }

# ───────────────────────────────────────────────────────────────
# Background Task
# ───────────────────────────────────────────────────────────────
async def process_audio(task_id: str, file_path: Path):
    log_buf: List[str] = tasks[task_id].setdefault('log', [])
    log_buf.append("=" * 62)
    log_buf.append(f"PianoMagic run log - task {task_id}")
    log_buf.append(f"started (UTC)           : {datetime.utcnow().isoformat()}")
    log_buf.append(f"source file             : {tasks[task_id].get('filename')}")
    log_buf.extend(_env_report())
    log_buf.append("=" * 62)

    import contextlib, sys as _sys
    tee = _TeeLog(log_buf, _sys.stdout)
    with contextlib.redirect_stdout(tee):
        await _process_audio_inner(task_id, file_path)
    tee.flush()
    if tee._partial:
        log_buf.append(tee._partial)
    log_buf.append(f"finished (UTC)          : {datetime.utcnow().isoformat()}")

async def _process_audio_inner(task_id: str, file_path: Path):
    try:
        tasks[task_id]['status'] = 'loading'
        tasks[task_id]['progress'] = 10
        print(f"[TASK {task_id}] Loading audio from {file_path}")

        y, sr = librosa.load(str(file_path), sr=22050, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        print(f"[TASK {task_id}] Audio loaded: {duration:.2f}s, {len(y)} samples")

        tasks[task_id]['status'] = 'analyzing'
        tasks[task_id]['progress'] = 30

        # v7.6.0: prefer the neural polyphonic engine; fall back to the
        # monophonic PYIN pipeline if it's unavailable or errors out, so
        # a bad wheel degrades quality instead of failing the request.
        global LAST_ENGINE_ERROR
        result = None
        engine_used = "librosa-pyin"
        engine_error = None
        if BASIC_PITCH_AVAILABLE:
            try:
                result = extract_melody_basic_pitch(file_path, y, sr)
                engine_used = "basic-pitch"
                LAST_ENGINE_ERROR = None
            except Exception as e:
                engine_error = traceback.format_exc()
                LAST_ENGINE_ERROR = engine_error
                print(f"[TASK {task_id}] basic-pitch FAILED at runtime "
                      f"({e!r}) - falling back to PYIN. Traceback:")
                print(engine_error)
        if result is None:
            result = extract_melody_librosa_v75(y, sr)
        print(f"[TASK {task_id}] Extracted {len(result.notes)} notes "
              f"via engine={engine_used}")

        tasks[task_id]['status'] = 'synthesizing'
        tasks[task_id]['progress'] = 60

        synth_dur = result.duration if np.isfinite(result.duration) and result.duration > 0 else duration
        print(f"[TASK {task_id}] Synthesizing, dur={synth_dur:.3f}s")

        synth_audio = synthesize_piano_v72(result.notes, sr=sr, duration=synth_dur)
        print(f"[TASK {task_id}] Synthesis done, shape={synth_audio.shape}, max={np.max(np.abs(synth_audio)):.4f}")

        tasks[task_id]['status'] = 'generating_score'
        tasks[task_id]['progress'] = 80

        musicxml = generate_musicxml_v72(result)
        print(f"[TASK {task_id}] MusicXML generated, len={len(musicxml)}")

        # Save files
        file_id = task_id
        wav_path = UPLOAD_DIR / f"PianoMagic_{file_id}.wav"
        xml_path = UPLOAD_DIR / f"PianoMagic_{file_id}.xml"

        print(f"[TASK {task_id}] Saving WAV to {wav_path}")
        sf.write(str(wav_path), synth_audio, sr)
        wav_size = wav_path.stat().st_size if wav_path.exists() else 0
        print(f"[TASK {task_id}] WAV saved, size={wav_size} bytes")

        print(f"[TASK {task_id}] Saving XML to {xml_path}")
        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write(musicxml)
        xml_size = xml_path.stat().st_size if xml_path.exists() else 0
        print(f"[TASK {task_id}] XML saved, size={xml_size} bytes")

        # Compare
        try:
            comparison = compare_audio_features(y, synth_audio[:, 0] if synth_audio.ndim > 1 else synth_audio, sr)
        except Exception as e:
            print(f"[TASK {task_id}] Compare error: {e}")
            comparison = {'chroma_correlation': 0.0, 'spectral_contrast_correlation': 0.0, 'onset_correlation': 0.0, 'overall_similarity': 0.0, 'chroma_orig': [], 'chroma_synth': []}

        # Serialize notes
        notes_json = []
        for n in result.notes:
            notes_json.append({
                'start': round(n.start, 3),
                'end': round(n.end, 3),
                'pitch_midi': n.pitch_midi,
                'velocity': n.velocity,
                'hand': n.hand
            })

        tasks[task_id]['status'] = 'completed'
        tasks[task_id]['progress'] = 100
        tasks[task_id]['result'] = {
            'file_id': file_id,
            'duration': round(duration, 2),
            'tempo': round(result.tempo, 1),
            'key': result.key,
            'notes_count': len(result.notes),
            'rh_notes': len([n for n in result.notes if n.hand == 'RH']),
            'lh_notes': len([n for n in result.notes if n.hand == 'LH']),
            'notes': notes_json,
            'comparison': comparison,
            'wav_url': f'/download/{file_id}.wav',
            'xml_url': f'/download/{file_id}.xml',
            # Which engine actually produced these notes. Reported per
            # result, not just per process, because the neural engine can
            # import cleanly and still fail per request - in which case
            # this says "librosa-pyin" and engine_error holds the reason.
            'engine': engine_used,
            'engine_error': engine_error,
        }
        print(f"[TASK {task_id}] COMPLETED")

    except Exception as e:
        tasks[task_id]['status'] = 'error'
        tasks[task_id]['error'] = str(e)
        tb = traceback.format_exc()
        tasks[task_id]['traceback'] = tb
        print(f"[TASK {task_id}] ERROR: {e}")
        print(tb)

# ───────────────────────────────────────────────────────────────
# API Endpoints
# ───────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"service": "PianoMagic API", "version": VERSION, "status": "running"}

@app.get("/version")
async def get_version():
    return {
        "version": VERSION,
        "backend": VERSION,
        "api": "v1",
        # Which engine this deploy IMPORTED. Note this is not proof it
        # works: v7.6.0 reported "basic-pitch" here while every actual
        # transcription silently fell back to PYIN because inference threw
        # per request. last_engine_error below is the honest signal - if
        # it's non-null, the neural engine is failing at runtime and
        # results are coming from the monophonic fallback.
        "engine": "basic-pitch" if BASIC_PITCH_AVAILABLE else "librosa-pyin",
        "basic_pitch_error": _BASIC_PITCH_IMPORT_ERROR,
        "last_engine_error": LAST_ENGINE_ERROR,
        "env": _env_report(),
    }

@app.get("/logs/{task_id}")
async def get_logs(task_id: str):
    """Plain-text run log for one transcription, for copy/paste."""
    if task_id not in tasks:
        raise HTTPException(404, "Task not found")
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse("\n".join(tasks[task_id].get('log', [])))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": VERSION}

@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    task_id = str(uuid.uuid4())

    ext = Path(file.filename).suffix.lower()
    if ext not in ['.mp3', '.wav', '.flac', '.ogg', '.m4a']:
        raise HTTPException(400, "Unsupported file format.")

    upload_path = UPLOAD_DIR / f"{task_id}{ext}"
    with open(upload_path, 'wb') as f:
        content = await file.read()
        f.write(content)

    print(f"[UPLOAD] task={task_id}, file={file.filename}, size={len(content)} bytes")

    tasks[task_id] = {
        'id': task_id,
        'status': 'queued',
        'progress': 0,
        'filename': file.filename,
        'created_at': datetime.utcnow().isoformat()
    }

    asyncio.create_task(process_audio(task_id, upload_path))

    return {"task_id": task_id, "status": "queued"}

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(404, "Task not found")
    return tasks[task_id]

@app.get("/download/{file_id}")
async def download_file(file_id: str):
    # The frontend requests /download/{file_id}.wav or .xml, so file_id
    # arrives here ALREADY carrying an extension (FastAPI path params
    # capture dots, they're not segment separators). The old code re-
    # appended .wav/.xml on top of that, producing paths like
    # "PianoMagic_<uuid>.wav.wav" that never existed on disk -> every
    # download 404'd with "File not found" even though the file was
    # saved correctly as "PianoMagic_<uuid>.wav". Strip any extension
    # first and look up the exact file that was actually written.
    stem = Path(file_id).stem
    requested_ext = Path(file_id).suffix.lower()
    candidates = [requested_ext] if requested_ext in ('.wav', '.xml') else ['.wav', '.xml']

    for ext in candidates:
        file_path = UPLOAD_DIR / f"PianoMagic_{stem}{ext}"
        if file_path.exists():
            media_type = 'audio/wav' if ext == '.wav' else 'application/vnd.recordare.musicxml+xml'
            return FileResponse(str(file_path), media_type=media_type,
                              filename=f"PianoMagic_{stem}{ext}")

    raise HTTPException(404, "File not found")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
