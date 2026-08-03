"""
PianoMagic Backend API — v7.2
Audio-to-piano-score transcription with continuous pitch contour segmentation
and two-voice separation.
"""

import os
import io
import uuid
import asyncio
import tempfile
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import librosa
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# MusicXML generation
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom

# Audio processing
import soundfile as sf
from scipy import signal

# ───────────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────────
VERSION = "7.2.0"
UPLOAD_DIR = Path(tempfile.gettempdir()) / "pianomagic_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Task storage (in-memory; for production use Redis)
tasks: Dict[str, dict] = {}

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
    start: float      # seconds
    end: float        # seconds
    pitch_midi: int
    velocity: int = 80
    hand: str = "RH"  # "RH" or "LH"

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
def midi_to_note_name(midi: int) -> str:
    names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi // 12) - 1
    return f"{names[midi % 12]}{octave}"

def midi_to_ly_step(midi: int) -> tuple:
    """Return step, alter, octave for MusicXML."""
    names = ['C', 'C', 'D', 'D', 'E', 'F', 'F', 'G', 'G', 'A', 'A', 'B']
    alters = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
    octave = (midi // 12) - 1
    idx = midi % 12
    return names[idx], alters[idx], octave

def estimate_key(chroma: np.ndarray) -> str:
    """Krumhansl-Schmuckler key estimation."""
    profiles = {
        'C major': [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        'C minor': [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    }
    # Rotate to all keys
    all_profiles = {}
    for i in range(12):
        for mode, prof in profiles.items():
            key_name = librosa.midi_to_note(i + 60)[:-1]  # Remove octave
            if mode == 'C major':
                all_profiles[f"{key_name} major"] = np.roll(prof, i)
            else:
                all_profiles[f"{key_name} minor"] = np.roll(prof, i)

    chroma_mean = chroma.mean(axis=1)
    correlations = {k: np.corrcoef(chroma_mean, v)[0, 1] for k, v in all_profiles.items()}
    return max(correlations, key=correlations.get)

# ───────────────────────────────────────────────────────────────
# Core Algorithm: v7.2 — Continuous Pitch Contour Segmentation
# ───────────────────────────────────────────────────────────────
def fill_short_gaps(voice: np.ndarray, times: np.ndarray, max_gap_ms: float = 80) -> np.ndarray:
    """Interpolate short gaps in a pitch contour."""
    result = voice.copy()
    valid = ~np.isnan(voice)
    if np.sum(valid) < 2:
        return result

    gaps = np.diff(valid.astype(int), prepend=0, append=0)
    gap_starts = np.where(gaps == -1)[0]
    gap_ends = np.where(gaps == 1)[0]

    sr = 22050  # assumed
    hop_length = 256
    for gs, ge in zip(gap_starts, gap_ends):
        gap_dur_ms = (ge - gs) * hop_length / sr * 1000
        if gap_dur_ms <= max_gap_ms and gs > 0 and ge < len(voice):
            result[gs:ge] = np.linspace(voice[gs-1], voice[ge], ge - gs)
    return result

def segment_pitch_contour(
    filled_voice: np.ndarray,
    times: np.ndarray,
    hand: str,
    min_dur_ms: float = 60,
    pause_thresh_ms: float = 100,
    pitch_jump_st: float = 0.5
) -> List[Note]:
    """
    Segment a continuous pitch contour into discrete notes.

    Rules:
    - New note when pitch jumps > 0.5 semitones
    - New note when gap > pause_thresh_ms
    - Filter notes shorter than min_dur_ms
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
                    if dur_ms >= min_dur_ms:
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
                    if dur_ms >= min_dur_ms:
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
        if dur_ms >= min_dur_ms:
            notes.append(Note(
                start=note_start,
                end=note_end,
                pitch_midi=int(round(librosa.hz_to_midi(current_pitch))),
                hand=hand
            ))

    return notes

def merge_close_notes(notes: List[Note], max_gap_ms: float = 80, max_pitch_diff_st: float = 1.0) -> List[Note]:
    """Merge consecutive notes of same pitch with small gaps."""
    if not notes:
        return notes

    notes_sorted = sorted(notes, key=lambda n: n.start)
    merged = [notes_sorted[0]]

    for note in notes_sorted[1:]:
        last = merged[-1]
        gap_ms = (note.start - last.end) * 1000
        pitch_diff = abs(note.pitch_midi - last.pitch_midi)

        if gap_ms < max_gap_ms and pitch_diff <= max_pitch_diff_st and last.hand == note.hand:
            # Merge
            last.end = note.end
            last.pitch_midi = int(round((last.pitch_midi + note.pitch_midi) / 2))
        else:
            merged.append(note)

    return merged

def prune_salient_notes(notes: List[Note], min_dur_ms: float = 60) -> List[Note]:
    """Remove very short notes that are likely artifacts."""
    return [n for n in notes if (n.end - n.start) * 1000 >= min_dur_ms]

def extract_melody_librosa_v72(y: np.ndarray, sr: int) -> TranscriptionResult:
    """
    v7.2: Continuous pitch contour segmentation with two-voice separation.

    Key changes from v7.1:
    - No onset-based detection; uses continuous f0 tracking
    - Two voices separated by 300Hz threshold
    - No octave correction (preserves real melodic leaps)
    - min_dur reduced to 60ms
    - No wait parameter
    """
    duration = librosa.get_duration(y=y, sr=sr)

    # 1. PYIN with fine hop resolution
    hop_length = 256
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr,
        hop_length=hop_length,
        frame_length=2048
    )
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)

    # 2. Voice separation by frequency threshold
    threshold_hz = 300
    voice_high = np.where((f0 > threshold_hz) & voiced_flag, f0, np.nan)
    voice_low = np.where((f0 <= threshold_hz) & voiced_flag, f0, np.nan)

    # 3. Fill short gaps in each voice independently
    vh_filled = fill_short_gaps(voice_high, times, max_gap_ms=80)
    vl_filled = fill_short_gaps(voice_low, times, max_gap_ms=80)

    # 4. Segment each voice into notes
    notes_high = segment_pitch_contour(vh_filled, times, hand="RH", min_dur_ms=60, pause_thresh_ms=100)
    notes_low = segment_pitch_contour(vl_filled, times, hand="LH", min_dur_ms=60, pause_thresh_ms=100)

    # 5. Merge close notes within each hand
    notes_high = merge_close_notes(notes_high, max_gap_ms=80, max_pitch_diff_st=1.0)
    notes_low = merge_close_notes(notes_low, max_gap_ms=80, max_pitch_diff_st=1.0)

    # 6. Prune very short artifacts
    notes_high = prune_salient_notes(notes_high, min_dur_ms=60)
    notes_low = prune_salient_notes(notes_low, min_dur_ms=60)

    # 7. Combine and sort
    all_notes = sorted(notes_high + notes_low, key=lambda n: n.start)

    # 8. Estimate tempo and key
    tempo = 120.0
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
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

    return TranscriptionResult(
        notes=all_notes,
        tempo=tempo,
        key=key,
        duration=duration,
        sr=sr
    )

# ───────────────────────────────────────────────────────────────
# Synthesis
# ───────────────────────────────────────────────────────────────
def synthesize_piano_v72(notes: List[Note], sr: int = 22050, duration: float = None) -> np.ndarray:
    """
    Inharmonic piano synthesis with ADSR envelope and stereo panning.
    v7.2: Updated for two-voice output.
    """
    if duration is None:
        duration = max((n.end for n in notes), default=1.0) + 1.0

    total_samples = int(duration * sr)
    audio = np.zeros((total_samples, 2), dtype=np.float64)

    # Inharmonicity coefficient
    B = 0.0003

    for note in notes:
        if note.pitch_midi < 21 or note.pitch_midi > 108:
            continue

        freq = librosa.midi_to_hz(note.pitch_midi)
        start_sample = int(note.start * sr)
        end_sample = int(note.end * sr)
        note_samples = end_sample - start_sample

        if note_samples <= 0:
            continue

        t = np.arange(note_samples) / sr

        # Inharmonic partials
        harmonic_amps = [1.0, 0.5, 0.25, 0.125, 0.06, 0.03, 0.015]
        note_audio = np.zeros(note_samples)

        for h, amp in enumerate(harmonic_amps, 1):
            # Inharmonic frequency: f_n = n * f0 * sqrt(1 + B * n^2)
            f_h = h * freq * np.sqrt(1 + B * h**2)
            phase = np.cumsum(2 * np.pi * f_h / sr * np.ones(note_samples))
            note_audio += amp * np.sin(phase)

        # ADSR envelope
        attack = int(0.01 * sr)
        decay = int(0.1 * sr)
        sustain_level = 0.7
        release = int(0.05 * sr)

        envelope = np.ones(note_samples) * sustain_level
        if attack > 0 and attack < note_samples:
            envelope[:attack] = np.linspace(0, 1, attack)
        if decay > 0 and attack + decay < note_samples:
            envelope[attack:attack+decay] = np.linspace(1, sustain_level, decay)
        if release > 0 and note_samples - release > 0:
            envelope[-release:] = np.linspace(sustain_level, 0, release)

        note_audio *= envelope

        # Stereo panning: RH right, LH left
        pan = 0.7 if note.hand == "RH" else 0.3
        left_amp = np.sqrt(1 - pan)
        right_amp = np.sqrt(pan)

        if start_sample + note_samples <= total_samples:
            audio[start_sample:start_sample+note_samples, 0] += note_audio * left_amp
            audio[start_sample:start_sample+note_samples, 1] += note_audio * right_amp

    # Normalize
    max_amp = np.max(np.abs(audio))
    if max_amp > 0:
        audio = audio / max_amp * 0.9

    # Simple reverb (comb filter)
    reverb = np.zeros_like(audio)
    delay_samples = int(0.05 * sr)
    decay = 0.3
    if delay_samples < len(audio):
        reverb[delay_samples:] = audio[:-delay_samples] * decay
    audio = audio + reverb * 0.3

    # Final normalize
    max_amp = np.max(np.abs(audio))
    if max_amp > 0:
        audio = audio / max_amp * 0.95

    return audio.astype(np.float32)

# ───────────────────────────────────────────────────────────────
# MusicXML Generation
# ───────────────────────────────────────────────────────────────
def generate_musicxml_v72(result: TranscriptionResult, title: str = "PianoMagic Transcription") -> str:
    """Generate MusicXML string from transcription result."""

    # Root
    score = Element('score-partwise', version='3.1')

    # Work
    work = SubElement(score, 'work')
    work_title = SubElement(work, 'work-title')
    work_title.text = title

    # Identification
    ident = SubElement(score, 'identification')
    creator = SubElement(ident, 'creator', type='software')
    creator.text = f'PianoMagic v{VERSION}'

    # Part list
    part_list = SubElement(score, 'part-list')
    score_part = SubElement(part_list, 'score-part', id='P1')
    part_name = SubElement(score_part, 'part-name')
    part_name.text = 'Piano'

    # MIDI instrument
    midi_inst = SubElement(score_part, 'midi-instrument', id='P1-I1')
    midi_ch = SubElement(midi_inst, 'midi-channel')
    midi_ch.text = '1'
    midi_prog = SubElement(midi_inst, 'midi-program')
    midi_prog.text = '1'

    # Part
    part = SubElement(score, 'part', id='P1')

    # Group notes by measure (4/4, tempo-based)
    beats_per_measure = 4
    seconds_per_beat = 60.0 / result.tempo
    measure_duration = beats_per_measure * seconds_per_beat

    # Sort notes
    notes_sorted = sorted(result.notes, key=lambda n: n.start)

    if not notes_sorted:
        # Empty measure
        measure = SubElement(part, 'measure', number='1')
        attr = SubElement(measure, 'attributes')
        div = SubElement(attr, 'divisions')
        div.text = '4'
        time = SubElement(attr, 'time')
        beats = SubElement(time, 'beats')
        beats.text = '4'
        beat_type = SubElement(time, 'beat-type')
        beat_type.text = '4'
        clef = SubElement(attr, 'clef', number='1')
        sign = SubElement(clef, 'sign')
        sign.text = 'G'
        line = SubElement(clef, 'line')
        line.text = '2'

        note_el = SubElement(measure, 'note')
        rest = SubElement(note_el, 'rest')
        dur = SubElement(note_el, 'duration')
        dur.text = '16'
        type_el = SubElement(note_el, 'type')
        type_el.text = 'whole'
    else:
        current_measure = 1
        measure_start = 0.0
        measure = None
        attr_set = False

        divisions = 4  # quarter = 4 divisions

        for note in notes_sorted:
            # Check if new measure needed
            while note.start >= measure_start + measure_duration:
                current_measure += 1
                measure_start += measure_duration
                attr_set = False

            if measure is None or note.start >= measure_start + measure_duration:
                measure = SubElement(part, 'measure', number=str(current_measure))

                if not attr_set:
                    attr = SubElement(measure, 'attributes')
                    div = SubElement(attr, 'divisions')
                    div.text = str(divisions)

                    if current_measure == 1:
                        time = SubElement(attr, 'time')
                        beats = SubElement(time, 'beats')
                        beats.text = '4'
                        beat_type = SubElement(time, 'beat-type')
                        beat_type.text = '4'

                        key = SubElement(attr, 'key')
                        fifths = SubElement(key, 'fifths')
                        # Simple key signature mapping
                        key_map = {
                            'C major': 0, 'G major': 1, 'D major': 2, 'A major': 3,
                            'E major': 4, 'B major': 5, 'F# major': 6,
                            'F major': -1, 'Bb major': -2, 'Eb major': -3,
                            'A minor': 0, 'E minor': 1, 'D minor': -1,
                            'G minor': -2, 'C minor': -3
                        }
                        fifths.text = str(key_map.get(result.key, 0))

                        # Two staves
                        staves = SubElement(attr, 'staves')
                        staves.text = '2'

                        # Treble clef
                        clef1 = SubElement(attr, 'clef', number='1')
                        sign1 = SubElement(clef1, 'sign')
                        sign1.text = 'G'
                        line1 = SubElement(clef1, 'line')
                        line1.text = '2'

                        # Bass clef
                        clef2 = SubElement(attr, 'clef', number='2')
                        sign2 = SubElement(clef2, 'sign')
                        sign2.text = 'F'
                        line2 = SubElement(clef2, 'line')
                        line2.text = '4'

                        # Tempo
                        direction = SubElement(measure, 'direction', placement='above')
                        direction_type = SubElement(direction, 'direction-type')
                        metronome = SubElement(direction_type, 'metronome')
                        beat_unit = SubElement(metronome, 'beat-unit')
                        beat_unit.text = 'quarter'
                        per_min = SubElement(metronome, 'per-minute')
                        per_min.text = str(int(result.tempo))

                    attr_set = True

            # Calculate duration in divisions
            note_dur = note.end - note.start
            dur_divs = max(1, int(round(note_dur / seconds_per_beat * divisions)))

            note_el = SubElement(measure, 'note')

            # Staff: RH=1, LH=2
            staff_num = '1' if note.hand == 'RH' else '2'

            step, alter, octave = midi_to_ly_step(note.pitch_midi)
            pitch_el = SubElement(note_el, 'pitch')
            step_el = SubElement(pitch_el, 'step')
            step_el.text = step
            if alter != 0:
                alter_el = SubElement(pitch_el, 'alter')
                alter_el.text = str(alter)
            octave_el = SubElement(pitch_el, 'octave')
            octave_el.text = str(octave)

            dur_el = SubElement(note_el, 'duration')
            dur_el.text = str(dur_divs)

            # Note type approximation
            type_name = 'quarter'
            if dur_divs >= 16:
                type_name = 'whole'
            elif dur_divs >= 8:
                type_name = 'half'
            elif dur_divs >= 4:
                type_name = 'quarter'
            elif dur_divs >= 2:
                type_name = 'eighth'
            elif dur_divs >= 1:
                type_name = '16th'

            type_el = SubElement(note_el, 'type')
            type_el.text = type_name

            staff_el = SubElement(note_el, 'staff')
            staff_el.text = staff_num

            # Voice
            voice_el = SubElement(note_el, 'voice')
            voice_el.text = staff_num

    # Pretty print
    rough = tostring(score, encoding='unicode')
    reparsed = minidom.parseString(rough)
    return reparsed.toprettyxml(indent='  ')

# ───────────────────────────────────────────────────────────────
# Comparison (kept for backend, removed from frontend display)
# ───────────────────────────────────────────────────────────────
def compare_audio_features(original_y: np.ndarray, synth_y: np.ndarray, sr: int) -> dict:
    """Compute similarity metrics between original and synthesized audio."""
    # Ensure same length
    min_len = min(len(original_y), len(synth_y))
    orig = original_y[:min_len]
    synth = synth_y[:min_len]

    # Chroma
    chroma_orig = librosa.feature.chroma_stft(y=orig, sr=sr)
    chroma_synth = librosa.feature.chroma_stft(y=synth, sr=sr)
    chroma_corr = np.corrcoef(chroma_orig.mean(axis=1), chroma_synth.mean(axis=1))[0, 1]

    # Spectral contrast
    sc_orig = librosa.feature.spectral_contrast(y=orig, sr=sr)
    sc_synth = librosa.feature.spectral_contrast(y=synth, sr=sr)
    sc_corr = np.corrcoef(sc_orig.mean(axis=1), sc_synth.mean(axis=1))[0, 1]

    # Onset
    onset_orig = librosa.onset.onset_strength(y=orig, sr=sr)
    onset_synth = librosa.onset.onset_strength(y=synth, sr=sr)
    onset_corr = np.corrcoef(onset_orig, onset_synth)[0, 1]

    # Overall
    overall = np.mean([chroma_corr, sc_corr, onset_corr])

    return {
        'chroma_correlation': round(float(chroma_corr), 3),
        'spectral_contrast_correlation': round(float(sc_corr), 3),
        'onset_correlation': round(float(onset_corr), 3),
        'overall_similarity': round(float(overall), 3)
    }

# ───────────────────────────────────────────────────────────────
# Background Task: Transcription
# ───────────────────────────────────────────────────────────────
async def process_audio(task_id: str, file_path: Path):
    """Main transcription pipeline."""
    try:
        tasks[task_id]['status'] = 'loading'
        tasks[task_id]['progress'] = 10

        # Load audio
        y, sr = librosa.load(str(file_path), sr=22050, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)

        tasks[task_id]['status'] = 'analyzing'
        tasks[task_id]['progress'] = 30

        # Transcribe
        result = extract_melody_librosa_v72(y, sr)

        tasks[task_id]['status'] = 'synthesizing'
        tasks[task_id]['progress'] = 60

        # Synthesize
        synth_audio = synthesize_piano_v72(result.notes, sr=sr, duration=duration)

        tasks[task_id]['status'] = 'generating_score'
        tasks[task_id]['progress'] = 80

        # Generate MusicXML
        musicxml = generate_musicxml_v72(result)

        # Save files
        file_id = task_id
        wav_path = UPLOAD_DIR / f"PianoMagic_{file_id}.wav"
        xml_path = UPLOAD_DIR / f"PianoMagic_{file_id}.xml"

        sf.write(str(wav_path), synth_audio, sr)
        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write(musicxml)

        # Compare features
        comparison = compare_audio_features(y, synth_audio[:, 0], sr)

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
            'comparison': comparison,
            'wav_url': f'/download/{file_id}.wav',
            'xml_url': f'/download/{file_id}.xml'
        }

    except Exception as e:
        tasks[task_id]['status'] = 'error'
        tasks[task_id]['error'] = str(e)
        import traceback
        tasks[task_id]['traceback'] = traceback.format_exc()

# ───────────────────────────────────────────────────────────────
# API Endpoints
# ───────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "service": "PianoMagic API",
        "version": VERSION,
        "status": "running"
    }

@app.get("/version")
async def get_version():
    return {"version": VERSION, "backend": VERSION, "api": "v1"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": VERSION}

@app.post("/upload")
async def upload_audio(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload audio file and start transcription."""
    task_id = str(uuid.uuid4())

    # Save uploaded file
    ext = Path(file.filename).suffix.lower()
    if ext not in ['.mp3', '.wav', '.flac', '.ogg', '.m4a']:
        raise HTTPException(400, "Unsupported file format. Use mp3, wav, flac, ogg, or m4a.")

    upload_path = UPLOAD_DIR / f"{task_id}{ext}"
    with open(upload_path, 'wb') as f:
        content = await file.read()
        f.write(content)

    tasks[task_id] = {
        'id': task_id,
        'status': 'queued',
        'progress': 0,
        'filename': file.filename,
        'created_at': datetime.utcnow().isoformat()
    }

    # Start processing
    asyncio.create_task(process_audio(task_id, upload_path))

    return {"task_id": task_id, "status": "queued"}

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    """Get transcription status and results."""
    if task_id not in tasks:
        raise HTTPException(404, "Task not found")
    return tasks[task_id]

@app.get("/download/{file_id}")
async def download_file(file_id: str):
    """Download generated file (wav or xml)."""
    # Determine extension
    for ext in ['.wav', '.xml']:
        file_path = UPLOAD_DIR / f"PianoMagic_{file_id}{ext}"
        if file_path.exists():
            media_type = 'audio/wav' if ext == '.wav' else 'application/vnd.recordare.musicxml+xml'
            return FileResponse(str(file_path), media_type=media_type, 
                              filename=f"PianoMagic_{file_id}{ext}")

    raise HTTPException(404, "File not found")

# ───────────────────────────────────────────────────────────────
# Startup
# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
