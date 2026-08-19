"""
PianoMagic Backend API — v7.3.0
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

# ───────────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────────
VERSION = "7.3.0"
UPLOAD_DIR = Path(tempfile.gettempdir()) / "pianomagic_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
print(f"[INIT] UPLOAD_DIR={UPLOAD_DIR}, exists={UPLOAD_DIR.exists()}")

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
# Core Algorithm: v7.3.0
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

def _track_voice_pyin(y: np.ndarray, sr: int, hop_length: int, fmin_note: str, fmax_note: str,
                       prob_thresh: float = 0.5):
    """
    Run PYIN independently within ONE register (e.g. just the melody range,
    or just the bass range) and return an f0 array masked down to frames
    that are both voiced AND above a confidence floor.

    This replaces the old approach of running PYIN ONCE over the full
    C2-C7 range and then splitting the single resulting (already jittery,
    monophonic) f0 trace into "high"/"low" by comparing each frame's raw
    Hz value to a global amplitude threshold. That post-hoc split had two
    problems: (a) it inherited whatever octave errors/noise PYIN produced
    over the full wide range, and (b) because it decided high-vs-low
    frame-by-frame with no continuity constraint, a single sustained note
    that happened to sit near the threshold got chopped into many tiny
    alternating RH/LH fragments.

    Running PYIN separately per register instead means each pass gets
    librosa's own internal probabilistic Viterbi smoothing *within* the
    correct register, so both voices come out as continuous, stable
    contours instead of a shared, fragmented one.
    """
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz(fmin_note),
        fmax=librosa.note_to_hz(fmax_note),
        sr=sr,
        hop_length=hop_length,
        frame_length=2048
    )
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
    # Confidence gate: voiced_flag alone just means "PYIN's internal voicing
    # probability crossed ~0.5"; frames near that boundary are exactly the
    # jittery, low-confidence ones that fragment notes. Requiring
    # voiced_probs >= prob_thresh on top of voiced_flag drops those.
    mask = voiced_flag & (voiced_probs >= prob_thresh)
    voice = np.where(mask, f0, np.nan)
    return voice, times, int(np.sum(mask))

def extract_melody_librosa_v73(y: np.ndarray, sr: int) -> TranscriptionResult:
    duration = librosa.get_duration(y=y, sr=sr)
    if len(y) < 2048:
        print(f"[EXTRACT WARN] Audio too short: {len(y)} samples")
        return TranscriptionResult(notes=[], tempo=120.0, key="C major", duration=duration, sr=sr)

    hop_length = 512

    # 1-2. Two independent, register-restricted PYIN passes (RH melody
    # register / LH bass register) instead of one wideband pass split by
    # a raw Hz threshold. A small overlap around middle C (C3-C6 vs
    # A0-C4) is intentional: real melodies dip below middle C and real
    # bass/accompaniment lines can rise above it.
    print("[EXTRACT] Running PYIN for RH (melody) register C3-C6...")
    voice_high, times, n_high = _track_voice_pyin(y, sr, hop_length, 'C3', 'C6')
    print(f"[EXTRACT] RH voiced+confident frames: {n_high}")

    print("[EXTRACT] Running PYIN for LH (bass) register A0-C4...")
    voice_low, _, n_low = _track_voice_pyin(y, sr, hop_length, 'A0', 'C4')
    print(f"[EXTRACT] LH voiced+confident frames: {n_low}")

    # 3. Fill short gaps
    vh_filled = fill_short_gaps(voice_high, times, max_gap_ms=100)
    vl_filled = fill_short_gaps(voice_low, times, max_gap_ms=100)

    # 4. Smooth pitch contours (remove vibrato)
    vh_smooth = smooth_pitch(vh_filled, window=7)
    vl_smooth = smooth_pitch(vl_filled, window=7)

    # 5. Segment with relaxed thresholds
    print("[EXTRACT] Segmenting voices...")
    notes_high = segment_pitch_contour(vh_smooth, times, hand="RH", min_dur_ms=120, pause_thresh_ms=250, pitch_jump_st=1.5)
    notes_low = segment_pitch_contour(vl_smooth, times, hand="LH", min_dur_ms=120, pause_thresh_ms=250, pitch_jump_st=1.5)
    print(f"[EXTRACT] Raw segments: RH={len(notes_high)}, LH={len(notes_low)}")

    # 6. Merge close notes (aggressive)
    notes_high = merge_close_notes(notes_high, max_gap_ms=150, max_pitch_diff_st=1.5)
    notes_low = merge_close_notes(notes_low, max_gap_ms=150, max_pitch_diff_st=1.5)
    print(f"[EXTRACT] After merge: RH={len(notes_high)}, LH={len(notes_low)}")

    # 7. Quantize timing
    notes_high = quantize_notes(notes_high)
    notes_low = quantize_notes(notes_low)

    # 8. Combine and sort
    all_notes = sorted(notes_high + notes_low, key=lambda n: n.start)

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
def generate_musicxml_v72(result: TranscriptionResult, title: str = "PianoMagic Transcription") -> str:
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
    seconds_per_beat = 60.0 / result.tempo
    measure_duration = beats_per_measure * seconds_per_beat

    notes_sorted = sorted(result.notes, key=lambda n: n.start)
    divisions = 4

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
        current_measure = 1
        measure_start = 0.0
        measure = None
        attr_set = False

        for note in notes_sorted:
            while note.start >= measure_start + measure_duration:
                current_measure += 1
                measure_start += measure_duration
                attr_set = False

            if measure is None or note.start >= measure_start + measure_duration:
                measure = SubElement(part, 'measure', number=str(current_measure))

                if not attr_set:
                    attr = SubElement(measure, 'attributes')
                    SubElement(attr, 'divisions').text = str(divisions)

                    if current_measure == 1:
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

                    attr_set = True

            note_dur = note.end - note.start
            dur_divs = max(1, int(round(note_dur / seconds_per_beat * divisions)))

            note_el = SubElement(measure, 'note')
            staff_num = '1' if note.hand == 'RH' else '2'

            step, alter, octave = midi_to_ly_step(note.pitch_midi)
            pitch_el = SubElement(note_el, 'pitch')
            SubElement(pitch_el, 'step').text = step
            if alter != 0:
                SubElement(pitch_el, 'alter').text = str(alter)
            SubElement(pitch_el, 'octave').text = str(octave)

            SubElement(note_el, 'duration').text = str(dur_divs)

            type_name = 'quarter'
            if dur_divs >= 16:
                type_name = 'whole'
            elif dur_divs >= 8:
                type_name = 'half'
            elif dur_divs >= 4:
                type_name = 'quarter'
            elif dur_divs >= 2:
                type_name = 'eighth'
            else:
                type_name = '16th'

            SubElement(note_el, 'type').text = type_name
            SubElement(note_el, 'staff').text = staff_num
            SubElement(note_el, 'voice').text = staff_num

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
    try:
        tasks[task_id]['status'] = 'loading'
        tasks[task_id]['progress'] = 10
        print(f"[TASK {task_id}] Loading audio from {file_path}")

        y, sr = librosa.load(str(file_path), sr=22050, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        print(f"[TASK {task_id}] Audio loaded: {duration:.2f}s, {len(y)} samples")

        tasks[task_id]['status'] = 'analyzing'
        tasks[task_id]['progress'] = 30

        result = extract_melody_librosa_v73(y, sr)
        print(f"[TASK {task_id}] Extracted {len(result.notes)} notes")

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
            'xml_url': f'/download/{file_id}.xml'
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
    return {"version": VERSION, "backend": VERSION, "api": "v1"}

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
    for ext in ['.wav', '.xml']:
        file_path = UPLOAD_DIR / f"PianoMagic_{file_id}{ext}"
        if file_path.exists():
            media_type = 'audio/wav' if ext == '.wav' else 'application/vnd.recordare.musicxml+xml'
            return FileResponse(str(file_path), media_type=media_type, 
                              filename=f"PianoMagic_{file_id}{ext}")

    raise HTTPException(404, "File not found")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
