#!/usr/bin/env python3
"""
PianoMagic v7.2 — Тест гипотезы continuous pitch contour segmentation
Запуск: python test_v72_hypothesis.py kuznechik.mp3
"""
import sys
import librosa
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

def test_v72_hypothesis(audio_path):
    print(f"[v7.2 Test] Загрузка {audio_path}...")
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"  Длительность: {duration:.2f}s, SR: {sr}")

    # 1. PYIN с мелким hop для высокого разрешения
    hop_length = 256
    print(f"[v7.2 Test] PYIN с hop={hop_length}...")
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
        sr=sr, hop_length=hop_length, frame_length=2048
    )
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)

    # 2. Разделение на два голоса по порогу 300 Hz
    threshold_hz = 300
    voice_high = np.where((f0 > threshold_hz) & voiced_flag, f0, np.nan)
    voice_low = np.where((f0 <= threshold_hz) & voiced_flag, f0, np.nan)

    print(f"  Верхний голос (>{threshold_hz}Hz): {np.sum(~np.isnan(voice_high))} фреймов")
    print(f"  Нижний голос (<={threshold_hz}Hz): {np.sum(~np.isnan(voice_low))} фреймов")

    # 3. Интерполяция пропусков в каждом голосе отдельно
    def interpolate_voice(voice):
        valid = ~np.isnan(voice)
        if np.sum(valid) < 2:
            return voice
        x = np.arange(len(voice))
        voice_interp = np.interp(x, x[valid], voice[valid])
        # Сбрасываем интерполированные участки длиннее 150ms обратно в nan
        gaps = np.diff(valid.astype(int), prepend=0, append=0)
        gap_starts = np.where(gaps == -1)[0]
        gap_ends = np.where(gaps == 1)[0]
        for gs, ge in zip(gap_starts, gap_ends):
            gap_dur_ms = (ge - gs) * hop_length / sr * 1000
            if gap_dur_ms > 150:
                voice_interp[gs:ge] = np.nan
        voice_interp[~valid] = np.where(np.isnan(voice_interp[~valid]), np.nan, voice_interp[~valid])
        # На самом деле проще: интерполируем только короткие gaps
        return voice_interp

    # Упрощённая интерполяция: заполняем nan-дырки до 100ms
    def fill_short_gaps(voice, max_gap_ms=100):
        result = voice.copy()
        valid = ~np.isnan(voice)
        gaps = np.diff(valid.astype(int), prepend=0, append=0)
        gap_starts = np.where(gaps == -1)[0]
        gap_ends = np.where(gaps == 1)[0]
        for gs, ge in zip(gap_starts, gap_ends):
            gap_dur_ms = (ge - gs) * hop_length / sr * 1000
            if gap_dur_ms <= max_gap_ms and gs > 0 and ge < len(voice):
                result[gs:ge] = np.linspace(voice[gs-1], voice[ge], ge-gs)
        return result

    vh_filled = fill_short_gaps(voice_high, max_gap_ms=80)
    vl_filled = fill_short_gaps(voice_low, max_gap_ms=80)

    # 4. Сегментация: скачок >0.5 полутона или пауза >100ms
    def segment_voice(filled_voice, times, min_dur_ms=60, pause_thresh_ms=100):
        notes = []
        in_note = False
        note_start = 0
        current_pitch = 0

        for i in range(len(filled_voice)):
            if not np.isnan(filled_voice[i]):
                pitch = filled_voice[i]
                if not in_note:
                    in_note = True
                    note_start = times[i]
                    current_pitch = pitch
                else:
                    # Проверяем скачок
                    semitone_diff = abs(librosa.hz_to_midi(pitch) - librosa.hz_to_midi(current_pitch))
                    if semitone_diff > 0.5:
                        # Завершаем предыдущую ноту
                        note_end = times[i]
                        dur_ms = (note_end - note_start) * 1000
                        if dur_ms >= min_dur_ms:
                            notes.append({
                                'start': note_start,
                                'end': note_end,
                                'pitch_hz': current_pitch,
                                'pitch_midi': round(librosa.hz_to_midi(current_pitch)),
                                'dur_ms': dur_ms
                            })
                        note_start = times[i]
                        current_pitch = pitch
            else:
                if in_note:
                    # Проверяем длину паузы
                    # Ищем следующий валидный фрейм
                    j = i
                    while j < len(filled_voice) and np.isnan(filled_voice[j]):
                        j += 1
                    pause_ms = (times[min(j, len(times)-1)] - times[i]) * 1000 if j < len(times) else 9999
                    if pause_ms > pause_thresh_ms or j >= len(times):
                        note_end = times[i]
                        dur_ms = (note_end - note_start) * 1000
                        if dur_ms >= min_dur_ms:
                            notes.append({
                                'start': note_start,
                                'end': note_end,
                                'pitch_hz': current_pitch,
                                'pitch_midi': round(librosa.hz_to_midi(current_pitch)),
                                'dur_ms': dur_ms
                            })
                        in_note = False
        # Закрываем последнюю ноту
        if in_note:
            note_end = times[-1]
            dur_ms = (note_end - note_start) * 1000
            if dur_ms >= min_dur_ms:
                notes.append({
                    'start': note_start,
                    'end': note_end,
                    'pitch_hz': current_pitch,
                    'pitch_midi': round(librosa.hz_to_midi(current_pitch)),
                    'dur_ms': dur_ms
                })
        return notes

    print("[v7.2 Test] Сегментация верхнего голоса...")
    notes_high = segment_voice(vh_filled, times, min_dur_ms=60, pause_thresh_ms=100)
    print(f"  Нот верхнего голоса: {len(notes_high)}")

    print("[v7.2 Test] Сегментация нижнего голоса...")
    notes_low = segment_voice(vl_filled, times, min_dur_ms=60, pause_thresh_ms=100)
    print(f"  Нот нижнего голоса: {len(notes_low)}")

    # 5. Визуализация
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    # Waveform
    ax = axes[0]
    librosa.display.waveshow(y, sr=sr, ax=ax, alpha=0.6)
    ax.set_title('Waveform')
    ax.set_ylabel('Amplitude')

    # Original f0
    ax = axes[1]
    ax.plot(times, f0, 'k.', markersize=1, alpha=0.5, label='Original f0')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('PYIN f0 (original)')
    ax.legend()
    ax.set_ylim(50, 1000)

    # Two voices
    ax = axes[2]
    ax.plot(times, vh_filled, 'r.', markersize=2, label='High voice (>300Hz)', alpha=0.7)
    ax.plot(times, vl_filled, 'b.', markersize=2, label='Low voice (≤300Hz)', alpha=0.7)
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Voice Separation (interpolated)')
    ax.legend()
    ax.set_ylim(50, 1000)

    # Segmented notes as piano roll
    ax = axes[3]
    for note in notes_high:
        ax.barh(note['pitch_midi'], note['end']-note['start'], left=note['start'], 
                height=0.8, color='red', alpha=0.7, edgecolor='darkred')
    for note in notes_low:
        ax.barh(note['pitch_midi'], note['end']-note['start'], left=note['start'], 
                height=0.8, color='blue', alpha=0.7, edgecolor='darkblue')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('MIDI Note')
    ax.set_title(f'Segmented Notes: RH={len(notes_high)} notes, LH={len(notes_low)} notes')
    ax.set_xlim(0, duration)

    plt.tight_layout()
    out_path = Path(audio_path).stem + '_v72_analysis.png'
    plt.savefig(out_path, dpi=150)
    print(f"[v7.2 Test] График сохранён: {out_path}")

    # 6. Статистика
    print("\n=== СТАТИСТИКА v7.2 ===")
    print(f"Верхний голос (RH): {len(notes_high)} нот")
    if notes_high:
        pitches_h = [n['pitch_midi'] for n in notes_high]
        print(f"  Диапазон MIDI: {min(pitches_h)} – {max(pitches_h)}")
        print(f"  Средняя длительность: {np.mean([n['dur_ms'] for n in notes_high]):.1f}ms")
    print(f"Нижний голос (LH): {len(notes_low)} нот")
    if notes_low:
        pitches_l = [n['pitch_midi'] for n in notes_low]
        print(f"  Диапазон MIDI: {min(pitches_l)} – {max(pitches_l)}")
        print(f"  Средняя длительность: {np.mean([n['dur_ms'] for n in notes_low]):.1f}ms")
    print(f"Всего нот: {len(notes_high) + len(notes_low)}")
    print("========================")

    return notes_high, notes_low

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python test_v72_hypothesis.py <audio_file>")
        sys.exit(1)
    test_v72_hypothesis(sys.argv[1])
