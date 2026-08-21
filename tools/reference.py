"""
Synthetic reference material with an exactly known answer.

Every judgement about this transcriber so far has been a listening
impression - "closer", "mush", "notes missing". Those cost a deploy cycle
each and cannot be compared across versions. A reference whose note list
we wrote ourselves can be scored by machine, in seconds, to one decimal.

Two renderings of the SAME melody are produced:

    mono - the tune alone
    poly - the identical tune under a left hand playing bass and chords

so the one variable between them is polyphony. If mono scores well and
poly does not, the loss is caused by the accompaniment and by nothing
else; the pair settles that question without a listening test.

The melody is "В траве сидел кузнечик" as the reference chart writes it:
1=C 3=E 4=F 5=G, and 5|6 the black key between G and A, i.e. G#.
"""
import math
from array import array

SR = 22050          # the backend resamples to this anyway; half the work
BPM = 99.4                      # matches the real recording under test
BEAT = 60.0 / BPM
E, Q, H = 0.5, 1.0, 2.0         # eighth / quarter / half, in beats
OVERLAP = 0.05                  # legato: a note rings 50 ms into the next
NOTE = {'1': 60, '3': 64, '4': 65, '5': 67, '56': 68}
HARMONICS = [(1, 1.0), (2, .45), (3, .22), (4, .12), (5, .07), (6, .04)]


def _phrase(symbols: str, last: float = Q):
    s = symbols.split()
    return [(NOTE[x], E) for x in s[:-1]] + [(NOTE[s[-1]], last)]


def melody():
    """(start_s, duration_s, midi) for every note of the tune."""
    song = []
    song += _phrase("4 1 4 1 4 3 3")          # В траве сидел кузнечик
    song += [(None, E)]
    song += _phrase("3 1 3 1 3 4 4")          # В траве сидел кузнечик,
    song += [(None, E)]
    song += _phrase("4 1 4 1 4 3 3")          # Совсем как огуречик,
    song += [(None, E)]
    song += _phrase("3 1 3 1 3 4", H)         # Зелёненький он был
    song += [(None, Q)]
    for _ in range(2):
        song += _phrase("4 5 5 5 5 5 56 56 56 56", Q)   # Представьте себе...
        song += [(None, E)]
        song += _phrase("56 56 5 4 3 4", H)
        song += [(None, Q)]
    out, t = [], 0.0
    for pitch, dur in song:
        if pitch is not None:
            out.append((t, dur * BEAT, pitch))
        t += dur * BEAT
    return out


_CACHE = {}


def _voice(midi, decay, seconds):
    """One struck note, cached per (pitch, decay) at the longest length
    needed and sliced for shorter ones. Pure Python is fast enough only
    because the piece uses ten distinct pitches, not six hundred notes'
    worth of fresh sine sums."""
    key = (midi, decay)
    have = _CACHE.get(key)
    if have is not None and len(have) >= int(seconds * SR):
        return have
    n = int(seconds * SR) + 1
    f = 440.0 * 2 ** ((midi - 69) / 12.0)
    buf = array('d', bytes(8 * n))
    for h, a in HARMONICS:
        w = 2 * math.pi * f * h / SR
        if f * h > SR * 0.45:                      # above Nyquist: skip
            continue
        for i in range(n):
            buf[i] += a * math.sin(w * i)
    for i in range(n):
        t = i / SR
        buf[i] *= math.exp(-t * decay) * (1 - math.exp(-t * 400))
    _CACHE[key] = buf
    return buf


def _tone(midi, dur, amp, decay=2.2):
    seconds = dur + OVERLAP
    src = _voice(midi, decay, max(seconds, 2.5))
    n = int(seconds * SR)
    return [amp * src[i] for i in range(n)]


def _mix(y, w, at):
    i = int(at * SR)
    for k, v in enumerate(w):
        y[i + k] += v


def render(with_accompaniment: bool):
    notes = melody()
    end = max(s + d for s, d, _ in notes) + 1.2
    y = array('d', bytes(8 * int(end * SR)))
    for start, dur, midi in notes:
        _mix(y, _tone(midi, dur, 1.0), start)

    if with_accompaniment:
        # The left hand sits strictly BELOW the tune (which lives at
        # 60-68) and plays bass-on-the-beat, chord-on-the-offbeat. Only
        # fifths and octaves: the accompaniment must never double a
        # melody pitch, or the two parts stop being separable even by
        # ear - an earlier draft put C4 and F4 in the chords and it
        # sounded like the melody echoing itself.
        prog = [(41, [48, 53]), (36, [43, 48])]      # F / C
        last = max(s + d for s, d, _ in notes)
        t, k = 0.0, 0
        while t < last:
            bass, chord = prog[(int(t / (2 * BEAT))) % 2]
            if k % 2 == 0:
                _mix(y, _tone(bass, E * BEAT, 1.10, decay=1.4), t)
            else:
                for c in chord:
                    _mix(y, _tone(c, E * BEAT, 0.62, decay=2.4), t)
            t += E * BEAT
            k += 1

    peak = max(abs(v) for v in y) * 1.05
    return array('d', (v / peak for v in y))
