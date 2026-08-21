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
import numpy as np

SR = 44100
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


def _tone(midi, dur, amp, decay=2.2):
    n = int((dur + OVERLAP) * SR)
    t = np.arange(n) / SR
    f = 440.0 * 2 ** ((midi - 69) / 12.0)
    env = np.exp(-t * decay) * (1 - np.exp(-t * 400))       # percussive
    wave = sum(a * np.sin(2 * np.pi * f * h * t) for h, a in HARMONICS)
    return amp * wave * env


def render(with_accompaniment: bool):
    notes = melody()
    end = max(s + d for s, d, _ in notes) + 1.2
    y = np.zeros(int(end * SR))
    for start, dur, midi in notes:
        w = _tone(midi, dur, 1.0)
        i = int(start * SR)
        y[i:i + len(w)] += w

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
            i = int(t * SR)
            if k % 2 == 0:
                w = _tone(bass, E * BEAT, 1.10, decay=1.4)
                y[i:i + len(w)] += w
            else:
                for c in chord:
                    w = _tone(c, E * BEAT, 0.62, decay=2.4)
                    y[i:i + len(w)] += w
            t += E * BEAT
            k += 1

    return y / (np.abs(y).max() * 1.05)
