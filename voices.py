"""
Voice separation: split polyphony into voices, then take one.

Everything before this picked a melody by scoring paths - "which single
chain of notes has the most salience and the least motion". That is the
wrong question. Polyphony is not one line plus rubbish; it is several
lines sounding at once, and the melody is one of them. Asked to choose a
path, the search happily stitches together fragments of different voices
whenever that scores well, which is exactly the mush the v7.8.x output
turned out to be: right pitch classes, no tune.

Asked instead to PARTITION the notes into voices, the problem becomes
well posed, and the melody is then simply the top part.

The method is contig mapping (Chew & Wu, 2004):

  1. cut time wherever a note starts or ends; between cuts the set of
     sounding notes is constant
  2. a CONTIG is a maximal run of those slices holding the same number
     of notes - inside one, nothing enters or leaves, so the voices can
     be read straight off by pitch order (voices are assumed not to
     cross, which is what a listener assumes too)
  3. neighbouring contigs are then joined by matching their voice ends
     to voice starts at minimum total pitch distance

The assumption that voices do not cross is wrong occasionally and right
overwhelmingly; it is what makes the problem tractable.
"""
from typing import List, Sequence, Tuple

Note = Tuple[float, float, int, float]        # start, end, midi, salience


def _cut_points(notes: Sequence[Note], tol: float) -> List[float]:
    """Boundaries, with near-coincident ones fused.

    Without this, legato wrecks the method. Real notes overlap their
    neighbour by a few tens of milliseconds, and every such overlap adds
    two extra boundaries and a sliver slice where the sounding count
    momentarily differs - which splits the piece into hundreds of
    one-slice contigs with nothing to align. Measured on a three-voice
    test, 60 ms of legato cost half the melody, the same failure the
    line-picking code had for the same underlying reason. Boundaries
    within tol of each other are one boundary."""
    raw = sorted({round(v, 6) for s, e, _p, _a in notes for v in (s, e)})
    if not raw:
        return []
    pts = [raw[0]]
    for v in raw[1:]:
        if v - pts[-1] > tol:
            pts.append(v)
    return pts


def _slices(notes: Sequence[Note], tol: float):
    """(t0, t1, [note indices sounding]) with a constant sounding set.

    A note counts as sounding in a slice when it covers most of it, not
    when it merely clips the edge - again so a legato tail does not
    masquerade as a voice."""
    pts = _cut_points(notes, tol)
    out = []
    for t0, t1 in zip(pts, pts[1:]):
        if t1 - t0 <= tol:
            continue
        lo, hi = t0 + (t1 - t0) * 0.25, t0 + (t1 - t0) * 0.75
        idx = [i for i, (s, e, _p, _a) in enumerate(notes) if s <= lo and e >= hi]
        if idx:
            out.append((t0, t1, sorted(idx, key=lambda i: notes[i][2])))
    return out


def _contigs(slices):
    """Maximal runs of slices with the same number of sounding notes AND
    the same notes - a contig must not have anything enter or leave."""
    out = []
    for t0, t1, idx in slices:
        if out and out[-1][2] == idx and abs(out[-1][1] - t0) < 1e-9:
            out[-1] = (out[-1][0], t1, idx)
        else:
            out.append((t0, t1, idx))
    return out


def _match(prev_pitches, next_pitches):
    """Cheapest way to join k voice ends to m voice starts, order kept.
    Voices do not cross, so this is a monotone alignment and a small
    dynamic program solves it exactly - no need for Hungarian."""
    k, m = len(prev_pitches), len(next_pitches)
    INF = float('inf')
    best = [[INF] * (m + 1) for _ in range(k + 1)]
    back = [[None] * (m + 1) for _ in range(k + 1)]
    best[0][0] = 0.0
    SKIP = 12.0                     # cost of a voice ending or starting
    for i in range(k + 1):
        for j in range(m + 1):
            if best[i][j] == INF:
                continue
            if i < k and j < m:
                c = best[i][j] + abs(prev_pitches[i] - next_pitches[j])
                if c < best[i + 1][j + 1]:
                    best[i + 1][j + 1], back[i + 1][j + 1] = c, ('m', i, j)
            if i < k and best[i][j] + SKIP < best[i + 1][j]:
                best[i + 1][j], back[i + 1][j] = best[i][j] + SKIP, ('e', i, j)
            if j < m and best[i][j] + SKIP < best[i][j + 1]:
                best[i][j + 1], back[i][j + 1] = best[i][j] + SKIP, ('s', i, j)
    pairs, i, j = [], k, m
    while (i, j) != (0, 0):
        op, pi, pj = back[i][j]
        if op == 'm':
            pairs.append((pi, pj))
        i, j = pi, pj
    return list(reversed(pairs))


def separate(notes: Sequence[Note], tol: float = 0.08) -> List[int]:
    """Return a voice id per note; 0 is the lowest voice in its contig.

    tol is how far apart two note boundaries must be to count as
    separate events - i.e. how much legato overlap to forgive."""
    if not notes:
        return []
    contigs = _contigs(_slices(notes, tol))
    if not contigs:
        return [0] * len(notes)

    # Inside a contig, voice order is pitch order.
    local = []
    for _t0, _t1, idx in contigs:
        local.append(list(idx))                # already sorted by pitch

    # Walk left to right, carrying voice identities across boundaries.
    n_voices = max(len(c) for c in local)
    voice_of = {}
    ids = list(range(len(local[0])))
    for i, note_i in enumerate(local[0]):
        voice_of.setdefault(note_i, ids[i])
    prev_ids = ids
    for a in range(1, len(local)):
        prev, cur = local[a - 1], local[a]
        pairs = _match([notes[i][2] for i in prev], [notes[i][2] for i in cur])
        cur_ids = [None] * len(cur)
        for pi, pj in pairs:
            cur_ids[pj] = prev_ids[pi]
        free = [v for v in range(n_voices) if v not in cur_ids]
        for j in range(len(cur)):
            if cur_ids[j] is None:
                cur_ids[j] = free.pop(0) if free else n_voices - 1
        for j, note_j in enumerate(cur):
            voice_of.setdefault(note_j, cur_ids[j])
        prev_ids = cur_ids
    return [voice_of.get(i, 0) for i in range(len(notes))]


def top_voice(notes: Sequence[Note], tol: float = 0.08,
              min_share: float = 0.12, min_salience: float = 0.35) -> List[int]:
    """
    Indices of the notes belonging to the highest substantial voice.

    Separation is purely combinatorial - it counts what is sounding and
    orders it by pitch, and knows nothing about loudness. A partial that
    harmonic suppression damped to a quarter of its weight is, to this
    code, exactly as much a note as the tune. On the real run that is
    what happened: the top voice climbed to MIDI 89 on a trail of
    partials and scored 16% octave leaps. So the faint stuff is removed
    before separation rather than left to form voices of its own;
    min_salience is a fraction of the median note salience.
    """
    if not notes:
        return []
    sal = sorted((e - s) * a for s, e, _p, a in notes)
    floor_sal = min_salience * sal[len(sal) // 2]
    keep = [i for i, (s, e, _p, a) in enumerate(notes)
            if (e - s) * a >= floor_sal]
    if len(keep) < 8:
        keep = list(range(len(notes)))
    sub = [notes[i] for i in keep]

    vid_sub = separate(sub, tol)
    if not vid_sub:
        return []
    vid = [None] * len(notes)
    for k, i in enumerate(keep):
        vid[i] = vid_sub[k]
    # "Highest" by mean pitch, but only among voices that are actually
    # THERE. Height alone is fooled by rubbish: twenty stray detections
    # scattered above the music form their own sparse top voice, and a
    # test with that added returned two junk notes and none of the tune.
    # A melody runs through the piece; a voice holding a handful of notes
    # is not a candidate however high it sits.
    from collections import defaultdict
    acc = defaultdict(list)
    for i, v in enumerate(vid):
        if v is not None:
            acc[v].append(notes[i][2])
    if not acc:
        return []
    floor = max(3, int(min_share * len(keep)))
    real = {v: ps for v, ps in acc.items() if len(ps) >= floor} or acc
    best = max(real, key=lambda v: sum(real[v]) / len(real[v]))
    return [i for i, v in enumerate(vid) if v == best]
