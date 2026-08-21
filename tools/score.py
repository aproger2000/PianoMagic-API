"""
Score a transcription against a known answer.

Reports the two things that actually matter for this project, kept
apart on purpose:

  NOTES  - did we find the right notes at the right moments? Standard
           onset+pitch matching: a reference note counts as found when
           some estimated note shares its pitch and starts within a
           tolerance of it. Precision punishes invented notes, recall
           punishes missing ones, F1 is the single number to watch.

  CONTOUR- at each moment the tune is sounding, are we playing the note
           the tune is playing? A transcription can score badly on notes
           (wrong rhythm, split notes) and still trace the melody, or
           score well and still wander - so this is measured separately.

Octave errors are counted apart from outright wrong pitches, because
they are a different failure with a different fix.
"""
import xml.etree.ElementTree as ET

STEP = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def read_musicxml(path):
    """(start_s, duration_s, midi) for every note in a MusicXML score."""
    root = ET.parse(path).getroot()
    part = root.find('.//part')
    divisions, tempo, cursor, out = None, None, 0.0, []
    for measure in part.findall('measure'):
        attrs = measure.find('attributes')
        if attrs is not None and attrs.find('divisions') is not None:
            divisions = int(attrs.find('divisions').text)
        pm = measure.find('.//per-minute')
        if pm is not None:
            tempo = float(pm.text)
        for el in measure:
            if el.tag == 'note':
                dur = int(el.find('duration').text) if el.find('duration') is not None else 0
                if el.find('rest') is None:
                    p = el.find('pitch')
                    alter = int(p.find('alter').text) if p.find('alter') is not None else 0
                    midi = ((int(p.find('octave').text) + 1) * 12
                            + STEP[p.find('step').text] + alter)
                    out.append((cursor, dur, midi))
                cursor += dur
            elif el.tag == 'backup':
                cursor -= int(el.find('duration').text)
            elif el.tag == 'forward':
                cursor += int(el.find('duration').text)
    if not divisions or not tempo:
        raise ValueError(f"{path}: missing divisions or tempo")
    q = 60.0 / tempo
    return [(c / divisions * q, d / divisions * q, m) for c, d, m in out]


def score(reference, estimate, onset_tol=0.10):
    ref = sorted(reference, key=lambda n: n[0])
    est = sorted(estimate, key=lambda n: n[0])
    used = set()
    hits = octave = 0
    for rs, _rd, rp in ref:
        best = None
        for i, (es, _ed, ep) in enumerate(est):
            if i in used or abs(es - rs) > onset_tol:
                continue
            if ep == rp:
                best = i
                break
            if best is None and abs(ep - rp) % 12 == 0:
                best = i
        if best is None:
            continue
        used.add(best)
        if est[best][2] == rp:
            hits += 1
        else:
            octave += 1
    found = hits + octave
    precision = found / len(est) if est else 0.0
    recall = found / len(ref) if ref else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    # contour: what is sounding in the middle of each reference note
    right = wrong = silent = 0
    for rs, rd, rp in ref:
        mid = rs + rd / 2
        sounding = [p for s, d, p in est if s <= mid < s + d]
        if not sounding:
            silent += 1
        elif rp in sounding:
            right += 1
        else:
            wrong += 1
    contour = right / len(ref) if ref else 0.0

    return {
        'ref_notes': len(ref), 'est_notes': len(est),
        'matched': found, 'exact': hits, 'octave_off': octave,
        'missed': len(ref) - found, 'spurious': len(est) - found,
        'precision': precision, 'recall': recall, 'f1': f1,
        'contour': contour, 'contour_wrong': wrong, 'contour_silent': silent,
    }


def format_report(name, s):
    return (
        f"{name}\n"
        f"  notes    ref {s['ref_notes']:3d} | est {s['est_notes']:3d} | "
        f"matched {s['matched']:3d} (exact {s['exact']}, octave off {s['octave_off']})\n"
        f"           missed {s['missed']:3d} | spurious {s['spurious']:3d}\n"
        f"  P {s['precision']:.2f}  R {s['recall']:.2f}  F1 {s['f1']:.2f}\n"
        f"  contour  {s['contour']*100:4.0f}% of the tune's time on the right note "
        f"(wrong {s['contour_wrong']}, silent {s['contour_silent']})"
    )
