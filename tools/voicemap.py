#!/usr/bin/env python3
"""
Look at the voices before deciding which one is the tune.

    python3 tools/voicemap.py <task-id>          # fetch a run's raw events
    python3 tools/voicemap.py events.json        # or read them from disk
    python3 tools/voicemap.py <task-id> --out map.html

Every decision about the melody so far has been made blind - a number
came out, we argued about what it meant. This draws the actual note
cloud, split into voices, so the structure can be looked at: where each
part sits, how much of the sound it accounts for, whether it moves like
a melody or stands like an accompaniment.

Two outputs. A short text table for the terminal - small enough to paste
into a conversation - and a self-contained HTML piano roll coloured by
voice, for looking at properly.

Standard library only.
"""
import argparse, json, os, sys, urllib.request
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import voices as V

DEFAULT_API = os.environ.get('PIANOMAGIC_API', 'https://pianomagic-api.onrender.com')
PALETTE = ['#2f6fb0', '#c0392b', '#27884a', '#8e44ad', '#d68910',
           '#16a085', '#7f8c8d', '#c2185b']


def load(src, api):
    if Path(src).exists():
        data = json.loads(Path(src).read_text())
    else:
        task = src.split('/')[-1].replace('.json', '')
        url = f"{api}/download/{task}.json"
        print(f"fetching {url}")
        with urllib.request.urlopen(url, timeout=120) as r:
            data = json.loads(r.read())
    ev = [(float(a), float(b), int(p), float(s)) for a, b, p, s in data['events']]
    return data, sorted(ev, key=lambda e: (e[0], e[2]))


def motion(pitches):
    if len(pitches) < 2:
        return 0.0, 0.0, 0.0
    j = [abs(b - a) for a, b in zip(pitches, pitches[1:])]
    n = float(len(j))
    return (sum(1 for x in j if x == 0) / n,
            sum(1 for x in j if 1 <= x <= 2) / n,
            sum(1 for x in j if x >= 12) / n)


def summarise(events, vid):
    total_sal = sum((e - s) * a for s, e, _p, a in events) or 1.0
    span = max(e for _s, e, _p, _a in events) or 1.0
    rows = []
    for v in sorted({x for x in vid if x is not None}):
        idx = [i for i, x in enumerate(vid) if x == v]
        idx.sort(key=lambda i: events[i][0])
        ps = [events[i][2] for i in idx]
        sal = sum((events[i][1] - events[i][0]) * events[i][3] for i in idx)
        rep, step, oct_ = motion(ps)
        rows.append({
            'voice': v, 'notes': len(idx), 'lo': min(ps), 'hi': max(ps),
            'median': sorted(ps)[len(ps) // 2], 'distinct': len(set(ps)),
            'per_sec': len(idx) / span, 'share': sal / total_sal,
            'rep': rep, 'step': step, 'oct': oct_,
        })
    return rows


def text_report(data, events, vid, rows):
    out = [f"raw events: {len(events)}   tempo {data.get('tempo')}   "
           f"key {data.get('key')}   backend v{data.get('version')}",
           f"voices found: {len(rows)}", "",
           "  v   notes  register   med  distinct  notes/s  loudness  "
           "repeat  step  oct+",
           "  " + "-" * 74]
    for r in rows:
        out.append(
            f"  {r['voice']:<3d} {r['notes']:6d}  {r['lo']:3d}-{r['hi']:<3d}   "
            f"{r['median']:3d}  {r['distinct']:8d}  {r['per_sec']:7.2f}  "
            f"{r['share']*100:7.0f}%  {r['rep']*100:5.0f}% {r['step']*100:5.0f}% "
            f"{r['oct']*100:5.0f}%")
    unassigned = sum(1 for x in vid if x is None)
    if unassigned:
        out.append(f"  (+{unassigned} events below the salience gate, not in any voice)")
    return "\n".join(out)


def html(data, events, vid, rows, title):
    span = max(e for _s, e, _p, _a in events)
    lo = min(p for _s, _e, p, _a in events) - 1
    hi = max(p for _s, _e, p, _a in events) + 1
    W, H = 2400, max(360, (hi - lo) * 9)
    sx = W / span
    sy = H / (hi - lo)
    parts = []
    for pitch in range(lo, hi + 1):
        if pitch % 12 == 0:
            y = H - (pitch - lo) * sy
            parts.append(f'<line x1="0" y1="{y:.1f}" x2="{W}" y2="{y:.1f}" '
                         f'class="oct"/>')
            parts.append(f'<text x="2" y="{y-2:.1f}" class="lbl">C{pitch//12-1}</text>')
    for sec in range(0, int(span) + 1, 5):
        x = sec * sx
        parts.append(f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{H}" class="grid"/>')
        parts.append(f'<text x="{x+2:.1f}" y="12" class="lbl">{sec}s</text>')
    for i, (s, e, p, a) in enumerate(events):
        v = vid[i]
        col = '#cfd6dd' if v is None else PALETTE[v % len(PALETTE)]
        y = H - (p - lo) * sy
        parts.append(
            f'<rect x="{s*sx:.1f}" y="{y-3:.1f}" width="{max(1.5,(e-s)*sx):.1f}" '
            f'height="6" fill="{col}" opacity="{0.35 + 0.6*min(1.0, a):.2f}"/>')
    legend = "".join(
        f'<span class="k"><i style="background:{PALETTE[r["voice"]%len(PALETTE)]}"></i>'
        f'voice {r["voice"]} — {r["notes"]} notes, MIDI {r["lo"]}–{r["hi"]}, '
        f'{r["share"]*100:.0f}% of the sound</span>' for r in rows)
    table = "".join(
        f"<tr><td>{r['voice']}</td><td>{r['notes']}</td>"
        f"<td>{r['lo']}–{r['hi']}</td><td>{r['median']}</td><td>{r['distinct']}</td>"
        f"<td>{r['per_sec']:.2f}</td><td>{r['share']*100:.0f}%</td>"
        f"<td>{r['rep']*100:.0f}%</td><td>{r['step']*100:.0f}%</td>"
        f"<td>{r['oct']*100:.0f}%</td></tr>" for r in rows)
    return f"""<title>{title}</title>
<style>
 :root{{--bg:#fbfcfd;--fg:#1b2733;--mut:#5b6b7b;--line:#e2e8ee;--card:#fff}}
 :root:not([data-theme="light"]){{}}
 @media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{
   --bg:#12171c;--fg:#e7edf3;--mut:#9aabbb;--line:#26313c;--card:#181f26}}}}
 :root[data-theme="dark"]{{--bg:#12171c;--fg:#e7edf3;--mut:#9aabbb;
   --line:#26313c;--card:#181f26}}
 body{{background:var(--bg);color:var(--fg);margin:0;padding:24px;
   font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}}
 h1{{font-size:1.35rem;margin:0 0 4px}} p.sub{{color:var(--mut);margin:0 0 20px}}
 .card{{background:var(--card);border:1px solid var(--line);border-radius:10px;
   padding:16px;margin-bottom:20px}}
 .roll{{overflow-x:auto}} svg{{display:block}}
 .oct{{stroke:var(--line);stroke-width:1}} .grid{{stroke:var(--line);
   stroke-width:1;stroke-dasharray:3 5}}
 .lbl{{fill:var(--mut);font-size:10px}}
 .k{{display:inline-flex;align-items:center;gap:6px;margin:0 16px 8px 0;
   font-size:.86rem;color:var(--mut)}}
 .k i{{width:11px;height:11px;border-radius:2px;display:inline-block}}
 table{{border-collapse:collapse;width:100%;font-size:.88rem}}
 th,td{{text-align:right;padding:6px 10px;border-bottom:1px solid var(--line)}}
 th:first-child,td:first-child{{text-align:left}}
 th{{color:var(--mut);font-weight:600}}
</style>
<h1>{title}</h1>
<p class="sub">{len(events)} raw detections split into {len(rows)} voices ·
 tempo {data.get('tempo')} · key {data.get('key')} · backend v{data.get('version')}</p>
<div class="card"><div>{legend}</div>
 <div class="roll"><svg width="{W}" height="{H}" viewBox="0 0 {W} {H}">{''.join(parts)}</svg></div></div>
<div class="card"><table>
<tr><th>voice</th><th>notes</th><th>register</th><th>median</th><th>distinct</th>
<th>notes/s</th><th>loudness</th><th>repeat</th><th>step</th><th>oct+</th></tr>
{table}</table></div>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('source', help='task id, download URL, or a local .json')
    ap.add_argument('--api', default=DEFAULT_API)
    ap.add_argument('--out', default='voicemap.html')
    ap.add_argument('--tol', type=float, default=0.08)
    ap.add_argument('--min-salience', type=float, default=0.35)
    args = ap.parse_args()

    data, events = load(args.source, args.api)
    if not events:
        sys.exit("no events in that file")

    sal = sorted((e - s) * a for s, e, _p, a in events)
    floor = args.min_salience * sal[len(sal) // 2]
    keep = [i for i, (s, e, _p, a) in enumerate(events) if (e - s) * a >= floor]
    sub = [events[i] for i in keep]
    vid_sub = V.separate(sub, args.tol)
    vid = [None] * len(events)
    for k, i in enumerate(keep):
        vid[i] = vid_sub[k]

    rows = summarise(events, vid)
    print(text_report(data, events, vid, rows))
    Path(args.out).write_text(
        html(data, events, vid, rows, "PianoMagic — voice map"))
    print(f"\nwritten to {args.out}")


if __name__ == '__main__':
    main()
