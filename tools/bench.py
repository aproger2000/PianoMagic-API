#!/usr/bin/env python3
"""
End-to-end benchmark: build reference audio, transcribe it, score the result.

    python3 tools/bench.py                     # both cases against the live API
    python3 tools/bench.py --case mono         # just the solo line
    python3 tools/bench.py --api http://localhost:10000
    python3 tools/bench.py --keep out/         # also write the audio, XML and logs

Why this exists: every verdict on this transcriber up to now has been a
listening impression, one deploy cycle each, not comparable between
versions. This prints numbers instead, in about a minute, and the mono
vs poly pair says whether a loss comes from polyphony or from the
pipeline itself.

Standard library only, numpy included - it has to run wherever the
service is reachable, with nothing installed first.
"""
import argparse, json, mimetypes, os, sys, time, urllib.request, urllib.error, uuid, wave
from array import array
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reference as R
import score as S

DEFAULT_API = "https://pianomagic-api.onrender.com"


def write_wav(path, y, sr=R.SR):
    pcm = array('h', (int(max(-1.0, min(1.0, v)) * 32767) for v in y))
    if sys.byteorder == 'big':
        pcm.byteswap()
    with wave.open(str(path), 'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr)
        f.writeframes(pcm.tobytes())


def post_file(api, path):
    boundary = uuid.uuid4().hex
    ctype = mimetypes.guess_type(str(path))[0] or 'application/octet-stream'
    body = b''.join([
        f'--{boundary}\r\n'.encode(),
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'.encode(),
        f'Content-Type: {ctype}\r\n\r\n'.encode(),
        path.read_bytes(), b'\r\n', f'--{boundary}--\r\n'.encode(),
    ])
    req = urllib.request.Request(
        f"{api}/upload", data=body,
        headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)['task_id']


def get(api, path, binary=False, timeout=120):
    with urllib.request.urlopen(f"{api}{path}", timeout=timeout) as r:
        return r.read() if binary else json.load(r)


def transcribe(api, audio_path, poll=3.0, limit=900):
    task = post_file(api, audio_path)
    deadline = time.time() + limit
    last = None
    while time.time() < deadline:
        st = get(api, f"/status/{task}")
        if st['status'] != last:
            last = st['status']
            print(f"    {last} ...", flush=True)
        if st['status'] == 'completed':
            return task, st['result']
        if st['status'] == 'error':
            raise RuntimeError(st.get('error', 'transcription failed'))
        time.sleep(poll)
    raise TimeoutError(f"no result after {limit}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--api', default=os.environ.get('PIANOMAGIC_API', DEFAULT_API))
    ap.add_argument('--case', choices=list(R.CASES) + ['both', 'all'],
                    default='all',
                    help="'all' walks the hardening ladder; 'both' is just mono+poly")
    ap.add_argument('--keep', metavar='DIR', help='keep audio, XML and run logs here')
    args = ap.parse_args()

    out = Path(args.keep) if args.keep else Path('.bench_tmp')
    out.mkdir(parents=True, exist_ok=True)
    cases = (list(R.CASES) if args.case == 'all'
             else ['mono', 'poly'] if args.case == 'both'
             else [args.case])
    ref = R.melody()
    print(f"reference: {len(ref)} notes, "
          f"{max(s+d for s,d,_ in ref):.1f}s, MIDI {min(p for _,_,p in ref)}-"
          f"{max(p for _,_,p in ref)}, {R.BPM} BPM")
    print(f"api: {args.api}\n")

    results = {}
    for case in cases:
        wav = out / f"reference_{case}.wav"
        write_wav(wav, R.render(case))
        print(f"[{case}] {wav.name} -> transcribing")
        task, res = transcribe(args.api, wav)
        xml_path = out / f"result_{case}.xml"
        xml_path.write_bytes(get(args.api, res['xml_url'], binary=True))
        try:
            (out / f"log_{case}.txt").write_bytes(get(args.api, f"/logs/{task}", binary=True))
        except Exception:
            pass
        est = S.read_musicxml(xml_path)
        st = S.score(ref, est)
        st['engine'] = res.get('engine')
        st['key'] = res.get('key')
        st['tempo'] = res.get('tempo')
        results[case] = st
        print(S.format_report(f"[{case}] engine={st['engine']} key={st['key']} "
                              f"tempo={st['tempo']}", st))
        print()

    if len(results) > 1:
        print("ladder")
        prev = None
        worst = None
        for case in cases:
            r = results.get(case)
            if not r:
                continue
            delta = "" if prev is None else f"   ({r['f1'] - prev:+.2f})"
            print(f"  {case:8s} F1 {r['f1']:.2f}  contour {r['contour']*100:3.0f}%"
                  f"  spurious {r['spurious']:3d}{delta}")
            if prev is not None and prev - r['f1'] > 0.15 and worst is None:
                worst = case
            prev = r['f1']
        if worst:
            print(f"  -> the score falls off at '{worst}'; that property is "
                  f"what the pipeline cannot handle.")
        elif prev is not None and prev >= 0.7:
            print("  -> every rung holds. Whatever breaks the real recording "
                  "is still not reproduced here.")

    (out / 'bench.json').write_text(json.dumps(results, indent=2))
    print(f"\nwritten to {out}/")


if __name__ == '__main__':
    try:
        main()
    except urllib.error.URLError as e:
        sys.exit(f"cannot reach the API: {e}")
