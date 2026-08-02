"""
PianoMagic API v7.1 — Clean Melody Engine
==========================================
Key fixes:
  1. Strict onset filtering (adaptive threshold, local maxima only)
  2. Pitch contour temporal smoothing + octave correction
  3. Note salience pruning (drop bottom 20% weak notes)
  4. Minimal LH: chord on measure boundaries only
  5. RH: melody only, harmony only on long stable notes
  6. Clean MusicXML with proper rests
"""

import os, uuid, shutil, subprocess, logging, struct, math, json
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring
from typing import List, Dict, Tuple, Optional

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

try:
    import librosa
    HAS_LIBROSA = True
    logger.info("[INIT] librosa ready")
except ImportError:
    HAS_LIBROSA = False
    logger.warning("[INIT] librosa missing")

app = FastAPI(title="PianoMagic API", version="7.1.0", docs_url=None, redoc_url=None)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["GET","POST","OPTIONS"], allow_headers=["*"])

@app.options("/{path:path}")
async def options_handler(path: str):
    return JSONResponse({}, status_code=200)

JOBS_DIR = Path("/tmp/pianomagic_jobs")
JOBS_DIR.mkdir(exist_ok=True)
jobs: Dict[str, dict] = {}

SR = 44100
HOP_LENGTH = 512
N_FFT = 2048

NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
KS_MAJOR = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
KS_MINOR = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

STAGES = {
    "upload": {"text":"Загрузка файла...","progress":5},
    "analyze":{"text":"Анализ аудио...","progress":15},
    "melody": {"text":"Извлечение мелодии...","progress":30},
    "fit":    {"text":"Подбор партии...","progress":50},
    "compare":{"text":"Сличение отпечатков...","progress":70},
    "render": {"text":"Рендер WAV/PDF...","progress":90},
    "done":   {"text":"Готово!","progress":100},
}

# ============================================================
# Endpoints
# ============================================================
@app.get("/")
async def root():
    return {"status":"ok","version":"7.1","engine":"librosa" if HAS_LIBROSA else "fallback","has_librosa":HAS_LIBROSA}

@app.post("/analyze")
async def analyze(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename or "input.mp3").suffix
    input_path = job_dir / f"input{ext}"
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    jobs[job_id] = {
        "status":"processing","stage":"upload","input_path":str(input_path),
        "pdf_path":str(job_dir/"output.pdf"),"xml_path":str(job_dir/"output.xml"),
        "wav_path":str(job_dir/"output.wav"),"melody_rh":None,"melody_lh":None,
        "spec":None,"duration":0.0,"similarity":0.0,
        "chroma_orig":None,"chroma_synth":None,"error":None,
        "key_name":None,"tempo":0.0,
    }
    background_tasks.add_task(process_audio_v71, job_id, input_path)
    return {"job_id":job_id,"status":"processing"}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs: raise HTTPException(404, detail="Job not found")
    j = jobs[job_id]
    si = STAGES.get(j.get("stage","upload"), STAGES["upload"])
    return {"job_id":job_id,"status":j["status"],"stage":j.get("stage","upload"),
            "stage_text":si["text"],"progress":si["progress"],
            "similarity":j.get("similarity",0.0),"error":j.get("error")}

@app.get("/melody/{job_id}")
async def get_melody(job_id: str):
    if job_id not in jobs: raise HTTPException(404, detail="Job not found")
    if jobs[job_id]["melody_rh"] is None: raise HTTPException(400, detail="Melody not ready")
    j = jobs[job_id]
    return JSONResponse({
        "melody_rh":j["melody_rh"],"melody_lh":j["melody_lh"],"spec":j["spec"],
        "duration":j["duration"],"similarity":j.get("similarity",0.0),
        "chroma_orig":j.get("chroma_orig"),"chroma_synth":j.get("chroma_synth"),
        "key_name":j.get("key_name"),"tempo":j.get("tempo",0.0),
    })

@app.post("/render/{job_id}")
async def render_pdf(job_id: str):
    if job_id not in jobs: raise HTTPException(404, detail="Job not found")
    if jobs[job_id]["melody_rh"] is None: raise HTTPException(400, detail="Melody not ready")
    xml_path = Path(jobs[job_id]["xml_path"])
    pdf_path = Path(jobs[job_id]["pdf_path"])
    wav_path = Path(jobs[job_id]["wav_path"])
    try:
        build_musicxml_v71(jobs[job_id]["melody_rh"], jobs[job_id]["melody_lh"], xml_path)
        generate_wav_v71(jobs[job_id]["melody_rh"], jobs[job_id]["melody_lh"], wav_path)
        mscore = find_musescore()
        if mscore:
            r = subprocess.run([mscore,"-o",str(pdf_path),str(xml_path)], capture_output=True, text=True, timeout=120)
            if r.returncode != 0: logger.warning(f"MuseScore: {r.stderr or r.stdout}")
        jobs[job_id]["status"] = "completed"
        return {"status":"completed","job_id":job_id}
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        raise HTTPException(500, detail=str(e))

@app.get("/download/{job_id}.pdf")
async def download_pdf(job_id: str):
    if job_id not in jobs: raise HTTPException(404, detail="Job not found")
    p = Path(jobs[job_id]["pdf_path"])
    if not p.exists(): raise HTTPException(404, detail="PDF not ready")
    return FileResponse(p, media_type="application/pdf", filename=f"PianoMagic_{job_id}.pdf")

@app.get("/download/{job_id}.wav")
async def download_wav(job_id: str):
    if job_id not in jobs: raise HTTPException(404, detail="Job not found")
    p = Path(jobs[job_id]["wav_path"])
    if not p.exists(): raise HTTPException(404, detail="WAV not ready")
    return FileResponse(p, media_type="audio/wav", filename=f"PianoMagic_{job_id}.wav")

# ============================================================
# Utilities
# ============================================================
def set_stage(job_id: str, stage: str):
    if job_id in jobs:
        jobs[job_id]["stage"] = stage
        logger.info(f"[{job_id}] Stage: {stage}")

def read_audio(path: Path) -> Tuple[np.ndarray, int]:
    if HAS_LIBROSA:
        y, sr = librosa.load(str(path), sr=SR, mono=True)
        return y, sr
    cmd = ['ffmpeg','-y','-i',str(path),'-ar',str(SR),'-ac','1','-f','f32le','-']
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0: raise RuntimeError(f"ffmpeg: {r.stderr.decode()[:200]}")
    return np.frombuffer(r.stdout, dtype=np.float32), SR

def midi_to_name(midi: int) -> Tuple[str, int, int]:
    name = NOTE_NAMES[midi % 12]
    octv = (midi // 12) - 1
    if '#' in name: return name[0], 1, octv
    return name, 0, octv

def dur_info(dur: int) -> Tuple[str, bool]:
    if dur >= 28: return 'whole', False
    if dur >= 20: return 'half', True
    if dur >= 12: return 'half', False
    if dur >= 10: return 'quarter', True
    if dur >= 6:  return 'quarter', False
    if dur >= 5:  return 'eighth', True
    if dur >= 3:  return 'eighth', False
    return '16th', False

def find_musescore() -> Optional[str]:
    for c in ["musescore3","musescore","mscore"]:
        if shutil.which(c): return c
    return None

def compute_spectrogram(audio: np.ndarray, sr: int) -> np.ndarray:
    if HAS_LIBROSA:
        D = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH))
        spec_db = librosa.amplitude_to_db(D, ref=np.max)
    else:
        frames=[]; w=np.hanning(N_FFT)
        for i in range(0, len(audio)-N_FFT, HOP_LENGTH):
            frames.append(np.abs(np.fft.rfft(audio[i:i+N_FFT]*w)))
        spec = np.array(frames).T if frames else np.zeros((N_FFT//2+1,1))
        spec_db = 20*np.log10(spec+1e-10)
    vmax = spec_db.max()
    spec_db = np.clip(spec_db, vmax-80, vmax)
    spec_db = (spec_db-spec_db.min())/(spec_db.max()-spec_db.min()+1e-10)
    fs = max(1, spec_db.shape[0]//128); ts = max(1, spec_db.shape[1]//200)
    spec_db = spec_db[::fs, ::ts]
    return (spec_db*255).astype(np.uint8)

def estimate_key(chroma: np.ndarray) -> Tuple[int, str, bool]:
    cm = np.mean(chroma, axis=1)
    if cm.sum() == 0: return 0, "C major", True
    cn = cm / (np.linalg.norm(cm)+1e-10)
    best, bk, bm = -1.0, 0, True
    for s in range(12):
        mp = np.roll(KS_MAJOR, s); mn = np.roll(KS_MINOR, s)
        mc = np.corrcoef(cn, mp/np.linalg.norm(mp))[0,1]
        nc = np.corrcoef(cn, mn/np.linalg.norm(mn))[0,1]
        if mc > best: best, bk, bm = mc, s, True
        if nc > best: best, bk, bm = nc, s, False
    kn = NOTE_NAMES[bk] + (" major" if bm else " minor")
    return bk, kn, bm

# ============================================================
# Note Segmentation v7.1 — Stricter, cleaner
# ============================================================
def segment_notes_v71(events: List[Dict], sr=SR, hop_length=HOP_LENGTH,
                     min_note_dur_ms=120, merge_gap_ms=80, pitch_tolerance=1) -> List[Dict]:
    if not events: return []
    # Adaptive magnitude threshold
    mags = np.array([e.get('mag',0.01) for e in events])
    mag_thr = max(0.004, np.median(mags)*0.15) if len(mags) else 0.005
    filtered = [e for e in events if e.get('mag',0) >= mag_thr]
    if not filtered: filtered = events
    filtered = sorted(filtered, key=lambda e:e['time'])

    MIN_DUR = min_note_dur_ms/1000.0
    MERGE = merge_gap_ms/1000.0

    merged=[]
    for e in filtered:
        if not merged:
            merged.append({'time':e['time'],'midi':e['midi'],'mag':e.get('mag',0.01),'dur':max(e.get('dur',MIN_DUR),MIN_DUR)})
            continue
        last = merged[-1]
        gap = e['time'] - (last['time']+last['dur'])
        pd = abs(e['midi']-last['midi'])
        if pd <= pitch_tolerance and gap < MERGE:
            ne = max(last['time']+last['dur'], e['time']+e.get('dur',MIN_DUR))
            last['dur'] = ne-last['time']
            last['mag'] = max(last['mag'], e.get('mag',0.01))
            if e.get('mag',0) > last['mag']*1.2: last['midi'] = e['midi']
        else:
            merged.append({'time':e['time'],'midi':e['midi'],'mag':e.get('mag',0.01),'dur':max(e.get('dur',MIN_DUR),MIN_DUR)})

    result = [m for m in merged if m['dur'] >= MIN_DUR]
    # Second pass: identical pitches tiny gap
    final=[]
    for m in result:
        if not final: final.append(m.copy()); continue
        last = final[-1]
        gap = m['time'] - (last['time']+last['dur'])
        if m['midi']==last['midi'] and gap < 0.03:
            ne = max(last['time']+last['dur'], m['time']+m['dur'])
            last['dur'] = ne-last['time']
            last['mag'] = max(last['mag'], m['mag'])
        else:
            final.append(m.copy())
    logger.info(f"[seg71] {len(events)} -> {len(filtered)} -> {len(merged)} -> {len(result)} -> {len(final)}")
    return final

# ============================================================
# Main Pipeline v7.1
# ============================================================
def process_audio_v71(job_id: str, input_path: Path):
    try:
        set_stage(job_id, "analyze")
        audio, sr = read_audio(input_path)
        duration = len(audio)/sr
        jobs[job_id]["duration"] = round(duration,2)
        jobs[job_id]["spec"] = compute_spectrogram(audio, sr).tolist()

        if not HAS_LIBROSA:
            set_stage(job_id, "melody")
            rh, lh = extract_melody_fallback(audio, sr)
            jobs[job_id]["melody_rh"], jobs[job_id]["melody_lh"] = rh, lh
            set_stage(job_id, "done")
            jobs[job_id]["status"] = "completed"
            return

        set_stage(job_id, "melody")
        md = extract_melody_librosa_v71(audio, sr)
        jobs[job_id]["tempo"] = round(md['tempo'],1)
        jobs[job_id]["key_name"] = md.get('key_name','C major')

        set_stage(job_id, "fit")
        rh, lh = fit_piano_arrangement_v71(md)
        jobs[job_id]["melody_rh"], jobs[job_id]["melody_lh"] = rh, lh

        set_stage(job_id, "render")
        build_musicxml_v71(rh, lh, Path(jobs[job_id]["xml_path"]))
        generate_wav_v71(rh, lh, Path(jobs[job_id]["wav_path"]))
        mscore = find_musescore()
        if mscore:
            subprocess.run([mscore,"-o",str(jobs[job_id]["pdf_path"]),str(jobs[job_id]["xml_path"])], capture_output=True, timeout=120)

        set_stage(job_id, "compare")
        sim, co, cs = compare_fingerprints_v71(audio, sr, Path(jobs[job_id]["wav_path"]))
        jobs[job_id]["similarity"] = round(sim,4)
        jobs[job_id]["chroma_orig"], jobs[job_id]["chroma_synth"] = co, cs

        set_stage(job_id, "done")
        jobs[job_id]["status"] = "completed"
        logger.info(f"[{job_id}] Done! sim={sim:.3f}")
    except Exception as e:
        logger.error(f"[{job_id}] Error: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)

# ============================================================
# Melody Extraction v7.1 — Clean, stable, strict
# ============================================================
def extract_melody_librosa_v71(audio: np.ndarray, sr: int) -> Dict:
    y_harm = librosa.effects.harmonic(audio, margin=4.0)

    # Tempo
    tempo = float(librosa.beat.beat_track(y=audio, sr=sr)[0])
    if tempo < 40 or tempo > 200:
        oe = librosa.onset.onset_strength(y=audio, sr=sr)
        of = librosa.onset.onset_detect(onset_envelope=oe, sr=sr)
        if len(of)>1: tempo = 60.0/(np.median(np.diff(of))*HOP_LENGTH/sr)
        tempo = max(40, min(200, tempo))
    beat_dur = 60.0/tempo

    # Strict onset detection
    onset_env = librosa.onset.onset_strength(y=y_harm, sr=sr, hop_length=HOP_LENGTH)
    onset_mean = np.mean(onset_env)
    onset_std = np.std(onset_env)
    onset_thr = onset_mean + 0.25 * onset_std

    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH,
        pre_max=3, post_max=3, pre_avg=4, post_avg=4,
        delta=max(0.07, onset_thr*0.08), wait=5
    )
    # Keep only strong onsets
    onset_frames = np.array([of for of in onset_frames if onset_env[of] > onset_thr*0.4])
    if len(onset_frames) == 0:
        # fallback: take top 20 peaks
        peaks = np.argsort(onset_env)[-20:]
        onset_frames = np.sort(peaks)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=HOP_LENGTH)

    # PYIN with longer window for stability
    f0_pyin, voiced_flag, _ = librosa.pyin(
        y_harm, sr=sr, hop_length=HOP_LENGTH,
        fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C8'),
        frame_length=2048
    )

    # PIPTRACK fallback
    pitches, mags = librosa.piptrack(
        y=y_harm, sr=sr, hop_length=HOP_LENGTH,
        fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C8')
    )

    events = []
    for i, of in enumerate(onset_frames):
        # Neighborhood stability check on PYIN
        nb = f0_pyin[max(0,of-2):min(len(f0_pyin),of+3)]
        valid = nb[~np.isnan(nb)]
        pitch = None
        if len(valid) >= 2:
            midi_vals = librosa.hz_to_midi(valid)
            if np.std(midi_vals) <= 2.0:
                pitch = float(np.median(valid))
        # Fallback piptrack
        if pitch is None or pitch <= 0:
            window = slice(of, min(of+8, pitches.shape[1]))
            bp, bm = 0, 0
            for t in range(window.start, window.stop):
                for f in range(pitches.shape[0]):
                    if mags[f,t] > bm and pitches[f,t] > 0:
                        bm = mags[f,t]; bp = pitches[f,t]
            if bp > 0 and bm > 0.004: pitch = bp

        if pitch and pitch > 0:
            midi = int(round(librosa.hz_to_midi(pitch)))
            midi = max(21, min(108, midi))
            # Octave continuity correction
            if events:
                pm = events[-1]['midi']
                while abs(midi - pm) > 10:
                    if midi > pm + 10: midi -= 12
                    elif midi < pm - 10: midi += 12
                    else: break
            end_t = onset_times[i+1] if i+1 < len(onset_times) else len(audio)/sr
            dur = min(end_t - onset_times[i], beat_dur * 2.5)
            dur = max(0.06, dur)
            events.append({'time':onset_times[i],'midi':midi,'mag':bm if 'bm' in dir() else 0.02,'dur':dur})

    # Temporal median smoothing on pitch contour
    if len(events) >= 5:
        try:
            from scipy.ndimage import median_filter
            midis = [e['midi'] for e in events]
            sm = median_filter(midis, size=3)
            for idx in range(len(events)): events[idx]['midi'] = int(sm[idx])
        except Exception: pass

    # Segment with stricter params
    final = segment_notes_v71(events, sr=sr, hop_length=HOP_LENGTH,
                                min_note_dur_ms=120, merge_gap_ms=80)

    # Salience pruning: drop weakest 20% if many notes
    if len(final) > 8:
        sal = [m['dur'] * m.get('mag',0.01) for m in final]
        thr = np.percentile(sal, 20)
        final = [m for m in final if m['dur']*m.get('mag',0.01) >= thr]
        logger.info(f"[prune] kept {len(final)} notes")

    # Key estimation
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=HOP_LENGTH)
    kidx, kname, ismaj = estimate_key(chroma)

    return {'melody':final,'tempo':tempo,'beat_dur':beat_dur,'key_idx':kidx,
            'key_name':kname,'is_major':ismaj,'duration':len(audio)/sr,'audio':audio,'sr':sr}

# ============================================================
# Arrangement v7.1 — Minimal, clean
# ============================================================
def fit_piano_arrangement_v71(melody_data: Dict) -> Tuple[List[Dict], List[Dict]]:
    melody = melody_data['melody']
    tempo = melody_data['tempo']
    beat_dur = melody_data['beat_dur']
    key_idx = melody_data['key_idx']
    is_major = melody_data.get('is_major', True)

    if not melody: return [], []

    rh = []
    scale_deg = [0,2,4,5,7,9,11] if is_major else [0,2,3,5,7,8,10]

    def is_diatonic(m): return (m%12 - key_idx)%12 in scale_deg
    def snap(m):
        pc = (m%12 - key_idx)%12
        if pc in scale_deg: return m
        ds = [(sd, min((pc-sd)%12,(sd-pc)%12)) for sd in scale_deg]
        ds.sort(key=lambda x:x[1])
        return m + (ds[0][0]-pc)

    # RH: clean melody, minimal harmony
    for i, e in enumerate(melody):
        p = max(21, min(108, e['midi']))
        if not is_diatonic(p):
            sp = snap(p)
            if abs(sp-p) <= 1: p = sp
        dur = max(0.08, min(e['dur'], beat_dur*2.2))
        vel = min(127, int(75 + e.get('mag',0.5)*45))
        if i%4==0: vel = min(127, vel+12)
        rh.append({'start':round(e['time'],3),'dur':round(dur*0.95,3),'pitch':p,'velocity':vel})

        # Add harmony ONLY on strong beats with long notes
        if i%4==0 and dur > beat_dur*0.8 and p < 90:
            deg = (p%12 - key_idx)%12
            interval = 4 if (is_major and deg in [0,5,7]) else 3 if (is_major and deg in [2,4,9]) else 7
            hp = snap(p+interval)
            if abs(hp-p) in [3,4,7] and is_diatonic(hp):
                rh.append({'start':round(e['time'],3),'dur':round(dur*0.25,3),'pitch':hp,'velocity':max(25,vel-40)})

    # LH: chord on measure boundaries only (every 4 beats)
    measure_starts = set()
    for e in melody:
        m = int(e['time'] // (beat_dur*4))
        measure_starts.add(m)

    lh = []
    for m in sorted(measure_starts):
        m_time = m * beat_dur * 4
        # Find melody note closest to this measure start
        closest = None
        for e in melody:
            if abs(e['time'] - m_time) < beat_dur*2:
                closest = e; break
        if closest is None: continue

        deg = (closest['midi']%12 - key_idx)%12
        if is_major:
            if deg in [0,5,7]: iv = [0,4,7]
            elif deg in [2,4,9]: iv = [0,3,7]
            else: iv = [0,4,7]
        else:
            if deg in [0,3,7,10]: iv = [0,3,7]
            elif deg in [2,5,8]: iv = [0,4,7]
            else: iv = [0,3,7]

        root = 36 + (key_idx + iv[0])%12
        while root < 28: root += 12
        while root > 48: root -= 12

        # Single bass note + chord stab on measure start
        lh.append({'start':round(m_time,3),'dur':round(beat_dur*1.5,3),'pitch':root,'velocity':38})
        for add in iv[1:]:
            pp = root + add
            if pp <= 72:
                lh.append({'start':round(m_time,3),'dur':round(beat_dur*1.2,3),'pitch':pp,'velocity':28})

    return rh, lh

# ============================================================
# Comparison v7.1
# ============================================================
def compare_fingerprints_v71(orig: np.ndarray, sr: int, wav_path: Path) -> Tuple[float, List[float], List[float]]:
    try:
        cand, _ = librosa.load(wav_path, sr=sr, mono=True)
        ml = min(len(orig), len(cand))
        o, c = orig[:ml], cand[:ml]

        c_o = librosa.feature.chroma_stft(y=o, sr=sr, hop_length=HOP_LENGTH)
        c_c = librosa.feature.chroma_stft(y=c, sr=sr, hop_length=HOP_LENGTH)
        mc = min(c_o.shape[1], c_c.shape[1])
        csims = []
        for i in range(mc):
            no, nc = np.linalg.norm(c_o[:,i])+1e-8, np.linalg.norm(c_c[:,i])+1e-8
            csims.append(np.dot(c_o[:,i],c_c[:,i])/(no*nc))
        csim = float(np.mean(csims))

        oo = librosa.onset.onset_strength(y=o, sr=sr, hop_length=HOP_LENGTH)
        oc = librosa.onset.onset_strength(y=c, sr=sr, hop_length=HOP_LENGTH)
        mo = min(len(oo), len(oc))
        o1, o2 = oo[:mo]/(np.max(oo)+1e-8), oc[:mo]/(np.max(oc)+1e-8)
        ocorr = float(np.corrcoef(o1,o2)[0,1])
        if np.isnan(ocorr): ocorr=0.0

        f0_o,_,_ = librosa.pyin(o, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C8'), sr=sr)
        f0_c,_,_ = librosa.pyin(c, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C8'), sr=sr)
        f0_o, f0_c = np.nan_to_num(f0_o,nan=0.0), np.nan_to_num(f0_c,nan=0.0)
        mf = min(len(f0_o), len(f0_c))
        valid = (f0_o[:mf]>0)&(f0_c[:mf]>0)
        if np.sum(valid)>10:
            m1 = librosa.hz_to_midi(f0_o[:mf][valid])
            m2 = librosa.hz_to_midi(f0_c[:mf][valid])
            diff = np.abs(m1-m2); diff = np.minimum(diff, 12-diff)
            psim = float(np.mean(np.exp(-diff/2.0)))
        else: psim=0.0

        mel_o = librosa.feature.melspectrogram(y=o, sr=sr, hop_length=HOP_LENGTH, n_mels=40)
        mel_c = librosa.feature.melspectrogram(y=c, sr=sr, hop_length=HOP_LENGTH, n_mels=40)
        mm = min(mel_o.shape[1], mel_c.shape[1])
        m1 = librosa.power_to_db(mel_o[:,:mm]+1e-8); m2 = librosa.power_to_db(mel_c[:,:mm]+1e-8)
        msim = max(0, 1-np.mean(np.abs(m1-m2))/80.0)

        sc_o = librosa.feature.spectral_contrast(y=o, sr=sr, hop_length=HOP_LENGTH)
        sc_c = librosa.feature.spectral_contrast(y=c, sr=sr, hop_length=HOP_LENGTH)
        msc = min(sc_o.shape[1], sc_c.shape[1])
        scc = float(np.corrcoef(sc_o[:,:msc].flatten(), sc_c[:,:msc].flatten())[0,1])
        if np.isnan(scc): scc=0.0
        ssim = max(0, (scc+1)/2)

        total = csim*0.25 + max(0,ocorr)*0.20 + psim*0.25 + msim*0.20 + ssim*0.10
        return min(1.0,max(0.0,total)), c_o.mean(axis=1).tolist(), c_c.mean(axis=1).tolist()
    except Exception as e:
        logger.warning(f"Compare failed: {e}")
        return 0.5, [0.0]*12, [0.0]*12

# ============================================================
# Fallback
# ============================================================
def extract_melody_fallback(audio: np.ndarray, sr: int) -> Tuple[List[Dict], List[Dict]]:
    N=4096; H=int(sr*0.05); freqs=np.fft.rfftfreq(N,1/sr); frames=[]
    for i in range(0, len(audio)-N, H):
        spec=np.abs(np.fft.rfft(audio[i:i+N]*np.hanning(N)))
        mask=(freqs>=130)&(freqs<=4200)
        bf, bs = freqs[mask], spec[mask]
        if len(bs)==0: continue
        idx=np.argmax(bs); pf, pa = bf[idx], bs[idx]
        gn=np.max(spec)+1e-10; an=min(1.0, pa/gn)
        if an<0.015: continue
        midi=int(round(69+12*np.log2(pf/440.0)))
        frames.append({'time':i/sr,'midi':midi,'amp':an})
    if not frames: return [],[]
    try:
        from scipy.ndimage import median_filter
        sm = median_filter([f['midi'] for f in frames], size=7)
        for i in range(len(frames)): frames[i]['midi']=int(sm[i])
    except: pass
    notes=[]; cur={'start':frames[0]['time'],'pitch':frames[0]['midi'],'amp':frames[0]['amp']}
    for f in frames[1:]:
        if f['midi']!=cur['pitch']:
            d=f['time']-cur['start']
            if d>=0.15: notes.append({'start':round(cur['start'],2),'dur':round(d,2),'pitch':cur['pitch'],'velocity':min(127,int(cur['amp']*127))})
            cur={'start':f['time'],'pitch':f['midi'],'amp':f['amp']}
    d=len(audio)/sr-cur['start']
    if d>=0.15: notes.append({'start':round(cur['start'],2),'dur':round(d,2),'pitch':cur['pitch'],'velocity':min(127,int(cur['amp']*127))})
    if not notes: return [],[]
    dd=[notes[0]]
    for n in notes[1:]:
        gap=n['start']-(dd[-1]['start']+dd[-1]['dur'])
        if n['pitch']==dd[-1]['pitch'] and gap<0.15:
            dd[-1]['dur']=round(n['start']+n['dur']-dd[-1]['start'],2)
            dd[-1]['velocity']=max(dd[-1]['velocity'],n['velocity'])
        elif n['pitch']!=dd[-1]['pitch']:
            dd.append(n)
    return [n for n in dd if n['pitch']>=60], [n for n in dd if n['pitch']<60]

# ============================================================
# Synthesis v7.1 — Stereo piano model
# ============================================================
def generate_wav_v71(rh: List[Dict], lh: List[Dict], wav_path: Path, sr=44100):
    all_notes = rh+lh
    if not all_notes: return
    td = max(n["start"]+n["dur"] for n in all_notes)
    ts = int(td*sr)+int(sr*0.5)
    L = np.zeros(ts, dtype=np.float64)
    R = np.zeros(ts, dtype=np.float64)
    B=0.0003
    for n in all_notes:
        f = 440.0*(2.0**((n["pitch"]-69)/12.0))
        ss=int(n["start"]*sr); ds=int(n["dur"]*sr)
        if ds<=0: continue
        t=np.linspace(0,n["dur"],ds,endpoint=False)
        w=np.zeros(ds, dtype=np.float64)
        for hn, ha in [(1,0.55),(2,0.28),(3,0.14),(4,0.07),(5,0.04),(6,0.02),(7,0.015)]:
            hf=hn*f*np.sqrt(1+B*hn**2)
            w += ha*np.sin(2*np.pi*hf*t)
        att=int(0.006*sr); dec=int(0.10*sr); rel=int(0.15*sr); sus=0.55
        env=np.ones(ds)*sus
        if att>0: env[:att]=np.linspace(0,1,att)
        if dec>0 and att+dec<ds: env[att:att+dec]=np.linspace(1,sus,dec)
        if rel>0 and ds>rel: env[-rel:]*=np.linspace(1,0,rel)
        w*=env
        pan = 0.35 if n in rh else -0.35
        vel = n.get("velocity",80)/127.0
        w *= vel*0.35
        end=min(ss+ds,ts); act=end-ss
        L[ss:end] += w[:act]*(0.5-pan/2)
        R[ss:end] += w[:act]*(0.5+pan/2)
    # Light reverb
    decay=np.exp(-np.linspace(0,5,int(0.12*sr))); decay/=decay.sum()
    L=np.convolve(L,decay,mode='same'); R=np.convolve(R,decay,mode='same')
    stereo=np.stack([L,R],axis=1)
    peak=np.max(np.abs(stereo))
    if peak>0: stereo=stereo/peak*0.95
    i16=(stereo*32767).astype(np.int16)
    with open(wav_path,'wb') as f:
        f.write(b'RIFF'); f.write(struct.pack('<I',36+ts*4))
        f.write(b'WAVEfmt '); f.write(struct.pack('<IHHIIHH',16,1,2,sr,sr*4,4,16))
        f.write(b'data'); f.write(struct.pack('<I',ts*4)); f.write(i16.tobytes())

# ============================================================
# MusicXML v7.1
# ============================================================
def build_musicxml_v71(rh: List[Dict], lh: List[Dict], xml_path: Path):
    DIV=8
    root=Element('score-partwise',{'version':'3.1'})
    pl=SubElement(root,'part-list')
    sp=SubElement(pl,'score-part',{'id':'P1'})
    SubElement(sp,'part-name').text='Piano'
    p1=SubElement(root,'part',{'id':'P1'})

    def group(notes):
        m={}
        for n in notes:
            mm=int(n["start"]//4.0); off=n["start"]%4.0
            if off>=3.999: mm+=1; off=0.0
            m.setdefault(mm,[]).append({"offset":int(round(off*DIV)),"dur":int(round(n["dur"]*DIV)),"pitch":n["pitch"]})
        return m

    mr, ml = group(rh), group(lh)
    am = sorted(set(list(mr.keys())+list(ml.keys())))

    for mi in am:
        me=SubElement(p1,'measure',{'number':str(mi+1)})
        if mi==0:
            at=SubElement(me,'attributes')
            SubElement(at,'divisions').text=str(DIV)
            SubElement(at, 'staves').text = '2'
            c1=SubElement(at,'clef',{'number':'1'}); SubElement(c1,'sign').text='G'; SubElement(c1,'line').text='2'
            c2=SubElement(at,'clef',{'number':'2'}); SubElement(c2,'sign').text='F'; SubElement(c2,'line').text='4'
            SubElement(SubElement(at,'key'),'fifths').text='0'
            tm=SubElement(at,'time'); SubElement(tm,'beats').text='4'; SubElement(tm,'beat-type').text='4'

        def write(events, staff_num):
            events=sorted(events,key=lambda x:x['offset'])
            le=0
            for ev in events:
                off,dur,pitch=ev['offset'],ev['dur'],ev['pitch']
                gap=off-le
                if gap>=1:
                    r=SubElement(me,'note'); SubElement(r,'rest'); SubElement(r,'duration').text=str(gap)
                    SubElement(r,'staff').text=str(staff_num)
                    t,d=dur_info(gap); SubElement(r,'type').text=t
                    if d: SubElement(r,'dot')
                s,a,o=midi_to_name(pitch)
                n=SubElement(me,'note'); pc=SubElement(n,'pitch')
                SubElement(pc,'step').text=s
                if a!=0: SubElement(pc,'alter').text=str(a)
                SubElement(pc,'octave').text=str(o)
                SubElement(n,'duration').text=str(dur)
                SubElement(n,'staff').text=str(staff_num)
                t,d=dur_info(dur); SubElement(n,'type').text=t
                if d: SubElement(n,'dot')
                le=off+dur
            fill=32-le
            if fill>=1:
                r=SubElement(me,'note'); SubElement(r,'rest'); SubElement(r,'duration').text=str(fill)
                SubElement(r,'staff').text=str(staff_num)
                t,d=dur_info(fill); SubElement(r,'type').text=t
                if d: SubElement(r,'dot')

        write(mr.get(mi,[]),1)
        write(ml.get(mi,[]),2)

    with open(xml_path,'w',encoding='utf-8') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n'+tostring(root,encoding='unicode'))

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
