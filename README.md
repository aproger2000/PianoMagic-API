# PianoMagic 7.1 — Clean Melody Engine

## What's New in 7.1

- **Stricter Onset Detection**: adaptive threshold, wait=5 frames, local maxima only
- **Pitch Stability**: neighborhood std check on PYIN, octave continuity correction
- **Temporal Smoothing**: median filter on pitch contour
- **Note Pruning**: salience-based dropping of weakest 20% notes
- **Minimal LH**: chords only on measure boundaries (every 4 beats)
- **RH Harmony**: only on long stable notes, not everywhere
- **Web Audio Synthesizer**: browser playback with stereo panning
- **iOS Upload Fix**: explicit MIME types + opacity input

## Files

- `backend_api.py` — FastAPI backend
- `index.html` — Frontend
- `app.js` — Frontend logic + Web Audio synth
- `styles.css` — Dark theme UI
- `requirements.txt` — Python deps

## Deploy

Backend: `uvicorn backend_api:app --host 0.0.0.0 --port 8000`
Frontend: static host (GitHub Pages)
