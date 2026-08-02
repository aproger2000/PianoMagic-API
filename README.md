# PianoMagic 7.0

**AI-powered audio-to-piano-score transcription**

Upload any melody audio — get a professionally arranged piano score for two hands.

## What's New in 7.0

- **Deep Note Segmentation** (`segment_notes_v7`): adaptive magnitude thresholding, intelligent merging of similar notes (±1 semitone, gap < 150 ms), minimum duration filter (> 100 ms)
- **Dual Pitch Tracking**: PYIN primary + piptrack fallback with cross-validation
- **Krumhansl-Schmuckler Key Estimation**: proper major/minor key detection
- **Diatonic Piano Arrangement**: voice-leading aware RH melody + Alberti-bass LH accompaniment
- **Harmonic Piano Synthesis**: inharmonic string model, stereo field, light reverb
- **Multi-feature Fingerprinting**: chroma + onset + pitch + mel + spectral contrast
- **iOS Upload Fix**: explicit MIME types + opacity-based invisible input
- **Real-time Staff Visualization**: animated playhead, color-coded hands

## Architecture

```
Audio Input
    ↓
Harmonic-Percussive Separation (librosa.effects.harmonic)
    ↓
Multi-scale Onset Detection (librosa.onset.onset_detect)
    ↓
Dual Pitch Tracking (PYIN + piptrack)
    ↓
segment_notes_v7() — adaptive threshold, merge, filter
    ↓
Key Estimation (Krumhansl-Schmuckler)
    ↓
Adaptive Piano Arrangement (diatonic voicing)
    ↓
Synthesis (inharmonic piano model → stereo WAV)
    ↓
Fingerprint Comparison (5 features)
    ↓
MusicXML + PDF (MuseScore)
```

## Deployment

### Backend (Render / VPS)

```bash
pip install -r requirements.txt
# Ensure ffmpeg and MuseScore are installed
uvicorn backend_api:app --host 0.0.0.0 --port 8000
```

### Frontend (GitHub Pages)

Upload `index.html`, `app.js`, `styles.css` to your `gh-pages` branch.

Update `API_BASE` in `app.js` to point to your backend.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check + version |
| POST | `/analyze` | Upload audio file |
| GET | `/status/{job_id}` | Poll processing status |
| GET | `/melody/{job_id}` | Get melody + metadata |
| POST | `/render/{job_id}` | Render PDF (optional) |
| GET | `/download/{job_id}.pdf` | Download PDF |
| GET | `/download/{job_id}.wav` | Download WAV |

## File Structure

```
pianomagic-v7/
├── backend_api.py      # FastAPI backend
├── index.html          # Frontend entry
├── app.js              # Frontend logic
├── styles.css          # Frontend styles
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## License

MIT
