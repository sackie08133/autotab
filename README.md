# AutoTab

A Python prototype that **listens to a performance, watches the fretboard, and creates editable tablature**. Six-string guitar is the primary workflow; standard four-string bass is also supported. You supply the BPM. The app never estimates tempo.

This is an intentionally modular first pass, not a trained end-to-end transcription system. A pretrained hand model supplies landmarks; an audio pitch/onset estimator and geometric fretboard mapping cross-check them. Best results come from clean, isolated, single-note playing with a stationary camera and neck.

## Run locally

Python 3.11 or newer is required. This build was tested on Windows with Python 3.14.3. Python 3.12 or 3.13 is also a reasonable choice if a dependency has no wheel for your platform. From this directory in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[app,dev]"
.\.venv\Scripts\autotab.exe download-model
.\.venv\Scripts\autotab.exe run
```

On macOS/Linux use `.venv/bin/python` and `.venv/bin/autotab`. Alternatively, activate the environment and run `streamlit run app.py`. Dependencies and the hand model need an initial network download; video analysis then runs locally. FFmpeg is included through `imageio-ffmpeg`. No API key or hosted AI service is required.

## Deploy

This is a Streamlit app. For Streamlit Community Cloud, push this entire project folder to a GitHub repository, choose the root `app.py` as the entrypoint, and let the service install `requirements.txt`. The requirements file installs the local `src/autotab` package as well as its runtime libraries. If the repository contains only the package folder, use `autotab/app.py` after copying the new package-level wrapper. The model weights are downloaded automatically when the first analysis starts, so they do not need to be committed. For a container host, build the included `Dockerfile` and expose port 8501:

```powershell
docker build -t autotab .
docker run --rm -p 8501:8501 autotab
```

## Known limits and next steps

- Audio is **monophonic**. Chords, overlapping sustained notes, distortion, backing tracks, and bends can confuse it. There is no trained polyphonic audio model yet.
- Vision infers contact from fingertip proximity. It cannot reliably see string pressure, barres, occluded fingers, or finger rolls. Handedness is not used to hardcode a fretting hand.
- Bass uses standard G–D–A–E tuning and a longer pitch-analysis window. Full capos from fret 0–12 are supported with relative tab numbering. Five-string bass, alternate tunings, partial capos, and microtonal techniques are not implemented.
- The first calibration is manual, then optical flow tracks small neck/camera movement. Large movement, severe blur, or an obstructed neck can still require recalibration. A/V tracks must already be synchronized; a small tolerance only covers frame sampling.
- Open-string pitches can be detected from audio, but vision cannot confirm an open string merely from the absence of a visible finger. Such notes remain audio-only.
- Automatic durations are estimates. In vision-only mode they describe finger dwell, not audible sustain; repeated picking is invisible there.
- The graphical score currently uses 4/4. Other time signatures, tuplets, and full multi-voice engraving are not implemented; unusual durations retain explicit beat-length labels. Correct the Duration column when the estimated sustain differs from the intended musical note value.
- No representative guitar/bass dataset is bundled. Real-world accuracy is unmeasured. A useful next step is annotated clips with onset, pitch, string, and fret labels, followed by accuracy measurement before model training.

See [architecture notes](docs/architecture.md) for extension points and an evaluation plan.
