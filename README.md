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

Camera access requires HTTPS or localhost. A deployed HTTPS URL can request camera and microphone permissions; an HTTP container URL generally cannot. Uploaded MP4 files work without camera permissions.

If Community Cloud reports `ModuleNotFoundError: autotab`, confirm the app's **Main file path** is `app.py` at the repository root, then reboot the app so it reinstalls the updated `requirements.txt`. Do not select `src/autotab/ui/app.py` as the entrypoint; that file is an internal UI module.

The editor and example work without model weights. The model downloader retrieves the official version-1 [MediaPipe Hand Landmarker model](https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/hand-landmarker.ts). The adapter uses the [documented video-mode API](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python). Model weights retain their upstream licensing.

## Workflow

1. Enter BPM, select guitar or bass, enter any capo fret (0 means none), and enter the video timestamp of the first beat. Beat 0 in the editor corresponds to this timestamp. The default grid has four subdivisions per beat.
2. In **1. Add a video**, choose **Upload video** and select an MP4 (also MOV/AVI/MKV/WebM, up to 200 MB), or choose **Record with camera** → **Enable camera + microphone** → allow both permissions → **Start recording** → **Stop recording**. Camera clips are capped at 2 minutes / 24 MB and become downloadable MP4s. Input is visible even before entering BPM. The sidebar's JSON uploader is only for saved tab projects. Keep the fretting hand, outer strings, and fret wires visible; record the instrument's sound too.
3. Calibrate the visible neck by clicking four points directly on the preview, then **Apply four points**. Follow the perimeter in this order: **near/highest string → far/highest string → far/lowest string → near/lowest string**. Near means closer to the nut, independent of camera orientation. Wire 0 is the nut or capo; a crop from relative wire 3 to wire 12 is valid too. Use the applied grid to check alignment; manual coordinates remain under **Fine-adjust coordinates**. With a capo, all calibration and tab fret numbers are relative to it.
4. Choose **Audio + video**, then analyze. Audio determines pitch and attacks. The calibrated box now follows small fretboard/camera movement frame by frame using visible neck texture; if a frame is too blurred or blank, it keeps the last reliable box. General hand placement on the neck narrows down the fingerings that produce that exact pitch; precise fingertip estimates only help choose among nearby alternatives. A conflicting visual guess never overrides an audio pitch. A position inferred only from the hand region stays labeled `audio` for review. For silent videos, explicitly select **Vision only**.
5. Edit string, fret, beat, and duration in the table. Add or delete rows, including open strings or notes the detector missed. The graphical tab underneath uses numbered **4/4 bars**, with rhythm symbols below the strings: whole, half, quarter, eighth, sixteenth, and thirty-second notes, including dots, flags/beams, and barline ties. Duration 1 means a quarter note, 2 a half, and 4 a whole. Unusual durations show their beat length instead of rounding it. Download the graphical tab as SVG, or expand the plain-text version. Your changes export as `manual`. The example button lets you try this before recording anything.
6. In **Check notes against the recording**, click a row to replay a precise source segment and see the proposed fret position. Flagged notes explain missing/conflicting evidence and offer same-pitch alternative positions. You can apply an alternative or mark a note reviewed.
7. In **Play your tab**, directly below the tab, press **Play tab** to hear a simple synth play the current notes at your BPM. **Whole tab** skips setup silence before the first note; **Selected section** lets you choose a beat range. Enable **Repeat playback** to loop, or **Download synth audio** to save a WAV. Note, tempo, instrument, and capo edits automatically update the sound. **Prepare recording loop** creates the matching source-video section for comparison. Each player has its own controls; synthesized sound is a pitch/rhythm reference. Selections are limited to 120 seconds; use sections for longer scores.
8. Use **Undo/Redo** to reverse valid edits (up to 100 snapshots per session). Valid scores automatically save to `.autotab/drafts/`; after reopening, choose **Recover an autosaved draft** in the sidebar. Videos and undo history are not part of disk recovery. Download JSON for a portable project, or export text/CSV. Re-upload the original video or download your camera clip before leaving if you want to review it later.

Camera capture requests webcam and microphone access directly; it does not request screen sharing. Open the app in Chrome or Edge at `http://localhost:8501` if an embedded browser does not expose camera access. The browser requires localhost or HTTPS for [camera/microphone access](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia). This is record-then-analyze, not live transcription; stop recording before calibrating/analyzing.

String 1 is high e on guitar and G on bass. BPM changes preserve existing beat positions. Re-analyze the video to quantize source timestamps at a different tempo. Text exports preserve every listed note onset, add an exact column for off-grid edits, and retain the same string/fret assignment as the editor. JSON/CSV also retain durations. When two rapid attacks quantize to the same string and grid cell, the second keeps its measured beat instead of being discarded.

## Code structure

```text
app.py                         Thin development entry point
src/autotab/
  domain.py                    Validated notes, tuning, calibration, project
  ports.py                     Frame source / contact detector interfaces
  transcription.py             Temporal contact filtering, beat quantization
  fusion.py                    Audio/visual cross-matching and string selection
  pipeline.py                  Application orchestration through interfaces
  recording.py                 File-based service composing media/model adapters
  storage.py                   Versioned, portable JSON serialization
  tablature.py                 One-to-one note-list to tab-cell layout
  score.py                     Numbered bars and vector rhythm notation
  export.py                    Text tablature and note CSV
  history.py                   Bounded immutable undo/redo snapshots
  autosave.py                  Atomic per-session local recovery files
  review.py                    Uncertainty, alternate positions, replay timing
  playback.py                  Beat-range reference audio synthesis
  cli.py                       Local launcher and explicit model download
  adapters/
    video.py                   OpenCV decoding and sampling
    geometry.py                Perspective transform and physical fret spacing
    motion.py                  Conservative optical-flow neck tracking
    hand_detector.py           MediaPipe model integration
    audio.py                   FFmpeg decoding, YIN pitch, attack detection
    clips.py                   Precise fractional-second video replay segments
    camera.py                  Browser recording validation and MP4 conversion
  ui/
    app.py                     Streamlit widgets and session lifecycle
    editor.py                  Independently testable table validation
    recording_input.py         Always-visible file upload and camera recording
    calibration.py             Click-to-calibrate plus coordinate fallback
    review.py                  Selectable notes, replay, alternatives, looping
    session.py                 History, widget resets, and autosave coordination
    *_component/index.html     Bundled calibration and camera browser controls
tests/                         Unit, synthetic signal, and integration tests
docs/architecture.md           Contracts, algorithms, limitations, extension plan
```

The domain, fusion, persistence, and temporal filtering use only the Python standard library. Optional ML/video/UI imports stay at the edges. The pretrained model is replaceable through `ContactDetector`; it does not own the note format. File headers and function docstrings document assumptions and units.

## Checks

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
```

Synthetic signals test pitch and timing, including bass low E. Fake detectors test fusion without model weights. UI tests exercise project rendering and session behavior; table tests cover editing. Real codec tests generate a video locally. The model smoke test runs when weights exist and skips otherwise. These checks verify behavior; they do not establish transcription accuracy on real performances. Follow the [recording test checklist](docs/testing-your-recordings.md) to evaluate your own playing.

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
