# Architecture and development guide

## Design priorities

The project prioritizes code structure, explicit contracts, and editable results. It separates musical meaning from UI widgets and external ML libraries. The first detector is a replaceable baseline, not an assumption baked into every part of the application.

The dependency direction is:

```text
Streamlit UI → recording service → adapters + pipeline
                                  pipeline → tracker + fusion → domain
             editor/storage/export ───────────────────────────→ domain
```

The domain has no third-party imports. The application pipeline accepts structural protocols for frames and visual contacts. `recording.py` owns concrete adapter creation and resource cleanup, keeping model construction out of the UI. Numerical processing lives in adapters because it depends on NumPy, OpenCV, and the audio decoder.

## Musical and timing contracts

- String numbers are one-based, highest pitch first. Guitar: e/B/G/D/A/E; bass: G/D/A/E.
- Fret numbers are relative to the nut or entered capo. Capo 2 plus tab fret 3 means physical fret 5. Physical fret 24 is the current upper bound. Partial capos are not supported.
- BPM is required user input. A beat is a quarter note for the grid descriptions; no tempo or time-signature inference runs.
- Video/audio observations use absolute seconds from the recording origin. Project notes use beats relative to `beat_offset_seconds`.
- `beat = (seconds - offset) × BPM / 60`. Nearest-grid rounding uses half-up semantics; durations are at least one grid cell. Notes ending before the offset are excluded, and notes crossing it are clipped.
- All imported/manual notes are validated. Positive durations, finite timing, legal strings/frets, supported tuning, and capo bounds are enforced. UI row errors prevent exporting invalid state.
- `manual`, `matched`, `audio`, and `vision` are provenance labels, not confidence probabilities. A cross-match can still be wrong if both components are wrong.

## Analysis stages

1. **Decode audio:** FFmpeg extracts a bounded mono PCM track at 22,050 Hz. Audio timestamp padding preserves a delayed track's origin. An absent track is an error in audiovisual mode, not permission to change modes.
2. **Estimate acoustic events:** YIN-style fixed-window difference estimates the fundamental frequency. Guitar uses 2,048 samples and bass 4,096. A 256-sample hop, short median filter, spectral flux, and energy attack detector split stable pitches and repeated plucks. This is conventional signal processing; the current neural model is in the visual branch.
3. **Decode video:** OpenCV samples RGB frames using presentation timestamps where available. If timestamps are unavailable, nominal FPS gives a monotonic fallback. Decode length and frame sampling are explicit settings.
4. **Track the neck:** The initial four-point calibration seeds texture points inside the quadrilateral. Sparse Lucas–Kanade flow and a RANSAC homography update all four corners before each detector frame. New corners are blended with the prior estimate; low-texture frames keep the last good calibration and reseed later. This handles small camera/neck motion while avoiding a jump from a bad frame.
5. **Infer hand landmarks:** The pretrained MediaPipe Hand Landmarker supplies fingertips. For two detected hands, the one with the most fingertips on the currently tracked neck is selected. No fixed left/right fretting-hand assumption is made.
6. **Map fingertips and contacts:** Four tracked corners define a homography. Physical wire locations use `1 - 2^(-fret/12)` along the neck. Each landmark retains a continuous string position and fret-cell position, including points that lie between string thresholds. A stricter view also emits discrete contacts for temporal tracking. Capo-relative geometry works because shifting scale length cancels when normalizing the two reference wires.
7. **Stabilize contacts:** Require repeated observations, bridge brief occlusions, and let the highest visible fret on a string mask lower ones. This produces intervals of evidence, not picked-note claims.
8. **Cross-match:** Enumerate every string/fret that produces an acoustic MIDI pitch under the selected tuning and capo. Estimate general hand placement by taking the median fingertip fret per frame, then the median across frames around the attack (100 ms before to at most 250 ms after, bounded by the next onset). First retain candidates whose distance to this region is within one fret of the nearest candidate's distance. Continuous fingertip/contact scores then distinguish these nearby alternatives; continuity and low fret break remaining ties. Precise contact guesses cannot override the broader hand region. Region-only or ambiguous choices stay labeled `audio` for review. Open strings remain pitch candidates, but cannot be visually confirmed by absence of a finger.
9. **Quantize and edit:** Use supplied BPM and beat offset. If two rapid automatic events on one string round to the same grid cell, the later event keeps its measured beat instead of being deleted. The editor permits additions, deletions, and timing corrections. Text layout maps every note index to exactly one cell, preserves off-grid beats, and rejects only impossible same-string/same-beat conflicts. JSON/CSV preserve exact edited timing and durations.

## Persistence and UI state

`score.py` provides a standalone SVG score with numbered 4/4 bars and rhythm below
the strings. It draws vector noteheads, stems, flags/beams, and dots without a music
font dependency. Barline intersections retain their source note indices and are
displayed as tied continuations, preserving the original synth timing. Same-onset
chords share a rhythm symbol when their durations agree; different durations occupy
separate lanes. Nonstandard durations are labeled in beats, never rounded for display.
The UI embeds a base64 SVG image because Streamlit sanitizes inline SVG out of HTML,
and allows horizontal scrolling for dense bars. Downloaded SVG is standalone.

JSON is a complete, versioned score plus calibration and timing settings. Videos and model weights are external. Loading applies domain validation, and missing optional version-1 fields use defaults. Future breaking changes should use migrations and a new schema version.

The UI keeps an original editor seed stable across reruns; Streamlit retains row/cell deltas against that seed. A loaded or newly analyzed project gets a new editor revision. Analysis builds a replacement Project and only publishes it after success, so a decoder/model error does not destroy the current draft. Valid edits enter a bounded 100-snapshot undo history and atomically autosave to a session-specific UUID file under `.autotab/drafts/`. New edits after undo discard the redo branch. Disk failure preserves the last valid recovery file and is surfaced in the UI. Recovery starts a new session; undo history and media bytes are not persisted. JSON downloads remain the portable save format.

## Recording and review UI boundaries

Video intake renders before BPM validation. The JSON project loader is separate from the main MP4/video uploader. A bundled browser component records webcam + microphone with `getUserMedia`/`MediaRecorder`, requesting permissions only on a user click, releasing hardware at stop, and limiting clips to two minutes/24 MB. The Python boundary validates base64 messages and converts browser WebM/MP4 to seekable MP4 because browser WebM may lack duration metadata. Captured and uploaded videos expose the same bytes/name/identity contract. Recording is batch input, not live transcription.

A second bundled component collects four normalized calibration clicks, validated in Python before applying. Review reasons are optional backward-compatible note metadata. Alternatives preserve pitch; applying one enters manual provenance and the same undo/autosave flow. The review table shows the derived pitch name, selects source-video segments, draws attack-local model fingertips, and projects the proposed position onto the calibrated image. Fractional-second clips are transcoded locally rather than using Streamlit's integer-rounded media seek bounds. Reference playback synthesizes the selected beat range, respects tuning/capo and overlapping durations, and normalizes the mix. Native players own looping; audio/video comparison uses separate controls.

## Extending the first pass

- **Better visual contact inference:** Implement `ContactDetector.detect(VideoFrame)`. Keep contact outputs in relative string/fret coordinates. Add a dedicated hand/neck contact model or improve fingertip pressure inference without changing notes/storage.
- **Neck tracking:** Replace static calibration with per-frame transforms in the visual adapter. Keep the persisted user calibration as the initial frame reference.
- **Polyphonic transcription:** Replace the audio estimator with a model emitting overlapping `AudioNote` observations. Fusion currently selects positions independently; chord-aware assignment will also need global constraints so simultaneous pitches do not conflict on one string.
- **Alternate tunings:** Extend instrument metadata and persistence to store explicit open MIDI pitches rather than adding special cases in fusion. The present two named tunings deliberately keep the first release small.
- **Evaluation before training:** Collect consented, labeled clips and measure onset/pitch accuracy, exact string/fret accuracy, unmatched rate, and edit effort separately. Split evaluation by player and recording setup to avoid a misleading train/test overlap.

## Test layers

`pytest` covers domain validation, JSON compatibility, contact state transitions, quantization, attack-local fusion, ambiguous evidence, the standard-tuning G-string 9–7–5 regression, capo transposition, synthetic harmonic tones including bass low E, re-picking, perspective/rotated calibration, actual FFmpeg/OpenCV decoding, exact tab-cell mapping, and headless Streamlit rendering. The real model smoke test runs when local weights exist and otherwise skips without a download.

The full-service smoke fixture is a black video with a known tone. It verifies inference and composition but intentionally produces **zero visual matches**. It does not prove hand tracking works on a real fretboard. The manual recording checklist in `testing-your-recordings.md` is the next evaluation step.
