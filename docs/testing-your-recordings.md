# Testing your own guitar or bass recordings

You can test through the app once installation is complete. Under **1. Add a video**, upload an MP4 **with its soundtrack embedded**, or choose **Record with camera**, enable camera/microphone, start, and stop. Separate WAV/MP3 uploads are not part of this UI. Files stay local. Start with a 10–20 second clip so calibration and corrections are quick. The sidebar JSON upload is for tab projects, not videos. Upload and camera controls are visible before entering BPM.

## First guitar test

1. Use standard tuning. Choose a known BPM, such as 80, and play clean, separate notes on one string. A metronome heard only through headphones avoids adding clicks to the recording.
2. Keep the camera and fretboard stationary, well lit, and in focus. Show the fingertips, both outer strings, and at least two known fret wires. Avoid an angle that hides the fingertips behind the neck.
3. Play a simple known sequence, for example low E string frets `0, 3, 5, 3`, one note per beat. Then repeat fret 3 twice while keeping the finger down. End with a second string to check position selection.
4. Enter your BPM and the timestamp of the first note/beat. Select the instrument and capo position. With a capo, all tab and calibration fret numbers are **relative to the capo**.
5. Click the four calibration corners on the preview and choose **Apply four points**. Inspect several preview times. If the overlaid wires drift away from the neck, the clip is unsuitable for static calibration.
6. Run Audio + video. Compare the resulting sequence and beat positions to what you played. Audio-only labels are expected for open strings and occluded contacts; they do not mean the acoustic pitch is necessarily wrong.
7. Correct a fret, change a duration, add a missing note, and delete a false positive. Check Undo/Redo. Click a note in the review table to replay its segment and try a same-pitch alternative. Use a beat range to loop reference tab playback and prepare the matching source-video loop.
8. Reopen the app and recover the autosaved draft from the sidebar. Export JSON, reopen it, and confirm the corrections remain. Export the text tab and check alignment. Download camera recordings as MP4 before closing the session: autosaves contain tabs and calibration, not videos.

For camera testing, open `http://localhost:8501` in Chrome or Edge and allow both camera and microphone access when prompted. Confirm that the live preview appears before recording, that the captured video has sound, and that the browser camera indicator turns off after stopping. Test denied permission and retry. Try switching camera devices if more than one is connected. Hardware capture has not been exercised by the automated tests; those use mocked recorder controls and generated WebM clips.

## Additional cases

| Case | What to verify |
| --- | --- |
| Same pitch in two positions | Play high e open, then B string fret 5. The acoustic pitch should match; visible evidence should help select the fretted position. |
| Repeated picking | Pick a held note twice. Audio should split the attacks even when the hand does not move. |
| Brief silence | Pause between notes. The draft should leave a beat gap. |
| Capo at fret 2 | Repeat a known shape, entering capo 2. Pitch rises two semitones while tab fret numbers stay relative to the capo. |
| Four-string bass | Select bass, then play E string frets 0, 3, 5 slowly. Check the low E pitch and four-line export. |
| Silent video | Select Vision only explicitly. Expect finger-state guesses and missing repeated attacks. |
| Chords/backing music | Treat this as a limitation test; reliable polyphonic transcription is not implemented. |

## Useful feedback for the next iteration

Keep the original recording and the exported project. Write down instrument, tuning, capo, BPM, first-beat timestamp, expected string/fret sequence, and the video timestamps of mistakes. Separate wrong pitch, wrong string, missed attack, duplicate note, and timing drift. A short clip with an exact expected sequence is more useful than a long recording with a general accuracy estimate.

Automated checks can run independently:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

The synthetic and black-video fixtures are regression tests, not a real-performance benchmark. No accuracy percentage should be inferred from their pass rate.
