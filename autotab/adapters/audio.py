"""Local, monophonic audio analysis for guitar and bass video soundtracks.

This adapter decodes audio through FFmpeg, estimates fundamental frequency
using YIN's cumulative normalized difference, and splits stable pitch runs
at spectral/energy attacks. It does not estimate tempo: seconds are converted
to beats later using the BPM supplied by the user.

The baseline is intentionally explainable and independent of the hand model.
It works best on isolated single-note playing. Chords, backing tracks, effects,
and weak bass fundamentals can produce octave errors or missed notes. Quality
is a periodicity score, not a calibrated probability of musical correctness.
"""

import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np

from autotab.domain import INSTRUMENTS, AudioNote

SAMPLE_RATE = 22050
HOP = 256


def decode_audio(path: Path, max_seconds: float) -> np.ndarray:
    """Decode the first bounded part of a video's audio to mono float PCM.

    FFmpeg is provided by imageio-ffmpeg, so a system installation is optional.
    The command uses an argument list (no shell) and a timeout. Missing audio
    raises ValueError instead of silently substituting visual-only transcription.
    """
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-v",
        "error",
        "-i",
        str(path),
        "-t",
        str(max_seconds),
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-af",
        "aresample=async=1:first_pts=0",
        "-f",
        "f32le",
        "pipe:1",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            timeout=max(60, max_seconds * 2),
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"Audio decoding failed: {exc}") from exc
    if result.returncode or not result.stdout:
        detail = result.stderr.decode(errors="replace")[-500:]
        raise ValueError(
            f"No usable audio track. Choose vision-only mode for a silent video. {detail}"
        )
    return np.frombuffer(result.stdout, dtype="<f4").copy()


def estimate_pitch(
    samples: np.ndarray, minimum_hz: float, maximum_hz: float
) -> tuple[float, float]:
    """Return (frequency in Hz, periodicity quality) for one audio window.

    The YIN difference function compares a fixed half-window with delayed copies,
    avoiding the long-lag bias of shrinking-overlap autocorrelation. FFT correlation
    and cumulative energies make this inexpensive. The first convincing local
    minimum favors the fundamental over its longer-period subharmonics.
    """
    samples = samples.astype(np.float64) - np.mean(samples)
    half = len(samples) // 2
    maximum_lag = min(half - 1, int(SAMPLE_RATE / minimum_hz) + 1)
    minimum_lag = max(2, int(SAMPLE_RATE / maximum_hz))
    size = 1 << (len(samples) * 2 - 1).bit_length()
    reference = samples[:half]
    correlation = np.fft.irfft(
        np.fft.rfft(samples, size) * np.conj(np.fft.rfft(reference, size)),
        size,
    )[: maximum_lag + 1]
    energy = np.concatenate(([0.0], np.cumsum(samples * samples)))
    lags = np.arange(maximum_lag + 1)
    difference = np.maximum(0, energy[half] + energy[lags + half] - energy[lags] - 2 * correlation)
    normalized = np.ones_like(difference)
    normalized[1:] = (
        difference[1:]
        * np.arange(1, len(difference))
        / np.maximum(
            np.cumsum(difference[1:]),
            1e-12,
        )
    )
    chosen = None
    for lag in range(minimum_lag, maximum_lag):
        if normalized[lag] < 0.18:
            while lag + 1 <= maximum_lag and normalized[lag + 1] < normalized[lag]:
                lag += 1
            chosen = lag
            break
    if chosen is None:
        return 0.0, 0.0
    refined = float(chosen)
    if 1 <= chosen < maximum_lag:
        left, center, right = normalized[chosen - 1 : chosen + 2]
        denominator = left - 2 * center + right
        if abs(denominator) > 1e-12:
            refined += float(0.5 * (left - right) / denominator)
    return SAMPLE_RATE / refined, float(np.clip(1 - normalized[chosen], 0, 1))


def analyze_audio(
    samples: np.ndarray, instrument: str = "guitar", capo: int = 0
) -> tuple[AudioNote, ...]:
    """Extract stable single-pitch events and repeated attacks in video seconds.

    Bass uses a longer analysis window so low E has enough periods for detection.
    Three-frame median filtering rejects isolated pitch glitches. Short silence
    holes are filled only between equal pitches; longer gaps remain real rests.
    Spectral flux plus an energy-rise threshold identifies re-picked held notes.
    """
    tuning = INSTRUMENTS[instrument].open_midi
    window = 4096 if instrument == "bass" else 2048
    if len(samples) < window:
        return ()
    minimum_hz = 440 * 2 ** ((min(tuning) + capo - 69 - 0.5) / 12)
    maximum_hz = 440 * 2 ** ((max(tuning) + 24 - 69 + 0.5) / 12)
    pitches, qualities, energies, fluxes = [], [], [], []
    previous_spectrum = np.zeros(window // 2 + 1)
    for start in range(0, len(samples) - window + 1, HOP):
        frame = samples[start : start + window]
        rms = float(np.sqrt(np.mean(frame * frame)))
        spectrum = np.abs(np.fft.rfft(frame * np.hanning(window)))
        fluxes.append(float(np.sum(np.maximum(spectrum - previous_spectrum, 0))))
        previous_spectrum = spectrum
        energies.append(rms)
        hz, quality = estimate_pitch(frame, minimum_hz, maximum_hz) if rms >= 0.003 else (0, 0)
        pitches.append(round(69 + 12 * np.log2(hz / 440)) if hz else -1)
        qualities.append(quality)
    filtered = np.asarray(pitches)
    original = filtered.copy()
    for i in range(1, len(filtered) - 1):
        filtered[i] = int(np.median(original[i - 1 : i + 2]))
    # Find attacks with a local, level-independent energy comparison.
    attacks: set[int] = set()
    previous_attack = -1000
    for i in range(2, len(filtered) - 2):
        baseline = float(np.median(fluxes[max(0, i - 12) : i]))
        old_energy = energies[max(0, i - 4)]
        if (
            fluxes[i] >= max(fluxes[i - 1], fluxes[i + 1])
            and fluxes[i] > max(0.5, baseline * 2.5)
            and energies[i] > max(0.004, old_energy * 1.15)
            and (i - previous_attack) * HOP / SAMPLE_RATE >= 0.09
        ):
            attacks.add(i)
            previous_attack = i
    events: list[AudioNote] = []
    begin = 0
    for end in range(1, len(filtered) + 1):
        changed = end == len(filtered) or filtered[end] != filtered[begin]
        attacked = end in attacks and (end - begin) * HOP / SAMPLE_RATE >= 0.09
        if not changed and not attacked:
            continue
        duration = (end - begin) * HOP / SAMPLE_RATE
        if filtered[begin] >= 0 and duration >= 0.055:
            # Window starts approximate attack times; tails include half a window.
            start_seconds = begin * HOP / SAMPLE_RATE
            end_seconds = min(
                len(samples) / SAMPLE_RATE, end * HOP / SAMPLE_RATE + window / SAMPLE_RATE / 2
            )
            events.append(
                AudioNote(
                    int(filtered[begin]),
                    start_seconds,
                    end_seconds,
                    float(np.mean(qualities[begin:end])),
                )
            )
        begin = end
    return tuple(events)
