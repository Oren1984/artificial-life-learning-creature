"""Generates the demo's background music: a quiet ambient pad, synthesized from scratch.

    python make_music.py <seconds> <out.wav>

Nothing is sampled or downloaded. The output is fully determined by this
script (fixed random seed), so the track is original to this repository and
carries no third-party licence.
"""
from __future__ import annotations

import sys
import wave

import numpy as np

RATE = 44_100
CHORD_SECONDS = 8.0
# A minor 9 -> F major 7 -> C major 7 -> G 6, as MIDI notes (bass first).
CHORDS = [
    (45, 57, 60, 64, 71),
    (41, 57, 60, 64, 69),
    (48, 55, 59, 64, 67),
    (43, 55, 59, 62, 64),
]


def hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def pad(notes, n: int, pan_rng: np.random.Generator) -> np.ndarray:
    """One chord: detuned sines with a slow swell, slightly different per channel."""
    t = np.arange(n) / RATE
    out = np.zeros((n, 2))
    for i, note in enumerate(notes):
        level = 0.9 if i == 0 else 0.55
        for ch in (0, 1):
            detune = 1.0 + (0.0018 if ch else -0.0018) * (1 + i % 2)
            f = hz(note) * detune
            phase = pan_rng.uniform(0, 2 * np.pi)
            tone = np.sin(2 * np.pi * f * t + phase) + 0.18 * np.sin(4 * np.pi * f * t + phase)
            out[:, ch] += level * tone
    swell = np.sin(np.pi * np.clip(t / (n / RATE), 0, 1)) ** 1.5  # rise and fall across the chord
    return out * swell[:, None]


def bell(note: int, n: int) -> np.ndarray:
    t = np.arange(n) / RATE
    f = hz(note)
    return (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2.01 * f * t)) * np.exp(-t * 2.2)


def render(seconds: float) -> np.ndarray:
    rng = np.random.default_rng(7)
    total = int(seconds * RATE)
    chord_n = int(CHORD_SECONDS * RATE)
    hop = chord_n // 2  # chords overlap by half, so swells cross-fade
    mix = np.zeros((total + chord_n * 2, 2))

    k = 0
    while k * hop < total:
        mix[k * hop : k * hop + chord_n] += pad(CHORDS[(k // 2) % len(CHORDS)], chord_n, rng)
        k += 1

    # Sparse, soft bell notes from the current chord, with one quiet echo.
    bell_n = int(3.0 * RATE)
    at = 4.0
    while at < seconds - 6.0:
        chord = CHORDS[int(at // CHORD_SECONDS) % len(CHORDS)]
        note = int(rng.choice(chord[2:])) + 12
        tone = bell(note, bell_n) * 0.5
        pan = rng.uniform(0.25, 0.75)
        for delay, gain in ((0.0, 1.0), (0.42, 0.35)):
            i = int((at + delay) * RATE)
            mix[i : i + bell_n, 0] += tone * gain * (1 - pan)
            mix[i : i + bell_n, 1] += tone * gain * pan
        at += rng.uniform(2.5, 5.0)

    mix = mix[:total]
    t = np.arange(total) / RATE
    fade = np.clip(t / 2.5, 0, 1) * np.clip((seconds - t) / 4.0, 0, 1)
    mix *= fade[:, None]
    return mix / np.abs(mix).max() * 0.5  # peak at about -6 dBFS; the video mixes it lower


def main() -> None:
    seconds, out = float(sys.argv[1]), sys.argv[2]
    pcm = (render(seconds) * 32767).astype("<i2")
    with wave.open(out, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())


if __name__ == "__main__":
    main()
