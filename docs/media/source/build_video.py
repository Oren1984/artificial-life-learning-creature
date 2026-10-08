"""Assembles the demo video from a recording made by record-demo.mjs.

    python build_video.py <work dir>

Reads <work dir>/frames.ffconcat, captions.srt and timing.json; writes
learning-creature-demo.mp4, demo-poster.jpg, captions.srt and recording.json
into docs/media/. Requires ffmpeg on PATH.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

MEDIA = Path(__file__).resolve().parents[1]
MUSIC_VOLUME = 0.35


def run(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def main() -> None:
    work = Path(sys.argv[1]).resolve()
    timing = json.loads((work / "timing.json").read_text(encoding="utf-8"))
    seconds = timing["duration"]
    video = MEDIA / "learning-creature-demo.mp4"

    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("make_music.py")), f"{seconds:.3f}", str(work / "music.wav")],
        check=True,
    )
    run(
        "-f", "concat", "-safe", "0", "-i", str(work / "frames.ffconcat"),
        "-i", str(work / "music.wav"),
        "-vf", "fps=30,scale=in_range=full:out_range=tv,format=yuv420p",
        "-color_range", "tv",
        "-af", f"volume={MUSIC_VOLUME}",
        "-c:v", "libx264", "-preset", "slow", "-crf", "21",
        "-c:a", "aac", "-b:a", "128k",
        "-t", f"{seconds:.3f}",
        "-movflags", "+faststart",
        "-metadata", "title=Artificial Life - The Learning Creature (demo)",
        str(video),
    )  # fmt: skip

    srt = (work / "captions.srt").read_text(encoding="utf-8")
    (MEDIA / "captions.srt").write_text(srt, encoding="utf-8", newline="\n")
    (MEDIA / "recording.json").write_text(json.dumps(timing, indent=2) + "\n", encoding="utf-8", newline="\n")

    # Poster: the before/after scene, two seconds into its second caption.
    cue = re.search(r"(\d+):(\d+):(\d+),(\d+) --> [^\n]+\nSame creature\.", srt)
    h, m, s, ms = map(int, cue.groups())
    at = h * 3600 + m * 60 + s + ms / 1000 + 2.0
    run("-ss", f"{at:.3f}", "-i", str(video), "-frames:v", "1", "-vf", "scale=1280:-2", "-q:v", "3", str(MEDIA / "demo-poster.jpg"))
    shutil.rmtree(work / "__pycache__", ignore_errors=True)
    print(f"wrote {video} ({video.stat().st_size / 1e6:.1f} MB, {seconds:.1f} s)")


if __name__ == "__main__":
    main()
