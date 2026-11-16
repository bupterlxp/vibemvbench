"""Decode delivered MP4s: probe, audio track, downsampled frames with per-frame features."""
import json
import subprocess

import numpy as np

from .. import FFMPEG, FFPROBE


def probe(path):
    out = subprocess.run([FFPROBE, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
                         capture_output=True, text=True)
    if out.returncode != 0:
        return None
    return json.loads(out.stdout)


def read_audio(path, sr=22050):
    out = subprocess.run([FFMPEG, "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                         capture_output=True)
    return np.frombuffer(out.stdout, dtype=np.float32), sr


def read_frames(path, fps=10, width=256):
    """Return (times, frames[H,W,3] uint8) sampled at `fps`."""
    info = probe(path)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    w, h = int(v["width"]), int(v["height"])
    nh = int(round(h * width / w / 2) * 2)
    out = subprocess.run([FFMPEG, "-v", "error", "-i", str(path), "-vf", f"fps={fps},scale={width}:{nh}",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
    fr = np.frombuffer(out.stdout, dtype=np.uint8)
    n = len(fr) // (width * nh * 3)
    fr = fr[: n * width * nh * 3].reshape(n, nh, width, 3)
    return (np.arange(n) + 0.5) / fps, fr
