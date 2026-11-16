"""Vibe MV-Bench: evaluating coding agents on code-driven music video generation."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin"
FFMPEG = str(BIN / "ffmpeg")
FFPROBE = str(BIN / "ffprobe")
DATA = ROOT / "data"
TASKS_DIR = ROOT / os.environ.get("VMB_TASKS", "tasks_free")


def load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
