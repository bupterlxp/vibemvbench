"""Reference annotations for a song: beats/downbeats (beat_this), sections (SSM novelty
snapped to downbeats), lyrics lines/words (JamendoLyrics human annotations)."""
import csv
import json
import subprocess
from pathlib import Path

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from . import FFMPEG

_F2B = None


def to_wav(src, dst, sr=44100):
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([FFMPEG, "-loglevel", "error", "-y", "-i", str(src), "-ac", "2", "-ar", str(sr), str(dst)], check=True)


def beats(wav, device="mps"):
    global _F2B
    if _F2B is None:
        from beat_this.inference import File2Beats
        _F2B = File2Beats(checkpoint_path="final0", device=device, dbn=False)
    b, d = _F2B(str(wav))
    return np.asarray(b, float), np.asarray(d, float)


def sections(y, sr, beat_times, downbeats, min_len=8.0, kernel_beats=16):
    """Foote checkerboard novelty on a beat-synchronous MFCC+chroma SSM; peaks snapped to downbeats."""
    frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=512)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=512)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
    feat = np.vstack([librosa.util.normalize(librosa.util.sync(mfcc, frames), axis=0),
                      librosa.util.normalize(librosa.util.sync(chroma, frames), axis=0)])
    feat = feat / (np.linalg.norm(feat, axis=0, keepdims=True) + 1e-9)
    ssm = feat.T @ feat
    k = kernel_beats
    g = np.outer(np.r_[-np.ones(k), np.ones(k)], np.r_[-np.ones(k), np.ones(k)])
    g *= np.outer(np.hanning(2 * k), np.hanning(2 * k))
    pad = np.pad(ssm, k, mode="edge")
    nov = np.array([np.sum(pad[i:i + 2 * k, i:i + 2 * k] * g) for i in range(ssm.shape[0])])
    nov = gaussian_filter1d(np.maximum(nov, 0), 1.5)
    bt = np.r_[0.0, beat_times]  # sync() prepends a segment starting at frame 0
    bt = bt[: len(nov)]
    beat_period = np.median(np.diff(beat_times)) if len(beat_times) > 1 else 0.5
    peaks, _ = find_peaks(nov, distance=max(1, int(min_len / beat_period)), prominence=np.percentile(nov, 75) * 0.5)
    bounds = []
    for p in peaks:
        t = bt[p]
        if len(downbeats):
            t = float(downbeats[np.argmin(np.abs(downbeats - t))])
        if t > min_len / 2 and (not bounds or t - bounds[-1] >= min_len):
            bounds.append(t)
    return bounds, nov.tolist()


def jamendo_lyrics(root, stem):
    lines, words = [], []
    lp = Path(root) / "annotations" / "lines" / f"{stem}.csv"
    wp = Path(root) / "annotations" / "words" / f"{stem}.csv"
    if lp.exists():
        for r in csv.DictReader(open(lp, encoding="utf-8")):
            lines.append({"start": float(r["start_time"]), "end": float(r["end_time"]), "text": r["lyrics_line"].strip()})
    txt = Path(root) / "lyrics" / f"{stem}.words.txt"
    toks = txt.read_text(encoding="utf-8").split() if txt.exists() else []
    if wp.exists():
        rows = list(csv.DictReader(open(wp, encoding="utf-8")))
        for i, r in enumerate(rows):
            words.append({"start": float(r["word_start"]), "end": float(r["word_end"]), "text": toks[i] if i < len(toks) else ""})
    return lines, words


def annotate(wav, out_json, lyrics_root=None, stem=None):
    y, sr = librosa.load(str(wav), sr=22050, mono=True)
    b, d = beats(wav)
    bounds, _ = sections(y, sr, b, d)
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    lines, words = jamendo_lyrics(lyrics_root, stem) if lyrics_root else ([], [])
    ann = {
        "duration": float(len(y) / sr),
        "bpm": float(60 / np.median(np.diff(b))) if len(b) > 1 else None,
        "beats": np.round(b, 3).tolist(),
        "downbeats": np.round(d, 3).tolist(),
        "section_boundaries": [round(x, 3) for x in bounds],
        "lyrics_lines": lines,
        "lyrics_words": words,
        "rms_hop_s": 512 / sr,
        "rms": np.round(rms, 4).tolist(),
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(ann, open(out_json, "w"), ensure_ascii=False)
    return ann
