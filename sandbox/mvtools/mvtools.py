#!/usr/bin/env python3
"""mvtools: music analysis helpers for the Vibe MV-Bench 'Tools' condition.

  mvtools beats    AUDIO   -> {"bpm", "beats": [s], "downbeats": [s]}      (beat_this, CPU)
  mvtools sections AUDIO   -> {"boundaries": [s]}                          (self-similarity novelty, snapped to downbeats)
  mvtools energy   AUDIO   -> {"hop_s", "rms": [...], "onsets": [s]}       (librosa)
  mvtools all      AUDIO   -> all of the above in one JSON
Output is JSON on stdout. Times are seconds from the start of AUDIO.
"""
import json
import subprocess
import sys
import tempfile

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def _wav(path):
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-ac", "2", "-ar", "44100", f], check=True)
    return f


def beats(path):
    from beat_this.inference import File2Beats
    b, d = File2Beats(checkpoint_path="final0", device="cpu", dbn=False)(_wav(path))
    b, d = np.asarray(b), np.asarray(d)
    return {"bpm": round(float(60 / np.median(np.diff(b))), 2) if len(b) > 1 else None,
            "beats": np.round(b, 3).tolist(), "downbeats": np.round(d, 3).tolist()}


def sections(path, bt=None):
    y, sr = librosa.load(path, sr=22050, mono=True)
    bt = bt or beats(path)
    b, d = np.array(bt["beats"]), np.array(bt["downbeats"])
    fr = librosa.time_to_frames(b, sr=sr, hop_length=512)
    mf = librosa.util.sync(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=512), fr)
    ch = librosa.util.sync(librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512), fr)
    X = np.vstack([librosa.util.normalize(mf, axis=0), librosa.util.normalize(ch, axis=0)])
    X /= np.linalg.norm(X, axis=0, keepdims=True) + 1e-9
    S = X.T @ X; k = min(16, max(2, S.shape[0] // 4))
    g = np.outer(np.r_[-np.ones(k), np.ones(k)], np.r_[-np.ones(k), np.ones(k)]) * np.outer(np.hanning(2 * k), np.hanning(2 * k))
    P = np.pad(S, k, mode="edge")
    nov = gaussian_filter1d(np.maximum([np.sum(P[i:i + 2 * k, i:i + 2 * k] * g) for i in range(S.shape[0])], 0), 1.5)
    bt0 = np.r_[0.0, b][: len(nov)]
    per = np.median(np.diff(b)) if len(b) > 1 else 0.5
    pk, _ = find_peaks(nov, distance=max(1, int(8 / per)), prominence=np.percentile(nov, 75) * 0.5)
    out = []
    for p in pk:
        t = float(d[np.argmin(np.abs(d - bt0[p]))]) if len(d) else float(bt0[p])
        if t > 4 and (not out or t - out[-1] >= 8):
            out.append(round(t, 3))
    return {"boundaries": out}


def energy(path):
    y, sr = librosa.load(path, sr=22050, mono=True)
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    on = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=False)
    return {"hop_s": 512 / sr, "rms": np.round(rms, 4).tolist(), "onsets": np.round(on, 3).tolist()}


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("beats", "sections", "energy", "all"):
        print(__doc__); sys.exit(1)
    cmd, path = sys.argv[1:]
    if cmd == "beats": r = beats(path)
    elif cmd == "sections": r = sections(path)
    elif cmd == "energy": r = energy(path)
    else:
        bt = beats(path); r = dict(bt); r.update(sections(path, bt)); r.update(energy(path))
    json.dump(r, sys.stdout)


if __name__ == "__main__":
    main()
