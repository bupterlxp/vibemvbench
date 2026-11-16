"""Reference beats v2: an ensemble of independent trackers + an agreement score that decides what needs human review.

Candidates per song: beat_this final0/final1/final2 (different training seeds) and librosa (DP tracker with a tempo
prior set to the neural consensus tempo). For each 10 s window we measure pairwise agreement (F-measure allowing
octave equivalence: x2, x0.5 grids) and onset salience of each candidate. The window's reference is the candidate with
the highest salience among those that agree with at least one other; windows with no agreement are flagged
`needs_review`. Human edits made in tools/beat_editor.html override everything (data/annotations_human/).
"""
import json
from pathlib import Path

import librosa
import numpy as np

from . import DATA
from .eval.listening import f_measure

CKPTS = ("final0", "final1", "final2")
_TRK = {}
WIN = 10.0


def _trackers(device):
    if not _TRK:
        from beat_this.inference import File2Beats
        for c in CKPTS:
            _TRK[c] = File2Beats(checkpoint_path=c, device=device, dbn=False)
    return _TRK


def octave_f(ref, est, tol=0.07):
    """Best F over the estimate, its double-time and half-time versions (metrical-level agnostic)."""
    est = np.asarray(est)
    if len(est) < 2:
        return f_measure(ref, est, tol)
    mids = (est[:-1] + est[1:]) / 2
    cands = [est, np.sort(np.r_[est, mids]), est[::2], est[1::2]]
    return max(f_measure(ref, c, tol) for c in cands)


def salience(beats, onset_env, sr, hop):
    if len(beats) == 0:
        return 0.0
    fr = np.clip(librosa.time_to_frames(beats, sr=sr, hop_length=hop), 0, len(onset_env) - 1)
    return float(onset_env[fr].mean() / (onset_env.mean() + 1e-9))


def ensemble_beats(wav, device="mps"):
    y, sr = librosa.load(str(wav), sr=22050, mono=True)
    hop = 256
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    cand = {}
    for c, f in _trackers(device).items():
        b, d = f(str(wav)); cand[c] = (np.asarray(b), np.asarray(d))
    tempos = [60 / np.median(np.diff(b)) for b, _ in cand.values() if len(b) > 2]
    t0 = float(np.median(tempos)) if tempos else 120.0
    _, lb = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop, units="time", start_bpm=t0, tightness=200)
    cand["librosa"] = (np.asarray(lb), np.array([]))
    dur = len(y) / sr
    windows, chosen, chosen_down = [], [], []
    for s in np.arange(0, dur, WIN):
        e = s + WIN
        seg = {k: v[0][(v[0] >= s) & (v[0] < e)] for k, v in cand.items()}
        keys = list(seg)
        agree = {k: max([octave_f(seg[k], seg[j]) for j in keys if j != k] or [0]) for k in keys}
        sal = {k: salience(seg[k], oenv, sr, hop) for k in keys}
        ok = [k for k in keys if agree[k] >= 0.8 and len(seg[k]) >= 2]
        if ok:
            best = max(ok, key=lambda k: sal[k])
        else:
            best = max(keys, key=lambda k: sal[k])
        has_beats = any(len(v) >= 2 for v in seg.values())
        windows.append({"start": round(float(s), 2), "end": round(float(min(e, dur)), 2), "chosen": best,
                        "agreement": round(float(agree[best]), 3), "salience": {k: round(v, 2) for k, v in sal.items()},
                        "needs_review": bool(has_beats and (not ok or agree[best] < 0.9))})
        chosen.append(seg[best])
        if best in CKPTS:
            d = cand[best][1]; chosen_down.append(d[(d >= s) & (d < e)])
    beats = np.sort(np.concatenate(chosen)) if chosen else np.array([])
    beats = beats[np.r_[True, np.diff(beats) > 0.05]] if len(beats) else beats
    downs = np.sort(np.concatenate(chosen_down)) if chosen_down else np.array([])
    return beats, downs, windows, {k: v[0].tolist() for k, v in cand.items()}


def annotate_v2(song_id, device="mps"):
    out = DATA / "annotations_v2" / f"{song_id}.json"
    if out.exists():
        return json.load(open(out))
    old = json.load(open(DATA / "annotations" / f"{song_id}.json"))
    b, d, win, cand = ensemble_beats(DATA / "audio" / f"{song_id}.wav", device)
    ann = dict(old)
    ann.update({"beats": np.round(b, 3).tolist(), "downbeats": np.round(d, 3).tolist(),
                "bpm": float(60 / np.median(np.diff(b))) if len(b) > 2 else None,
                "beat_windows": win, "beat_candidates": cand,
                "needs_review_frac": round(float(np.mean([w["needs_review"] for w in win])), 3),
                "version": "v2-ensemble"})
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(ann, open(out, "w"), ensure_ascii=False)
    return ann
