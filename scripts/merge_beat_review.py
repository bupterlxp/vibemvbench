"""Merge beat-review exports into corrected reference beats for the flagged windows.

Verdicts per window: ok (keep), double (reference twice too dense: keep the better half), half (too sparse: insert midpoints),
shift (right tempo, off by half a beat), fix (annotator's taps: fit a steady grid to the taps, then snap each grid beat to the
nearest onset within 70 ms), skip (no clear beat: window excluded from beat-scale scoring).
Writes data/beat_review_corrections.json and prints the share of flagged windows corrected (the paper's figure).

  python scripts/merge_beat_review.py beat_review_<name>.json [...]
"""
import json, sys
import numpy as np
import librosa

items = {i["id"]: i for i in json.load(open("beat_review/items.json"))}


def onsets(it):
    y, sr = librosa.load("beat_review/" + it["audio"], sr=22050, mono=True)
    env = librosa.onset.onset_strength(y=y, sr=sr)
    return librosa.times_like(env, sr=sr), env


def strength(times, t_env, env):
    return float(np.mean([env[np.argmin(np.abs(t_env - t))] for t in times])) if len(times) else 0.0


def corrected(it, verdict, taps):
    ref = np.asarray(it["reference"], float); lo, hi = it["win"]
    t_env, env = onsets(it)
    if verdict == "ok" or verdict == "skip":
        return ref
    if verdict == "double":
        a, b = ref[0::2], ref[1::2]
        return a if strength(a, t_env, env) >= strength(b, t_env, env) else b
    if verdict == "half":
        mids = (ref[:-1] + ref[1:]) / 2
        return np.sort(np.concatenate([ref, mids]))
    if verdict == "shift":
        p = np.median(np.diff(ref)) if len(ref) > 1 else 0.5
        return ref + p / 2 if strength(ref + p / 2, t_env, env) >= strength(ref - p / 2, t_env, env) else ref - p / 2
    # fix: steady grid fitted to taps, snapped to onsets
    taps = np.asarray(taps, float)
    if len(taps) < 3:
        return ref
    period = float(np.median(np.diff(taps)))
    k = np.round((taps - taps[0]) / period)
    slope, icept = np.polyfit(k, taps, 1)
    grid = icept + slope * np.arange(np.floor((lo - icept) / slope), np.ceil((hi - icept) / slope) + 1)
    grid = grid[(grid >= lo) & (grid <= hi)]
    peaks = t_env[librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.1, wait=3)]
    out = []
    for g in grid:
        d = np.abs(peaks - g)
        out.append(float(peaks[np.argmin(d)]) if len(d) and d.min() <= 0.07 else float(g))
    return np.asarray(out)


res = {}
for f in sys.argv[1:]:
    for r in json.load(open(f))["results"]:
        res.setdefault(r["id"], r)
out, v = {}, []
for wid, r in res.items():
    it = items[wid]; v.append(r["verdict"])
    beats = corrected(it, r["verdict"], r.get("taps_s_in_snippet", []))
    out[wid] = {"task": it["task"], "verdict": r["verdict"], "clip_window": [it["win"][0] + it["clip_t0"], it["win"][1] + it["clip_t0"]],
                "beats_in_clip": [round(b + it["clip_t0"], 3) for b in beats]}
json.dump(out, open("data/beat_review_corrections.json", "w"), indent=1)
n = len(items); done = len(res); changed = sum(x not in ("ok", "skip") for x in v)
print(f"reviewed {done}/{n}: ok {v.count('ok')}, double {v.count('double')}, half {v.count('half')}, shift {v.count('shift')}, "
      f"retapped {v.count('fix')}, no clear beat {v.count('skip')}")
print(f"share of flagged windows corrected: {changed / max(1, done):.0%}")
