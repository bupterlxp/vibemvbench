"""Objective audio-visual sync metrics against reference annotations of the clip.

Every event metric is reported raw and chance-normalised: the same score computed against
the beat grid phase-shifted by random offsets (uniform within one beat period) gives the
expectation for a video whose events are unrelated to the music; norm = (raw - chance) / (1 - chance).
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import spearmanr

from .media import read_frames

CUT_TOL = 0.07      # s, MIREX-style beat tolerance for hard cuts
ACCENT_TOL = 0.10   # s, motion/flash accents (one frame at 30 fps is 33 ms)
N_SHIFTS = 64


def frame_signals(frames):
    f = frames.astype(np.float32) / 255.0
    lum = f.mean(axis=(1, 2, 3))
    diff = np.r_[0.0, np.abs(np.diff(f, axis=0)).mean(axis=(1, 2, 3))]
    hists = np.stack([np.histogramdd(x.reshape(-1, 3), bins=(6, 6, 6), range=[(0, 1)] * 3)[0].ravel() for x in f])
    hists /= hists.sum(1, keepdims=True)
    hd = np.r_[0.0, 0.5 * np.abs(np.diff(hists, axis=0)).sum(1)]  # total-variation distance, 0..1
    thumbs = np.stack([x[::8, ::8].ravel() for x in f])
    return {"lum": lum, "diff": diff, "hist_dist": hd, "hists": hists, "thumbs": thumbs}


def detect_cuts(t, sig):
    """Hard cuts: histogram-distance spikes far above local level (also catches full-frame flashes/scene swaps)."""
    hd = sig["hist_dist"]
    if len(hd) < 5:
        return np.array([])
    local = gaussian_filter1d(hd, 5)
    thr = np.maximum(0.25, local + 3 * np.std(hd))
    k, _ = find_peaks(hd, height=thr, distance=3)
    return t[k] - 0.5 * (t[1] - t[0])  # change happens between frame k-1 and k


def detect_accents(t, sig, min_gap=0.15):
    """Visual accents: peaks of the positive derivative of frame-difference energy (onset-like)."""
    d = gaussian_filter1d(sig["diff"], 0.7)
    flux = np.maximum(0, np.r_[0.0, np.diff(d)])
    if flux.max() <= 0:
        return np.array([])
    k, _ = find_peaks(flux, height=flux.mean() + 1.0 * flux.std(), distance=max(1, int(min_gap / (t[1] - t[0]))))
    return t[k]


def hit_rate(events, grid, tol):
    if len(events) == 0 or len(grid) == 0:
        return np.nan
    return float(np.mean(np.min(np.abs(events[:, None] - grid[None, :]), axis=1) <= tol))


def coverage(grid, events, tol):
    if len(grid) == 0:
        return np.nan
    if len(events) == 0:
        return 0.0
    return float(np.mean(np.min(np.abs(grid[:, None] - events[None, :]), axis=1) <= tol))


def shifted_grids(grid, period, rng):
    for o in rng.uniform(0.25 * period, 0.75 * period, N_SHIFTS):
        yield grid + o


def normed(raw, chance):
    if raw is None or np.isnan(raw) or np.isnan(chance):
        return np.nan
    return float((raw - chance) / (1 - chance + 1e-9))


def event_metrics(events, beats, downbeats, tol, prefix, seed=0):
    rng = np.random.default_rng(seed)
    period = float(np.median(np.diff(beats))) if len(beats) > 1 else 0.5
    out = {f"{prefix}_n": int(len(events)), f"{prefix}_none": int(len(events) == 0)}
    p = hit_rate(events, beats, tol)
    pc = np.nanmean([hit_rate(events, g, tol) for g in shifted_grids(beats, period, rng)]) if len(events) else np.nan
    out[f"{prefix}_on_beat"] = p
    out[f"{prefix}_on_beat_norm"] = normed(p, pc)
    if len(downbeats):
        out[f"{prefix}_on_downbeat"] = hit_rate(events, downbeats, tol)
    c = coverage(beats, events, tol)
    cc = np.nanmean([coverage(g, events, tol) for g in shifted_grids(beats, period, rng)])
    out[f"{prefix}_beat_coverage"] = c
    out[f"{prefix}_beat_coverage_norm"] = normed(c, cc)
    if not np.isnan(p) and not np.isnan(c) and p + c > 0:
        out[f"{prefix}_f1"] = 2 * p * c / (p + c)
    return out


def energy_corr(t, sig, rms, hop, win=0.5, detrend=1.0, local_win=0.1):
    """Loudness vs visual activity (frame-diff or brightness, whichever couples better), Spearman.
    global: smoothed envelopes over the whole clip (dominated by slow build-ups / fade-outs);
    local : both envelopes lightly smoothed (`local_win`) and high-passed by subtracting a `detrend`-second average,
            i.e. beat-scale coupling, insensitive to deliberate section-level contrasts (quiet music, busy picture)."""
    if len(rms) < 4:
        return np.nan, np.nan
    at = np.arange(len(rms)) * hop
    dt = t[1] - t[0]
    s = int(max(1, round(win / dt))); S = int(max(2, round(detrend / dt)))
    r = gaussian_filter1d(np.interp(t, at, rms), s)
    vis = [gaussian_filter1d(sig["diff"], s), gaussian_filter1d(sig["lum"], s)]
    g = np.nanmax([spearmanr(r, v).correlation for v in vis])
    s2 = max(1, round(local_win / dt))
    hp = lambda x: gaussian_filter1d(x, s2) - gaussian_filter1d(x, S)
    ri = np.interp(t, at, rms)
    l = np.nanmax([spearmanr(hp(ri), hp(v)).correlation for v in (sig["diff"], sig["lum"])])
    return float(g), float(l)


def structure_response(t, sig, boundaries, downbeats, win=2.0):
    """Visual contrast across each section boundary relative to contrast across ordinary downbeats (>1 = responds)."""
    feat = np.hstack([sig["hists"], 0.5 * sig["thumbs"]])

    def contrast(x):
        a = feat[(t >= x - win) & (t < x)]; b = feat[(t >= x) & (t < x + win)]
        if len(a) < 2 or len(b) < 2:
            return np.nan
        return float(np.linalg.norm(a.mean(0) - b.mean(0)))

    cb = [contrast(b) for b in boundaries]
    others = [d for d in downbeats if all(abs(d - b) > win for b in boundaries)]
    co = [contrast(d) for d in others]
    cb = [x for x in cb if not np.isnan(x)]; co = [x for x in co if not np.isnan(x)]
    if not cb or not co:
        return np.nan
    return float(np.mean(cb) / (np.median(co) + 1e-6))


def sync_metrics(video, clip_ann, fps=30, width=160):
    t, frames = read_frames(video, fps=fps, width=width)
    if len(t) < 10:
        return {"error": "too few frames"}
    sig = frame_signals(frames)
    beats = np.array(clip_ann["beats"]); downs = np.array(clip_ann["downbeats"])
    cuts = detect_cuts(t, sig)
    acc = detect_accents(t, sig)
    m = {}
    m.update(event_metrics(cuts, beats, downs, CUT_TOL, "cut"))
    m.update(event_metrics(acc, beats, downs, ACCENT_TOL, "accent"))
    m["cuts_per_min"] = 60 * len(cuts) / t[-1]
    m["energy_corr_global"], m["energy_corr"] = energy_corr(t, sig, np.array(clip_ann["rms"]), clip_ann["rms_hop_s"])
    # chance level: same picture against circularly shifted loudness (>= 2 s shifts); report the excess over chance
    rms = np.array(clip_ann["rms"]); hop = clip_ann["rms_hop_s"]
    if len(rms) > int(6 / hop):
        rng = np.random.default_rng(0)
        null = [energy_corr(t, sig, np.roll(rms, int(rng.uniform(2, len(rms) * hop - 2) / hop)), hop)[1] for _ in range(32)]
        null = [x for x in null if not np.isnan(x)]
        if null and m["energy_corr"] is not None and not np.isnan(m["energy_corr"]):
            m["energy_corr_chance"] = float(np.mean(null))
            m["energy_corr_excess"] = float(m["energy_corr"] - np.mean(null))
    m["structure_response"] = structure_response(t, sig, clip_ann["section_boundaries"], downs)
    m["motion_mean"] = float(sig["diff"].mean())
    m["_cuts"] = np.round(cuts, 3).tolist(); m["_accents"] = np.round(acc, 3).tolist()
    return {k: (round(v, 4) if isinstance(v, float) and not np.isnan(v) else (None if isinstance(v, float) else v))
            for k, v in m.items()}
