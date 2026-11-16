"""Ear lineage: for each run, which listening toolchain the agent built (from commands + code in /work).
Components are detected by pattern; a run's 'ear' is the sorted set of components. Report per system the component shares,
the most common ears, and the listening accuracy each ear reaches."""
import json, glob, os, re, collections, numpy as np
C = {
    "own DSP": r"np\.fft|rfft|\bstft\b|spectral[_ ]?flux|autocorr|find_peaks|checkerboard|self[_-]?similar",
    "librosa": r"\blibrosa\b",
    "madmom/essentia/aubio": r"\bmadmom\b|\bessentia\b|\baubio\b",
    "beat_this": r"beat_this|beat-this",
    "allin1/msaf": r"\ballin1\b|\bmsaf\b",
    "demucs": r"\bdemucs\b|htdemucs|\bspleeter\b",
    "whisper": r"whisper",
    "forced align": r"whisperx|stable_whisper|stable-ts|ctc[_ ]?segment|forced[_ ]?align|aeneas|word_timestamps",
    "web lyrics": r"genius\.com|musixmatch|azlyrics|lrclib|letras\.|jamendo\.com/.*lyrics|lyrics\.ovh",
}
def ear(rd):
    t = ""
    try:
        for l in open(rd + "/trajectory.jsonl", errors="ignore"):
            if '"command' in l or "tool_use" in l: t += l[:6000]
    except Exception: return None
    for p in glob.glob(rd + "/work/**/*.py", recursive=True):
        if "site-packages" in p or "node_modules" in p: continue
        try:
            if os.path.getsize(p) < 400000: t += open(p, errors="ignore").read()
        except OSError: pass
    return frozenset(k for k, p in C.items() if re.search(p, t, re.I))
per = collections.defaultdict(list)
for sp in glob.glob("runs_disc/bare/*/[fj]*_90/score.json"):
    rd = os.path.dirname(sp); e = ear(rd)
    if e is None: continue
    L = (json.load(open(sp)).get("listen") or {})
    per[os.path.basename(os.path.dirname(rd))].append((e, L.get("listen_beat_f"), L.get("listen_section_f"), L.get("listen_lyric_onset_err_med")))
out = {}
print("component share per system")
comps = list(C)
print("system".ljust(24) + "".join(c[:9].rjust(10) for c in comps))
for s, v in sorted(per.items()):
    print(s.replace("codex-", "").replace("claude-claude-", "")[:24].ljust(24) + "".join(f"{sum(c in e for e, *_ in v) / len(v):10.0%}" for c in comps))
allr = [x for v in per.values() for x in v]
print("\nbeat F by beat-tracking component (all systems):")
for c in ["own DSP", "librosa", "madmom/essentia/aubio", "beat_this"]:
    w = [b for e, b, *_ in allr if c in e and b is not None]; wo = [b for e, b, *_ in allr if c not in e and b is not None]
    print(f"  {c:22s} with {np.mean(w):.2f} (n={len(w)})  without {np.mean(wo):.2f} (n={len(wo)})")
print("\nlyric onset error by lyric pipeline (vocal songs):")
for name, test in [("whisper + forced align", lambda e: "whisper" in e and "forced align" in e), ("whisper only", lambda e: "whisper" in e and "forced align" not in e),
                   ("web only", lambda e: "web lyrics" in e and "whisper" not in e), ("demucs before ASR", lambda e: "demucs" in e and "whisper" in e)]:
    w = [x for e, _, _, x in allr if test(e) and x is not None]
    print(f"  {name:24s} median {np.median(w):.2f} s (n={len(w)})" if w else f"  {name:24s} -")
cnt = collections.Counter(e for e, *_ in allr)
print("\nmost common ears:")
for e, n in cnt.most_common(8):
    b = [x for ee, x, *_ in allr if ee == e and x is not None]
    print(f"  {n:3d}  beatF {np.mean(b) if b else float('nan'):.2f}  " + " + ".join(sorted(e)))
json.dump({s: [[sorted(e), b, sf, ly] for e, b, sf, ly in v] for s, v in per.items()}, open("runs_disc/ear_lineage.json", "w"))
