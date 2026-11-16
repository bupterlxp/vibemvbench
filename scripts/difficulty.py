"""Per-song difficulty features for stratified song selection (full songs). Uses the v2 ensemble beats.

Dimensions (each z-scored over the 73 songs, then combined into a composite):
  rhythm    tempo instability (CV of beat intervals), tempo-octave switches, share of windows where trackers disagree
  structure number of section boundaries per minute, max novelty contrast (how distinct sections are)
  dynamics  loudness range across the song (p95/p5 of 1 s RMS, in dB)
  lyrics    has lyrics, words per minute, non-English
  genre     low-percussion genres (no clear drum grid) get +1
  length    song duration
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vmb import DATA

LOW_PERC = {"Classical", "Jazz", "Folk", "Ambient Electronic", "Blues", "Country", "Experimental", "Avant-Garde"}
songs = json.load(open(DATA / "songs.json"))
rows = []
for s in songs:
    a = json.load(open(DATA / "annotations_v2" / f"{s['song_id']}.json"))
    b = np.array(a["beats"]); ibi = np.diff(b) if len(b) > 2 else np.array([0.5])
    ibi = ibi[(ibi > 0.15) & (ibi < 2.0)]
    tempo_cv = float(np.std(ibi) / np.mean(ibi)) if len(ibi) else 1.0
    loc = 60 / ibi if len(ibi) else np.array([120.0])
    med = np.median(loc)
    octave_sw = float(np.mean((loc > 1.7 * med) | (loc < med / 1.7)))
    disagree = a.get("needs_review_frac", 0.0)
    dur = a["duration"]
    nsec = len(a["section_boundaries"]) / dur * 60
    rms = np.array(a["rms"]); hop = a["rms_hop_s"]; k = max(1, int(1 / hop))
    sec = rms[: len(rms) // k * k].reshape(-1, k).mean(1) + 1e-6
    dyn_db = float(20 * np.log10(np.percentile(sec, 95) / np.percentile(sec, 5)))
    words = len(a.get("lyrics_words") or [])
    wpm = words / dur * 60
    rows.append({"song_id": s["song_id"], "title": s["title"], "artist": s["artist"], "genre": s["genre"],
                 "language": s.get("language"), "license": s["license"], "duration": round(dur, 1), "bpm": round(float(med), 1),
                 "tempo_cv": round(tempo_cv, 3), "octave_switch": round(octave_sw, 3), "tracker_disagree": disagree,
                 "sections_per_min": round(nsec, 2), "dyn_range_db": round(dyn_db, 1), "has_lyrics": words > 0,
                 "words_per_min": round(wpm, 1), "non_english": s.get("language") not in ("en", "instrumental", "unk", None),
                 "low_percussion": s["genre"] in LOW_PERC})


def z(key, sign=1):
    v = np.array([r[key] for r in rows], float)
    return sign * (v - v.mean()) / (v.std() + 1e-9)


dims = {
    "rhythm": (z("tempo_cv") + z("octave_switch") + z("tracker_disagree")) / 3,
    "structure": z("sections_per_min"),
    "dynamics": z("dyn_range_db"),
    "lyrics": (np.array([r["has_lyrics"] for r in rows], float) * 0.5 + z("words_per_min") * 0.5
               + np.array([r["non_english"] for r in rows], float) * 0.5),
    "genre": np.array([r["low_percussion"] for r in rows], float),
    "length": z("duration"),
}
comp = sum(dims.values()) / len(dims)
order = np.argsort(comp)
for i, r in enumerate(rows):
    for d, v in dims.items():
        r[f"d_{d}"] = round(float(v[i]), 2)
    r["difficulty"] = round(float(comp[i]), 3)
ranks = np.empty(len(rows), int); ranks[order] = np.arange(len(rows))
for i, r in enumerate(rows):
    r["tier"] = ["easy", "medium", "hard"][min(2, ranks[i] * 3 // len(rows))]
json.dump(rows, open(DATA / "difficulty.json", "w"), ensure_ascii=False, indent=1)
for t in ("easy", "medium", "hard"):
    rs = [r for r in rows if r["tier"] == t]
    print(f"{t:6s} n={len(rs)}  mean dur {np.mean([r['duration'] for r in rs]):.0f}s  lyrics {sum(r['has_lyrics'] for r in rs)}  "
          f"low-perc {sum(r['low_percussion'] for r in rs)}  tempo_cv {np.mean([r['tempo_cv'] for r in rs]):.2f}  "
          f"dyn {np.mean([r['dyn_range_db'] for r in rs]):.1f}dB")
print("hardest:", [(r["song_id"], r["genre"]) for r in sorted(rows, key=lambda r: -r["difficulty"])[:8]])
print("easiest:", [(r["song_id"], r["genre"]) for r in sorted(rows, key=lambda r: r["difficulty"])[:8]])
