"""Pick clip windows: start on a downbeat, contain >=1 section boundary well inside, maximise lyric coverage."""
import numpy as np


def pick_clip(ann, length, min_margin=None, avoid=()):
    d = np.array(ann["downbeats"]); bounds = np.array(ann["section_boundaries"])
    # v2 ensemble annotations can have sparse/no downbeats (windows won by librosa); fall back to every 4th beat
    if len(d) < max(4, 0.1 * len(ann["beats"])):
        d = np.array(ann["beats"])[::4]
    dur = ann["duration"]; m = min_margin or max(3.0, length * 0.2)
    lines = ann["lyrics_lines"]
    best, best_s = None, -1e9
    for s in d:
        e = s + length
        if e > dur - 0.5 or s < 1:
            continue
        if any(abs(s - a) < length for a in avoid):
            continue
        inside = bounds[(bounds > s + m) & (bounds < e - m)]
        if len(inside) == 0:
            continue
        cov = sum(max(0, min(e, l["end"]) - max(s, l["start"])) for l in lines) / length
        full_lines = sum(1 for l in lines if l["start"] >= s and l["end"] <= e)
        # prefer a single central boundary and decent lyric coverage
        centre = -abs(inside[0] - (s + length / 2)) / length
        score = 1.0 * min(cov, 0.8) + 0.05 * min(full_lines, 6) + 0.3 * centre
        if score > best_s:
            best, best_s = (float(s), float(e), [float(x) for x in inside], cov, full_lines), score
    return best


def clip_annotation(ann, s, e):
    sh = lambda xs: [round(x - s, 3) for x in xs if s <= x < e]
    lines = [{"start": round(max(l["start"], s) - s, 3), "end": round(min(l["end"], e) - s, 3), "text": l["text"]}
             for l in ann["lyrics_lines"] if l["end"] > s + 0.3 and l["start"] < e - 0.3]
    words = [{"start": round(w["start"] - s, 3), "end": round(w["end"] - s, 3), "text": w["text"]}
             for w in ann["lyrics_words"] if s <= w["start"] < e]
    hop = ann["rms_hop_s"]; i0, i1 = int(s / hop), int(e / hop)
    cb = [x for x in ann["beats"] if s <= x < e]
    bpm = float(60 / np.median(np.diff(cb))) if len(cb) > 2 else ann["bpm"]
    return {"duration": round(e - s, 3), "bpm": bpm, "song_bpm": ann["bpm"], "beats": sh(ann["beats"]), "downbeats": sh(ann["downbeats"]),
            "section_boundaries": sh(ann["section_boundaries"]), "lyrics_lines": lines, "lyrics_words": words,
            "rms_hop_s": hop, "rms": ann["rms"][i0:i1]}
