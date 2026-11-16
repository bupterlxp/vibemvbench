"""On-screen lyric accuracy and timing via OCR, matched against reference line timings."""
import difflib
import re
import unicodedata

import numpy as np

from .media import read_frames

_OCR = None


def norm(s):
    s = unicodedata.normalize("NFKC", s).lower()
    return re.sub(r"[\W_]+", "", s)


def cer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    if not r:
        return 0.0
    sm = difflib.SequenceMatcher(None, r, h)
    edits = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")
    return min(1.0, edits / len(r))


def ocr_frames(video, fps=4, width=960):
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR
        _OCR = RapidOCR()
    t, frames = read_frames(video, fps=fps, width=width)
    texts = []
    for f in frames:
        res, _ = _OCR(f[:, :, ::-1])
        texts.append(" ".join(r[1] for r in (res or []) if r[2] > 0.5))
    return t, texts


def best_partial(line, text):
    """Similarity of the line to the best-matching window of on-screen text (handles extra on-screen words). A partially
    revealed line (word-by-word display: the on-screen text is a >= 4-char prefix of the line) also counts."""
    a, b = norm(line), norm(text)
    if not a or not b:
        return 0.0
    if len(b) >= 4 and a.startswith(b):
        return max(0.6 + 0.4 * len(b) / len(a), difflib.SequenceMatcher(None, a, b).ratio())
    if len(b) <= len(a):
        return difflib.SequenceMatcher(None, a, b).ratio()
    best = 0.0
    for i in range(0, len(b) - len(a) + 1, max(1, len(a) // 4)):
        best = max(best, difflib.SequenceMatcher(None, a, b[i:i + len(a)]).ratio())
    return best


def prefix_sim(line, text, min_chars=4):
    """How much of the START of `line` is on screen: longest matched prefix (>= min_chars) / len(line)."""
    a, b = norm(line), norm(text)
    if len(a) < min_chars or not b:
        return 0.0
    best = 0
    for k in range(min_chars, len(a) + 1, max(1, len(a) // 12)):
        if a[:k] in b:
            best = k
        elif best:
            break
    return best / len(a)


def _runs(idx):
    """Split sorted frame indices into contiguous runs."""
    if len(idx) == 0:
        return []
    cut = np.where(np.diff(idx) > 1)[0] + 1
    return np.split(idx, cut)


def cached_ocr(video):
    import json
    from pathlib import Path
    cp = Path(video).with_suffix(".ocr.json")
    if "work" in Path(video).parts:  # sandbox-owned dir may not be writable by the host user
        cp = Path(str(Path(video).parent).split("/work/")[0]) / "ocr_cache.json"
    if cp.exists():
        d = json.load(open(cp)); return np.array(d["t"]), d["texts"]
    t, texts = ocr_frames(video)
    json.dump({"t": list(map(float, t)), "texts": texts}, open(cp, "w"), ensure_ascii=False)
    return t, texts


def lyric_metrics(video, clip_ann, match_thr=0.6, ocr=None):
    lines = [l for l in clip_ann["lyrics_lines"] if len(norm(l["text"])) >= 4]
    if not lines:
        return {}
    t, texts = ocr or cached_ocr(video)
    t = np.asarray(t) - 0.5 * (t[1] - t[0]) if len(t) > 1 else np.asarray(t)  # frame centre -> earliest possible onset
    S = np.array([[best_partial(l["text"], x) for x in texts] for l in lines])  # lines x frames
    # assign each frame to one line: best similarity; near-ties (repeated/overlapping lines) go to the line
    # whose sung interval is closest in time to the frame
    owner = np.full(len(t), -1)
    for j in range(len(t)):
        m = S[:, j].max()
        if m < match_thr:
            continue
        cand = np.where(S[:, j] >= m - 0.02)[0]
        dist = [0.0 if lines[k]["start"] <= t[j] <= lines[k]["end"] else min(abs(t[j] - lines[k]["start"]), abs(t[j] - lines[k]["end"]))
                for k in cand]
        owner[j] = cand[int(np.argmin(dist))]
    shown, onset_err, cers, in_window = 0, [], [], []
    for k, l in enumerate(lines):
        runs = _runs(np.where(owner == k)[0])
        if not runs:
            continue
        shown += 1
        r = min(runs, key=lambda r: abs(t[r[0]] - l["start"]))
        # word-by-word display: the line starts appearing before it is fully readable; look back for its first words
        j0 = r[0]
        while j0 > 0 and owner[j0 - 1] in (-1, k) and prefix_sim(l["text"], texts[j0 - 1]) > 0:
            j0 -= 1
        onset_err.append(t[j0] - l["start"])
        cers.append(1 - S[k, r].max())
        in_window.append(float(np.mean((t[r] >= l["start"] - 0.5) & (t[r] <= l["end"] + 0.5))))
    # sync: while a line is being sung, is THAT line on screen? (previewing the next line in a gap is not penalised)
    sung_ok, sung_wrong, sung_n = 0, 0, 0
    for j in range(len(t)):
        cur = [k for k, l in enumerate(lines) if l["start"] <= t[j] <= l["end"]]
        if not cur:
            continue
        sung_n += 1
        if owner[j] in cur:
            sung_ok += 1
        elif owner[j] >= 0:
            sung_wrong += 1
    out = {"lyric_line_coverage": shown / len(lines), "lyric_n_lines": len(lines)}
    if sung_n:
        out["lyric_sync"] = sung_ok / sung_n
        out["lyric_wrong_line"] = sung_wrong / sung_n
    if shown:
        out["lyric_cer"] = float(np.mean(cers))
        out["lyric_onset_err_med"] = float(np.median(np.abs(onset_err)))
        out["lyric_onset_bias"] = float(np.median(onset_err))
        out["lyric_in_window"] = float(np.mean(in_window))
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()}


def word_lyric_metrics(clip_ann, t, texts, min_len=3, at_tol=0.3, win=6.0):
    """Word-level lyric timing, robust to word-by-word, fragment and full-line display.

    For each sung word (human word timings; >= min_len chars) we look at frames whose normalised OCR text contains it:
      lyric_word_cov   share of sung words that appear on screen at all (within +-win s of being sung)
      lyric_word_at    share of sung words that are ON SCREEN WHILE BEING SUNG (a frame within +-at_tol s of the word start)
      lyric_word_lead  median lead (s) of the word's first appearance in the window before it is sung (>0 = shown early)
    OCR often merges a rendered line into one string, so words are matched as substrings of the normalised text."""
    words = [w for w in clip_ann.get("lyrics_words", []) if len(norm(w["text"])) >= min_len]
    if not words or not len(t):
        return {}
    toks = [norm(x) for x in texts]
    t = np.asarray(t)
    cov = at = 0; leads = []
    for w in words:
        k = norm(w["text"])
        hit = np.array([k in s for s in toks])
        m = hit & (np.abs(t - w["start"]) <= win)
        if not m.any():
            continue
        cov += 1
        if (hit & (np.abs(t - w["start"]) <= at_tol)).any():
            at += 1
        leads.append(float(w["start"] - t[m].min()))
    n = len(words)
    out = {"lyric_word_n": n, "lyric_word_cov": round(cov / n, 4), "lyric_word_at": round(at / n, 4)}
    if leads:
        out["lyric_word_lead"] = round(float(np.median(leads)), 3)
    return out
