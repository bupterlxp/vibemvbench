"""RQ1 on the criteria-disclosed main run: how each system listens and how it checks its own video.

Per rollout, from the commands in the trajectory plus the code left in /work, flag the listening and checking moves; per
system, report the share of delivered runs with each move. Codex logs commands, messages and web searches (view_image
calls are not logged, so seeing is inferred from frames the agent extracted).

  python scripts/rq1_behaviour.py [runs_dir=runs_disc] [cond=bare]
Output: <runs_dir>/rq1_behaviour.{json,md}
"""
import json, re, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vmb import ROOT

LISTEN = {
    "librosa": r"\blibrosa\b",
    "beat_this": r"beat_this|beat-this",
    "madmom_essentia_aubio": r"\bmadmom\b|\bessentia\b|\baubio\b",
    "own_dsp": r"np\.fft|rfft|\bstft\b|spectral[_ ]?flux|autocorr|find_peaks|novelty|checkerboard|self[_-]?similar",
    "whisper": r"whisper",
    "word_timestamps": r"word_timestamps|return_timestamps=.?word|\"words\"|\.words\b|whisperx|stable_whisper|ctc[_ ]?align|forced[_ ]?align",
    "vocal_separation": r"\bdemucs\b|\bspleeter\b|htdemucs|\bhpss\b",
    "web_lyrics": r"genius\.com|musixmatch|lyrics\.(ovh|com)|letras\.|azlyrics|jamendo\.com|lrclib",
}
CHECK = {
    "frames_extracted": r"(ffmpeg[^\n]{0,200}(-vframes|select=|fps=|-frames:v)[^\n]{0,200}\.(png|jpe?g))|contact[_ ]?sheet|montage|tile=",
    "measures_frames_by_code": r"(PIL|Image\.open|cv2\.imread|imageio)[^\n]{0,300}(mean|std|histogram|getextrema|diff)",
    "checks_soundtrack": r"(correlat|xcorr|correlate)[^\n]{0,200}(audio|wav)|ffprobe[^\n]{0,120}(audio|a:0|codec_type)",
    "checks_cuts_vs_beats": r"(cut|scene)[^\n]{0,80}(beat|onset)[^\n]{0,80}(error|offset|diff|dist)|scdet|select=.gt\(scene",
    "reads_criteria": r"CRITERIA\.md",
}


def texts(rd):
    cmds, msgs, searches = [], [], 0
    for line in open(rd / "trajectory.jsonl", errors="ignore"):
        try:
            e = json.loads(line)
        except Exception:
            continue
        it = e.get("item") or {}
        if e.get("type") != "item.completed":
            continue
        if it.get("type") == "command_execution":
            cmds.append(it.get("command", "") + "\n" + (it.get("aggregated_output") or "")[:2000])
        elif it.get("type") in ("agent_message", "reasoning"):
            msgs.append(it.get("text", ""))
        elif it.get("type") in ("web_search", "search"):
            searches += 1
        # Claude Code stream-json
        for c in (e.get("message") or {}).get("content", []) if isinstance(e.get("message"), dict) else []:
            if c.get("type") == "tool_use":
                cmds.append(json.dumps(c.get("input", {}))[:4000])
    code = " ".join(p.read_text(errors="ignore") for p in (rd / "work").rglob("*.py")
                    if p.stat().st_size < 400_000 and "site-packages" not in str(p) and "node_modules" not in str(p))
    return "\n".join(cmds), "\n".join(msgs), searches, code


def main():
    runs = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "runs_disc"); cond = sys.argv[2] if len(sys.argv) > 2 else "bare"
    rows = []
    for rp in sorted((runs / cond).glob("*/*/result.json")):
        rd = rp.parent; res = json.load(open(rp))
        if rd.name.startswith("_") or res.get("skipped") or res.get("infra_failure") or not (rd / "trajectory.jsonl").exists():
            continue
        cmds, msgs, searches, code = texts(rd)
        allt = cmds + "\n" + code
        r = {"system": rd.parent.name, "task": rd.name, "delivered": bool(res.get("delivered")), "n_commands": cmds.count("\n/bin/") + 1,
             "web_searches": searches, "wall_min": round(res.get("wall_s", 0) / 60, 1)}
        r.update({k: bool(re.search(p, allt, re.I)) for k, p in LISTEN.items()})
        r["web_lyrics"] = r["web_lyrics"] or bool(searches and re.search(r"lyric|letra|paroles|songtext", msgs + cmds, re.I))
        r.update({k: bool(re.search(p, allt, re.I)) for k, p in CHECK.items()})
        sp = rd / "score.json"
        if sp.exists():
            s = json.load(open(sp)); L = s.get("listen") or {}
            r.update({k: L.get(k) for k in ("listen_beat_f", "listen_section_f", "listen_lyric_line_err") if k in L})
        rows.append(r)
    json.dump(rows, open(runs / "rq1_behaviour.json", "w"), indent=1)
    by = defaultdict(list)
    for r in rows:
        by[r["system"]].append(r)
    keys = list(LISTEN) + list(CHECK)
    md = ["| system | n | deliv | " + " | ".join(keys) + " | searches/run | min/run |", "|" + "---|" * (len(keys) + 5)]
    for s, rs in sorted(by.items()):
        n = len(rs)
        md.append(f"| {s} | {n} | {sum(r['delivered'] for r in rs)} | " + " | ".join(f"{sum(r[k] for r in rs) / n:.0%}" for k in keys)
                  + f" | {sum(r['web_searches'] for r in rs) / n:.1f} | {sum(r['wall_min'] for r in rs) / n:.0f} |")
    (runs / "rq1_behaviour.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
