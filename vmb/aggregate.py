"""Collect scores of all runs, run same-task pairwise judging, fit Elo, and write paper tables."""
import itertools
import os
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import ROOT, TASKS_DIR
from .elo import bootstrap, bootstrap_by_task
from .eval.judge import judge_pair_av

PAPER = ROOT / "overleaf"


def collect(cond="tools", systems=None):
    """rows: one per (system, task). Templates are included as systems named tpl-*."""
    rows = []
    R = ROOT / os.environ.get("VMB_RUNS", "runs")
    for sd in sorted((R / cond).glob("*")) + sorted((ROOT / "runs" / "template").glob("*")):
        for rd in sorted(x for x in sd.iterdir() if not x.name.startswith("_")):
            sp = rd / "score.json"
            rp = rd / "result.json"
            if rp.exists() and json.load(open(rp)).get("skipped"):
                continue  # placeholder for a stopped system, not a run
            if not sp.exists():
                continue
            s = json.load(open(sp))
            tid = rd.name.split("__")[0]
            vid = rd / "work" / "out" / "final.mp4" if (rd / "work").exists() else rd / "final.mp4"
            if systems is not None and sd.name not in systems:
                continue
            rows.append({"system": sd.name, "task": tid, "video": str(vid), **_flat(s)})
    return rows


def _flat(s):
    f = {"delivered": bool(s.get("l1", {}).get("decodable")), "l1_pass": bool(s.get("l1", {}).get("pass"))}
    for k, v in s.get("sync", {}).items():
        if not k.startswith("_"):
            f[k] = v
    for blk in ("lyrics", "listen"):
        for k, v in (s.get(blk) or {}).items():
            f[k] = v
    for k, v in s.items():
        if k.startswith("judge:"):
            n = k[6:]
            f[f"{n}:overall"] = v.get("overall"); f[f"{n}:style"] = v.get("style") or (v.get("raw") or {}).get("style")
            f[f"{n}:dim_mean"] = v.get("dim_mean"); f[f"{n}:l2_pass"] = v.get("l2_pass"); f[f"{n}:l2_strict"] = float(v.get("l2_strict", 0))
            for d, x in (v.get("scores") or {}).items():
                f[f"{n}:{d}"] = x
    return f


def pairwise(rows, systems=None, workers=4):
    """Judge every same-task pair twice (A/B swapped); the game outcome is the mean of the two graded verdicts."""
    from concurrent.futures import ThreadPoolExecutor
    by_task = defaultdict(list)
    for r in rows:
        if r["delivered"] and (systems is None or r["system"] in systems):
            by_task[r["task"]].append(r)
    pairs = [(tid, a, b) for tid, rs in by_task.items() for a, b in itertools.combinations(rs, 2)]

    def one(p):
        tid, a, b = p; td = TASKS_DIR / tid
        try:
            jm = os.environ.get("VMB_PAIR_JUDGE")   # default: Gemini 3.1 Pro
            s1 = judge_pair_av(a["video"], b["video"], td, model=jm)["score_a"]
            s2 = 1 - judge_pair_av(b["video"], a["video"], td, model=jm)["score_a"]
        except Exception as e:
            print("pair failed", tid, a["system"], b["system"], repr(e)[:120], flush=True); return None
        print(tid, a["system"], b["system"], s1, s2, flush=True)
        return (a["system"], b["system"], (s1 + s2) / 2, tid, s1, s2)

    with ThreadPoolExecutor(workers) as ex:
        games = [g for g in ex.map(one, pairs) if g]
    json.dump(games, open(ROOT / os.environ.get("VMB_RUNS", "runs") / "pairwise_games.json", "w"), indent=0)
    return games


def summarise(rows, keys):
    by = defaultdict(list)
    for r in rows:
        by[r["system"]].append(r)
    out = {}
    for sysname, rs in by.items():
        d = {"n": len(rs), "delivered": np.mean([r["delivered"] for r in rs])}
        for k in keys:
            v = [r.get(k) for r in rs if isinstance(r.get(k), (int, float)) and r.get(k) is not None]
            d[k] = float(np.mean(v)) if v else None
        out[sysname] = d
    return out


COLS = [("delivered", "Deliv.", "{:.0%}"), ("l1_pass", "L1", "{:.0%}"), ("cut_on_beat_norm", "Cut$\\uparrow$", "{:.2f}"),
        ("accent_on_beat_norm", "Acc.", "{:.2f}"), ("structure_response", "Sect.", "{:.2f}"), ("energy_corr", "Enrg.", "{:.2f}"),
        ("lyric_onset_err_med", "Lyr.err", "{:.2f}"), ("listen_beat_f", "Hear", "{:.2f}"),
        ("gemini-av:l2_pass", "L2", "{:.0%}"), ("gemini-av:overall", "L3", "{:.2f}"), ("elo", "Elo", "{:.0f}")]


def latex_table(summary, elo, path, caption, label):
    for k, v in summary.items():
        v["elo"] = elo.get(k, {}).get("elo") if elo else None
    lines = ["\\begin{table}[t]\\centering\\small", f"\\caption{{{caption}}}\\label{{{label}}}",
             "\\resizebox{\\linewidth}{!}{\\begin{tabular}{l" + "c" * len(COLS) + "}", "\\toprule",
             "System & " + " & ".join(c[1] for c in COLS) + " \\\\", "\\midrule"]
    for name in sorted(summary, key=lambda k: (k.startswith("tpl"), -(summary[k].get("elo") or 0))):
        cells = []
        for k, _, fmt in COLS:
            x = summary[name].get(k)
            cells.append("--" if x is None else fmt.format(x).replace("%", "\\%"))
        lines.append(name.replace("_", "\\_") + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    Path(path).write_text("\n".join(lines) + "\n")


def main(cond="tools", judge_pairs=True):
    rows = collect(cond)
    games = pairwise(rows) if judge_pairs else []
    elo = bootstrap([g[:3] for g in games], B=500, prior=1.0) if games else {}
    summary = summarise(rows, [c[0] for c in COLS if c[0] not in ("delivered", "elo")])
    latex_table(summary, elo, PAPER / "tables" / "main.tex",
                "Main results (\\textsc{" + cond.title() + "} condition). Cut/Acc.: chance-normalised on-beat precision of hard "
                "cuts / visual accents; Sect.: section response ($>$1 = picture changes with the song); Enrg.: beat-scale "
                "loudness--activity correlation; Lyr.err: median lyric onset error (s); Hear: beat F of the agent's timing "
                "sheet; L2/L3: Gemini audio-visual judge; Elo: pairwise judge Bradley--Terry.", "tab:main")
    json.dump({"summary": summary, "elo": elo}, open(ROOT / "runs" / f"summary_{cond}.json", "w"), indent=1)
    return summary, elo


if __name__ == "__main__":
    import sys
    main(*(sys.argv[1:2] or ["tools"]))
