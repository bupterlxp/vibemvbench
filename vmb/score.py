"""Score one rollout directory (or any MP4 for a task): L1, sync, lyrics, listening, judge. Results cached in score.json."""
import json
import os
from pathlib import Path

from . import ROOT, TASKS_DIR
from .eval.judge import judge_absolute, judge_absolute_av
from .llm import GEMINI_PRO
from .eval.l1 import l1_checks
from .eval.listening import listening_metrics
from .eval.lyrics import cached_ocr, lyric_metrics, word_lyric_metrics
from .eval.sync import sync_metrics

DIMS = ["UN", "EM", "RH", "ST", "LY", "VA", "CR"]


def _summarise_judge(j, td):
    rub = json.load(open(td / "rubric.json"))
    lvl = {c["id"]: c["level"] for c in rub["checks"]}
    v = {k: (x.get("verdict") if isinstance(x, dict) else x) for k, x in j.get("checks", {}).items()}
    npass = sum(1 for x in v.values() if x == "pass"); nfail = sum(1 for x in v.values() if x == "fail")
    core_fail = any(v.get(k) == "fail" for k, l in lvl.items() if l == "core")
    core_pass = any(v.get(k) == "pass" for k, l in lvl.items() if l == "core")
    sc = {k: j.get("scores", {}).get(k) for k in DIMS}
    nums = [x for x in sc.values() if isinstance(x, (int, float))]
    return {"raw": j, "l2_pass": npass / max(1, npass + nfail), "l2_strict": (not core_fail) and core_pass,
            "scores": sc, "dim_mean": sum(nums) / len(nums) if nums else None,
            "overall": j.get("overall") if isinstance(j.get("overall"), (int, float)) else (sum(nums) / len(nums) if nums else None),
            "style": j.get("style")}


JUDGES = {"gemini-av": lambda v, td, sync: judge_absolute_av(v, td, GEMINI_PRO),
          # second panel: a non-contestant model (GPT-6 Astra is also an evaluated system, so it is no longer a default judge)
          "gemini-flash-av": lambda v, td, sync: judge_absolute_av(v, td, os.environ.get("VMB_GEMINI_FLASH_JUDGE", "gemini-3.5-flash")),
          "gemini38-av": lambda v, td, sync: judge_absolute_av(v, td, "gemini-3.8-flash-high"),
          "gemini38z-av": lambda v, td, sync: judge_absolute_av(v, td, "gemini-3.8-flash"),
          "gpt-frames": lambda v, td, sync: judge_absolute(v, td, sync, model="gpt-6-astra")}


# absolute judge scores are off by default (main ranking = pairwise Elo, overall and per dimension); set VMB_JUDGES to add them
DEFAULT_JUDGES = tuple(j for j in os.environ.get("VMB_JUDGES", "").split(",") if j)


def score_video(video, task_id, out_json, timing=None, judges=DEFAULT_JUDGES, do_judge=True):
    out_json = Path(out_json)
    s = json.load(open(out_json)) if out_json.exists() else {}
    td = TASKS_DIR / task_id
    task = json.load(open(td / "task.json")); ref = json.load(open(td / "reference.json"))
    if "l1" not in s:
        s["l1"] = l1_checks(video, task["spec"], td / "audio.wav") if Path(video).exists() else {"decodable": False, "pass": False}
    ok = s["l1"].get("decodable") and s["l1"].get("has_video")
    if ok and "sync" not in s:
        s["sync"] = sync_metrics(video, ref)
    if ok and "lyrics" not in s and (task.get("category") == "lyric" or (task.get("free") and ref["lyrics_lines"])):
        s["lyrics"] = lyric_metrics(video, ref)
        s["lyrics"].update(word_lyric_metrics(ref, *cached_ocr(video)))
    if timing is not None and "listen" not in s:
        s["listen"] = listening_metrics(timing, ref) if timing else {}
    json.dump(s, open(out_json, "w"), indent=1, ensure_ascii=False)  # objective metrics persist even if a judge fails
    if os.environ.get("VMB_NO_JUDGE"):
        do_judge = False
    for jn in (judges if do_judge else ()):
        if ok and f"judge:{jn}" not in s:
            s[f"judge:{jn}"] = _summarise_judge(JUDGES[jn](video, td, s.get("sync")), td)
            json.dump(s, open(out_json, "w"), indent=1, ensure_ascii=False)
    return s


def score_run(rd, **kw):
    rd = Path(rd); res = json.load(open(rd / "result.json"))
    tp = rd / "work" / "out" / "timing.json"
    if not tp.exists():
        tp = rd / "work" / "timing.json"
    try:
        timing = json.load(open(tp)) if tp.exists() else {}
    except Exception:
        timing = {}
    vid = rd / "work" / "out" / "final.mp4"
    return score_video(vid, res["task_id"], rd / "score.json", timing=timing, **kw)
