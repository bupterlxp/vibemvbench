"""Score every finished rollout under runs/<cond>/ that is not fully scored yet; with --watch keep going until N results."""
import json, os, sys, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vmb import ROOT
from vmb.score import DEFAULT_JUDGES, score_run

cond = sys.argv[1] if len(sys.argv) > 1 else "tools"
target = int(sys.argv[2]) if len(sys.argv) > 2 else 0
while True:
    done = 0
    shard = [int(x) for x in os.environ.get("VMB_SHARD", "0/1").split("/")]   # i/n: this process scores every n-th run
    for k, rp in enumerate(sorted((ROOT / os.environ.get("VMB_RUNS", "runs") / cond).glob("*/*/result.json"))):
        if k % shard[1] != shard[0]:
            continue
        rd = rp.parent; sp = rd / "score.json"
        if rd.name.startswith("_") or json.load(open(rp)).get("skipped"):
            continue
        s = json.load(open(sp)) if sp.exists() else {}
        done += 1
        need = "l1" not in s or "sync" not in s or (not os.environ.get("VMB_NO_JUDGE") and s["l1"].get("decodable") and not all(f"judge:{j}" in s for j in DEFAULT_JUDGES))
        if need:
            try:
                s = score_run(rd)
                g = s.get(f"judge:{DEFAULT_JUDGES[0]}", {}) if DEFAULT_JUDGES else {}
                print(rd.parent.name, rd.name, "L1", s["l1"].get("pass"), "gemini", g.get("overall"), g.get("style"), flush=True)
            except Exception:
                print("ERR", rd, traceback.format_exc()[-400:], flush=True)
    if done >= target:
        break
    time.sleep(60)
