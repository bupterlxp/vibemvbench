"""Automatic retry: re-run every finished run in the main experiment that has no decodable final.mp4, whatever the reason,
up to MAX_RETRIES extra attempts per run. Earlier attempts are kept as _retry<k>_<task> (aggregation skips "_" dirs).
Runs alongside the batch runners (per-run locks prevent collisions); polls until no runner is left and nothing is retryable.

  python scripts/retry_undelivered.py <workers> [systems, comma-separated; default all codex systems]
"""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
os.environ["VMB_DISCLOSE"] = "1"
os.environ.setdefault("VMB_TASKS", "tasks_main90"); os.environ.setdefault("VMB_RUNS", "runs_disc")
os.environ.setdefault("VMB_IMAGE_BASE", "vmb-sandbox:base-v2")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vmb import ROOT
from vmb.rollout import mp4_ok, run

MAX_RETRIES = 3
B = ROOT / "runs_disc" / "bare"
systems = sys.argv[2].split(",") if len(sys.argv) > 2 else None


def candidates():
    out = []
    for sd in sorted(B.glob("codex-*")):
        system = "codex:" + sd.name[len("codex-"):-len("-max")] + ":max"
        if systems and system not in systems:
            continue
        for rd in sorted(sd.glob("[fj]*_90")):
            rp = rd / "result.json"
            if not rp.exists() or Path(str(rd) + ".lock").exists():
                continue
            res = json.load(open(rp))
            if res.get("skipped") or mp4_ok(rd / "work" / "out" / "final.mp4"):
                continue
            k = len(list(sd.glob(f"_retry*_{rd.name}")))
            if k < MAX_RETRIES:
                out.append((rd, system, k + 1))
    return out


def retry(c):
    rd, system, k = c
    dst = rd.parent / f"_retry{k}_{rd.name}"
    subprocess.run(["docker", "run", "--rm", "-v", f"{rd.parent}:/p", "vmb-sandbox:base", "sh", "-c",
                    f"mv /p/{rd.name} /p/{dst.name} && chmod -R a+rwX /p/{dst.name}"], check=True)
    r = run(rd.name, system, "bare", budget_min=None)
    print(json.dumps({"task_id": rd.name, "system": system, "retry": k, "delivered": r.get("delivered")}), flush=True)


def runners_alive():
    return subprocess.run(["pgrep", "-f", "scripts/disclosed_batch.py"], capture_output=True).returncode == 0


with ThreadPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 3) as ex:
    while True:
        cs = candidates()
        list(ex.map(retry, cs))
        if not cs and not runners_alive():
            break
        time.sleep(600)
