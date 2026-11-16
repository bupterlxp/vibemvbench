"""Hearing test: each system hears every synthetic stimulus (audio only) and writes timing.json. Results: runs_hear/hear/.

  python scripts/hearing_batch.py <workers> <system,system,...>
"""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
os.environ.setdefault("VMB_RUNS", "runs_hear"); os.environ.setdefault("VMB_TASKS", "hearing_test")
os.environ.setdefault("VMB_IMAGE_BASE", "vmb-sandbox:base-v2")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vmb.hearing_test import STIMULI
from vmb.rollout import run

jobs = [(sid, s) for sid in STIMULI for s in sys.argv[2].split(",")]
def go(j):
    r = run(j[0], j[1], "hear", budget_min=None)
    print(json.dumps({k: r.get(k) for k in ("task_id", "system", "delivered", "wall_s", "infra_failure")}), flush=True)
with ThreadPoolExecutor(int(sys.argv[1])) as ex:
    list(ex.map(go, jobs))
