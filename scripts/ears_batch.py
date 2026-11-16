"""Swapped ears: one director hears each song only through another system's delivered timing sheet (or the reference).

  python scripts/ears_batch.py <workers> <director system> [sources, default sol,astra,grok,dspro,ref]
Songs: data/ears_songs.json (10 vocal songs where every source delivered a timing sheet). Results: runs_disc/ears/.
"""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
os.environ["VMB_DISCLOSE"] = "1"
os.environ.setdefault("VMB_TASKS", "tasks_main90"); os.environ.setdefault("VMB_RUNS", "runs_disc")
os.environ.setdefault("VMB_IMAGE_BASE", "vmb-sandbox:base-v2")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vmb import DATA
from vmb.rollout import run

director = sys.argv[2]
sources = sys.argv[3].split(",") if len(sys.argv) > 3 else ["sol", "astra", "grok", "dspro", "ref"]
jobs = [(t, src) for t in json.load(open(DATA / "ears_songs.json")) for src in sources]
def go(j):
    r = run(j[0], director, "ears", budget_min=None, tag=j[1])
    print(json.dumps({k: r.get(k) for k in ("task_id", "delivered", "wall_s", "infra_failure")} | {"ears": j[1]}), flush=True)
with ThreadPoolExecutor(int(sys.argv[1])) as ex:
    list(ex.map(go, jobs))
