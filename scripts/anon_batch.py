"""Retrieval ablation: the song with no title, artist or file metadata (cond "anon"), criteria disclosed, network on.
Compared with the main run on the same songs. Results: runs_disc/anon/.

  python scripts/anon_batch.py <workers> <system> [songs json, default data/ears_songs.json]
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

songs = json.load(open(sys.argv[3] if len(sys.argv) > 3 else DATA / "ears_songs.json"))
def go(t):
    r = run(t, sys.argv[2], "anon", budget_min=None)
    print(json.dumps({k: r.get(k) for k in ("task_id", "system", "delivered", "wall_s", "infra_failure")}), flush=True)
with ThreadPoolExecutor(int(sys.argv[1])) as ex:
    list(ex.map(go, songs))
