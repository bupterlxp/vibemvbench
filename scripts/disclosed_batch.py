"""Criteria-disclosed main experiment (VMB_DISCLOSE=1): the agent reads /task/CRITERIA.md (judged qualities and programmatic
checks; NOT the hidden song profile, key moments or song-specific checks). Results go to runs_disc/.

  python scripts/disclosed_batch.py <workers> pilot|main
"""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
os.environ["VMB_DISCLOSE"] = "1"
os.environ.setdefault("VMB_TASKS", "tasks_main90"); os.environ.setdefault("VMB_RUNS", "runs_disc")
os.environ.setdefault("VMB_IMAGE_BASE", "vmb-sandbox:base-v2")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vmb import DATA, TASKS_DIR
from vmb.rollout import run

SYSTEMS = ["claude:claude-opus-5-5:max", "codex:gpt-6-astra:max", "codex:gpt-6.1-sol:max",
           "codex:glm-5.3:max", "codex:deepseek-v4-pro-0813:max", "codex:deepseek-v4.1-flash:max", "codex:kimi-k3:max",
           "codex:grok-4.7:max", "codex:qwen3.8-max:max", "codex:gpt-6-luna:max", "codex:gpt-6.1-terra:max", "codex:nex-n2.6:max"]
mode = sys.argv[2] if len(sys.argv) > 2 else "pilot"
if len(sys.argv) > 3:   # optional comma-separated subset of SYSTEMS (e.g. while some gateways are down)
    SYSTEMS = [s for s in SYSTEMS if s in sys.argv[3].split(",")]
if mode == "pilot":
    tasks = ["jl_005_90", "jl_008_90"]   # Opus scored 1 and 8 on these without criteria; Sol 10 and 10
else:
    tasks = json.load(open(DATA / os.environ.get("VMB_SONGS", "main_disc_songs.json")))
# song-major order: every system gets the highest-priority songs first, so a partial run is still a complete comparison
jobs = [(t, s) for t in tasks for s in SYSTEMS]
def go(j):
    r = run(j[0], j[1], "bare", budget_min=None)
    print(json.dumps({k: r.get(k) for k in ("task_id", "system", "delivered", "wall_s", "infra_failure")}), flush=True)
with ThreadPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as ex:
    list(ex.map(go, jobs))
