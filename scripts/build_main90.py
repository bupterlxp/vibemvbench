"""Build 90 s clip tasks for all 73 songs into tasks_main90/ in parallel (Gemini listens once per clip)."""
import json, os, sys
os.environ.setdefault("VMB_TASKS", "tasks_main90")
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vmb import DATA, TASKS_DIR
from vmb.tasks_free import build

songs = [s["song_id"] for s in json.load(open(DATA / "songs.json"))]
diff = {r["song_id"]: r for r in json.load(open(DATA / "difficulty.json"))}


def one(sid):
    try:
        a = json.load(open(DATA / "annotations_v2" / f"{sid}.json"))
        if a["duration"] < 95:   # shorter than the clip length: use the whole song (task id <song>_full)
            out = build(song_ids=[sid], full=True, hint_annotation=False)
        else:
            out = build(lengths=(90,), song_ids=[sid], hint_annotation=False)
    except Exception as e:
        print("FAIL", sid, repr(e)[:150], flush=True); return []
    for tid in out:
        p = TASKS_DIR / tid / "task.json"; t = json.load(open(p)); d = diff[sid]
        t["difficulty_tier"] = d["tier"]; t["difficulty"] = {k: d[k] for k in d if k.startswith("d_") or k == "difficulty"}
        json.dump(t, open(p, "w"), ensure_ascii=False, indent=1)
    return out


with ThreadPoolExecutor(8) as ex:
    built = [t for r in ex.map(one, songs) for t in r]
print(len(built), "tasks in", TASKS_DIR)
