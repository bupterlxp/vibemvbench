import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vmb import DATA
from vmb.annotate_v2 import annotate_v2
songs = json.load(open(DATA / "songs.json"))
rows = []
for s in songs:
    a = annotate_v2(s["song_id"])
    rows.append((s["song_id"], a["needs_review_frac"], round(a["bpm"] or 0, 1), s["bpm"]))
    print(*rows[-1], flush=True)
json.dump(rows, open(DATA / "annotations_v2" / "_review_summary.json", "w"))
print("songs needing review (>0 windows):", sum(r[1] > 0 for r in rows), "/", len(rows))
