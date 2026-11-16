"""Heard vs did: for each run, three numbers on the same beat grid scale (F at +-70 ms):
  hear  = agent's timing.json beats vs reference beats            (perception)
  exec  = rendered cuts (precision) against the agent's OWN beats  (does it cut where it thinks the beat is?)
  out   = rendered cuts (precision) against the reference beats    (what the viewer gets)
"""
import json, glob, os, collections, numpy as np, sys
sys.path.insert(0, ".")
from vmb.eval.listening import f_measure
def prec(events, ref, tol=.07):
    if not events or not ref: return None
    ref = np.asarray(ref); return float(np.mean([np.min(np.abs(ref - e)) <= tol for e in events]))
def chance(events, ref, tol=.07, dur=90.0):
    if not ref: return None
    return min(1.0, len(ref) * 2 * tol / dur)
rows = collections.defaultdict(list)
for sp in glob.glob("runs_disc/bare/*/[fj]*_90/score.json"):
    rd = os.path.dirname(sp); s = json.load(open(sp)); sync = s.get("sync") or {}
    cuts = sync.get("_cuts") or []
    try: tm = json.load(open(rd + "/work/out/timing.json"))
    except Exception: continue
    ref = json.load(open(f"tasks_main90/{os.path.basename(rd)}/reference.json"))
    own = [b for b in (tm.get("beats") or []) if isinstance(b, (int, float))]
    if len(cuts) < 5 or len(own) < 10: continue
    hear = f_measure(ref["beats"], own)
    ex, out = prec(cuts, own), prec(cuts, ref["beats"])
    ch = chance(cuts, ref["beats"])
    rows[os.path.basename(os.path.dirname(rd))].append((hear, ex, out, ch))
print("system".ljust(30), "n   hear(F)  cut->own beats  cut->ref beats  chance")
for k, v in sorted(rows.items(), key=lambda kv: -np.mean([x[2] for x in kv[1]])):
    a = np.array(v, float)
    print(k.replace("codex-", "").replace("claude-claude-", "")[:30].ljust(30), str(len(v)).ljust(4), "%.2f     %.2f            %.2f            %.2f" % tuple(np.nanmean(a, 0)))
# runs that hear well but cut off-beat vs runs that hear badly but cut on their own grid
allv = [x for v in rows.values() for x in v]
a = np.array(allv, float)
print("\nall runs", len(a), " corr(hear, out) %.2f   corr(exec, out) %.2f" % (np.corrcoef(a[:, 0], a[:, 2])[0, 1], np.corrcoef(a[:, 1], a[:, 2])[0, 1]))
print("hear>=.8 but exec<.5 (heard, did not cut on it):", int(np.sum((a[:, 0] >= .8) & (a[:, 1] < .5))), "of", int(np.sum(a[:, 0] >= .8)))
print("hear<.5 but exec>=.8 (cut faithfully on a wrong grid):", int(np.sum((a[:, 0] < .5) & (a[:, 1] >= .8))), "of", int(np.sum(a[:, 0] < .5)))
