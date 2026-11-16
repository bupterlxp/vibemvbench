"""Per-dimension arena from cached pairwise verdicts (no judge calls).

Each cached pairwise verdict holds a per-dimension winner (UN, EM, RH, ST, LY, VA, CR). The cache key needs the SHA-256 of
the judge proxy of each video, so proxies are re-encoded once per video (deterministic ffmpeg settings) and the key is looked
up; a game whose key is missing is skipped. Writes runs_disc/elo_by_dimension.json.

  python scripts/dimension_elo.py [workers] [--test N]
"""
import hashlib, json, os, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("VMB_TASKS", "tasks_main90")
from vmb import TASKS_DIR
from vmb.llm import CACHE
from vmb.eval.judge import AV_INSTR, judge_proxy, task_context
from vmb.elo import bootstrap_by_task

MODEL = os.environ.get("VMB_PAIR_JUDGE", "gemini-3.8-flash")
DIMS = ["UN", "EM", "RH", "ST", "LY", "VA", "CR"]
PAIR_TAIL = ("\n\nTwo music videos made for this song are attached: the FIRST is video A, the "
             "SECOND is video B. Watch and listen to both completely. Compare them on: understanding of the song, emotion, "
             "rhythm, structure, lyrics, visual quality, creativity. Differences in concept are fine; judge how well each "
             "realises a music video FOR THIS SONG. Ties should be rare: decide unless they are genuinely equal. Return JSON: "
             '{"per_dimension": {"UN": "A"|"B"|"=", "EM": ..., "RH": ..., "ST": ..., "LY": ..., "VA": ..., "CR": ...}, '
             '"reason": "...", "verdict": "A>>B"|"A>B"|"A=B"|"B>A"|"B>>A"}')
games = json.load(open("runs_disc/pairwise_games.json"))
test = int(sys.argv[sys.argv.index("--test") + 1]) if "--test" in sys.argv else 0
if test:
    games = games[:test]
vid = lambda s, t: f"runs_disc/bare/{s}/{t}/work/out/final.mp4"
videos = sorted({vid(a, t) for a, b, o, t, *_ in games} | {vid(b, t) for a, b, o, t, *_ in games})
digest = {}


def dg(v):
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir=os.environ.get("TMPDIR")).name
    try:
        judge_proxy(v, out=tmp)
        return v, hashlib.sha256(Path(tmp).read_bytes()).hexdigest()
    except Exception:
        return v, None
    finally:
        os.path.exists(tmp) and os.remove(tmp)


with ThreadPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8) as ex:
    for v, h in ex.map(dg, videos):
        digest[v] = h
ctx = {}


def verdict(va, vb, t):
    if digest.get(va) is None or digest.get(vb) is None:
        return None
    if t not in ctx:
        ctx[t] = task_context(TASKS_DIR / t) + PAIR_TAIL
    key = hashlib.sha256(json.dumps([MODEL, AV_INSTR, ctx[t], [digest[va], digest[vb]]]).encode()).hexdigest()
    p = CACHE / f"av_{key}.json"
    if not p.exists():
        return None
    txt = json.load(open(p))["text"]
    try:
        return json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
    except Exception:
        return None


V = {"A": 1.0, "B": 0.0, "=": 0.5}
per = {d: [] for d in DIMS}; hit = miss = 0
for a, b, out, t, *_ in games:
    r1, r2 = verdict(vid(a, t), vid(b, t), t), verdict(vid(b, t), vid(a, t), t)
    if not r1 or not r2:
        miss += 1; continue
    hit += 1
    for d in DIMS:
        x = V.get(str((r1.get("per_dimension") or {}).get(d)), 0.5)
        y = 1 - V.get(str((r2.get("per_dimension") or {}).get(d)), 0.5)
        per[d].append((a, b, (x + y) / 2, t))
print(f"games {len(games)}: both orders found {hit}, missing {miss}", flush=True)
if test:
    sys.exit(0)
res = {d: bootstrap_by_task(per[d], B=500, prior=1.0) for d in DIMS}
json.dump({"games": hit, "dims": res}, open("runs_disc/elo_by_dimension.json", "w"), indent=1)
sysn = sorted(res["UN"], key=lambda k: -res["UN"][k]["elo"])
print("system".ljust(22) + "".join(d.rjust(7) for d in DIMS))
for s in sysn:
    print(s.replace("codex-", "").replace("claude-claude-", "")[:22].ljust(22) + "".join(f"{res[d][s]['elo']:7.0f}" for d in DIMS))
