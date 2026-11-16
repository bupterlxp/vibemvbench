"""Label each main-run video's MV style from the agent's production note (notes.md), with the non-contestant judge model.

Styles (one primary): narrative (characters and a story), performance (a singer, band or dancer performing), lyric (the
sung words are the main visual element), abstract (shapes, colour fields, particles with no characters), visualiser
(graphics driven directly by the audio signal: bars, waveforms, spectra). Text only, ~1-2k tokens per call.
Writes runs_disc/style_labels.json and prints per-style counts per system.

  python scripts/style_labels.py [workers]
"""
import json, os, sys, glob, re, urllib.request, collections
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vmb import load_env
load_env()
STYLES = ["narrative", "performance", "lyric", "abstract", "visualiser"]
Q = ("Below is a production note an AI agent wrote about the music video it made. Classify the video's PRIMARY style as exactly "
     "one of: narrative (characters and a story), performance (a singer, band or dancer performing), lyric (the sung words "
     "are the main visual element), abstract (shapes, colour fields or particles without characters), visualiser (graphics "
     "driven directly by the audio signal, such as bars, waveforms or spectra). Answer with the one word only.\n\nNOTE:\n")
OUT = "runs_disc/style_labels.json"
labels = json.load(open(OUT)) if os.path.exists(OUT) else {}


def ask(note):
    body = {"model": "gemini-3.8-flash", "messages": [{"role": "user", "content": Q + note[:6000]}], "max_tokens": 400}
    for _ in range(5):
        try:
            r = urllib.request.Request(os.environ["ZGC_BASE_URL"] + "/v1/chat/completions", data=json.dumps(body).encode(),
                                       headers={"Authorization": "Bearer " + os.environ["ZGC_API_KEY"], "Content-Type": "application/json"})
            t = (json.load(urllib.request.urlopen(r, timeout=120))["choices"][0]["message"]["content"] or "").lower()
            for s in STYLES:
                if s in t or (s == "visualiser" and "visualizer" in t):
                    return s
            return "other"
        except Exception:
            continue
    return None


def one(p):
    sysn, t = p.split("/")[2], p.split("/")[3]
    k = f"{sysn}/{t}"
    if k in labels:
        return k, labels[k]
    note = open(p, errors="ignore").read()
    return k, ask(note) if note.strip() else None


paths = sorted(glob.glob("runs_disc/bare/*-max/[fj]*/work/out/notes.md"))
paths = [p for p in paths if "nex-" not in p]
with ThreadPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as ex:
    for k, v in ex.map(one, paths):
        if v: labels[k] = v
json.dump(labels, open(OUT, "w"), indent=0)
c = collections.defaultdict(collections.Counter)
for k, v in labels.items(): c[k.split("/")[0]][v] += 1
print("labelled", len(labels), "of", len(paths))
for s in sorted(c): print(s.replace("codex-", "")[:22].ljust(22), dict(c[s]))
