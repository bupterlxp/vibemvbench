"""Arena for the criteria-disclosed main run: graded pairwise judge (A/B swapped) on every same-song pair of delivered
videos, Bradley-Terry Elo with a song-clustered bootstrap. Judge calls are cached, so re-running only judges new pairs.

  VMB_PAIR_JUDGE=gemini-3.8-flash-high python scripts/disc_elo.py [workers] [systems, comma-separated]
Output: runs_disc/pairwise_games.json, runs_disc/elo.{json,md}
"""
import json, os, sys
from pathlib import Path
os.environ.setdefault("VMB_RUNS", "runs_disc"); os.environ.setdefault("VMB_TASKS", "tasks_main90")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vmb import ROOT
from vmb.aggregate import collect, pairwise
from vmb.elo import bootstrap_by_task

rows = [r for r in collect("bare") if not r["system"].startswith("tpl")]
systems = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else {r["system"] for r in rows}
games = pairwise(rows, systems, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 8)
elo = bootstrap_by_task(games, B=1000, prior=1.0)
agree = sum(1 for g in games if (g[4] - 0.5) * (g[5] - 0.5) > 0 or g[4] == g[5] == 0.5) / max(1, len(games))
out = ROOT / os.environ["VMB_RUNS"]
json.dump({"judge": os.environ.get("VMB_PAIR_JUDGE"), "n_games": len(games), "swap_agree": agree, "elo": elo},
          open(out / "elo.json", "w"), indent=1)
md = [f"judge {os.environ.get('VMB_PAIR_JUDGE')}, {len(games)} games, A/B swap agreement {agree:.2f}", "",
      "| system | Elo | 95% CI |", "|---|---|---|"]
md += [f"| {k} | {v['elo']:.0f} | [{v['lo']:.0f}, {v['hi']:.0f}] |" for k, v in sorted(elo.items(), key=lambda kv: -kv[1]["elo"])]
(out / "elo.md").write_text("\n".join(md) + "\n"); print("\n".join(md))
