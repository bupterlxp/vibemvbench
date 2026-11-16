# Vibe MV-Bench

Code, tasks, annotations and scores for **"The Deaf Director: Can a Coding Agent Hear Its Way to a Music Video?"**

A coding agent receives only a song (a 90 s Creative Commons clip, its title and artist, and the sentence
"Make a music video for this song") inside a sandbox, and must deliver a music video in which every frame is drawn by its own
code. It cannot hear, so it listens through code it writes; alongside the video it hands in a *timing sheet* of what it
believes it heard, which makes listening measurable on its own.

## Contents

| Path | What it is |
|---|---|
| `vmb/` | Benchmark package: task building, sandboxed rollouts (`rollout.py`), metrics (`eval/`), pairwise judge, Elo |
| `sandbox/` | Docker images the agents run in, `ENVIRONMENT.md` and `CRITERIA.md` exactly as the agents read them |
| `scripts/` | Batch drivers and every analysis in the paper (arena, swapped ears, retrieval ablation, hearing test, habits) |
| `tasks/<id>/` | The 73 tasks: `task.json` (prompt and spec), `reference.json` (beats, downbeats, sections, timed lyrics), `rubric.json` (hidden song profile and anchors used only by the judge) |
| `data/` | Song list with sources and licences, difficulty features and tiers, reference annotations, list prices used for cost |
| `hearing_test/` | Ground truth of the 20 synthetic hearing-test stimuli (`python -m vmb.hearing_test build` regenerates the audio) |
| `results/` | Arena games and Elo, per-dimension Elo, style labels, swapped-ears and native-ears results, human-check annotations |
| `tools/` | Annotation pages used for the beat review, profile review and blind pairwise study |

Audio clips, the agents' videos and full trajectories are large and are released separately (link to follow).

## Songs and licences

73 songs: 40 from JamendoLyrics MultiLang (with human line- and word-level lyric timings) and 33 from the Free Music Archive,
all under Creative Commons licences that permit derivatives. `data/songs.json` lists each song's source URL and licence;
52 carry a non-commercial clause, so the benchmark is for research use.

## Running a system

```bash
cp env.example .env        # fill in the endpoints and keys for the systems you run
docker build -t vmb-sandbox:base-v2 -f sandbox/Dockerfile sandbox/
export VMB_TASKS=tasks VMB_RUNS=runs VMB_DISCLOSE=1
python scripts/disclosed_batch.py 4 main codex:<model>:max      # Codex CLI systems
python scripts/disclosed_batch.py 4 main claude:<model>:max     # Claude Code systems
VMB_NO_JUDGE=1 python scripts/score_all.py bare                  # objective metrics
VMB_PAIR_JUDGE=<judge model> python scripts/disc_elo.py 8        # pairwise arena
```

Each run writes `runs/bare/<system>/<task>/` with `result.json`, the full trajectory, and the agent's `work/out/`
(`final.mp4`, `project/`, `timing.json`, `notes.md`).

## Licence

Code: MIT. Annotations and results: CC BY 4.0. Audio remains under each song's own licence (see `data/songs.json`).
