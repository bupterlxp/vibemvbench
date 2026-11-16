"""Tier A cleanup: in FINISHED runs (non-empty result.json, no lock) under runs_disc/ and runs_hear/, delete files that are
regenerable intermediates. Never touches final.mp4, timing.json, notes.md, project/ source files (*.py *.js *.html *.json
*.md *.sh *.txt *.css), trajectory.jsonl, stderr.log, result.json, score.json, ocr_cache.json, or anything in /task.
Deletes: pip/npm payloads (*.whl *.so *.so.* *.tar.gz *.zip in pydeps/site-packages/tmp dirs), HF partial downloads
(*.incomplete, sha-named blobs), feature caches (*.npy *.npz *.f32 *.raw), audio stems (*.wav *.flac *.mp3 inside work/,
regenerable from /task/audio.wav), intermediate videos (*.mp4 *.mov *.mkv *.webm other than out/final.mp4), and image
sequences (dirs with >300 images). Dry run unless --apply. Logs every path to PRUNE_LOG."""
import os, re, sys, json
APPLY = "--apply" in sys.argv; roots = [a for a in sys.argv[1:] if not a.startswith("--")]
KEEP_EXT = (".py", ".js", ".mjs", ".ts", ".html", ".css", ".json", ".md", ".sh", ".txt", ".glsl", ".frag", ".vert", ".ttf", ".otf", ".svg")
KEEP_NAMES = {"final.mp4", "timing.json", "notes.md"}
DEL_EXT = (".whl", ".npy", ".npz", ".f32", ".raw", ".incomplete", ".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg",
           ".mp4", ".mov", ".mkv", ".webm", ".avi", ".y4m", ".bin", ".safetensors", ".pt", ".pth", ".ckpt", ".onnx", ".gguf", ".th", ".h5")
SHA = re.compile(r"^[0-9a-f]{40,64}$")
log = open(os.environ.get("PRUNE_LOG", "/plog/prune_a.txt"), "a"); freed = 0; n = 0
def kill(p, sz):
    global freed, n
    freed += sz; n += 1; log.write(f"{sz}\t{p}\n")
    if APPLY:
        try: os.remove(p)
        except OSError: pass
for root in roots:
    for sysd in sorted(os.listdir(root)):
        sd = os.path.join(root, sysd)
        if not os.path.isdir(sd): continue
        for run in sorted(os.listdir(sd)):
            rd = os.path.join(sd, run)
            if run.endswith(".lock") or os.path.exists(rd + ".lock") or not os.path.isdir(rd): continue
            rj = os.path.join(rd, "result.json")
            if not os.path.exists(rj) or os.path.getsize(rj) == 0: continue
            w = os.path.join(rd, "work")
            if not os.path.isdir(w): continue
            for a, ds, fs in os.walk(w):
                imgs = [f for f in fs if f.lower().endswith((".png", ".jpg", ".jpeg", ".ppm", ".bmp", ".tga"))]
                seq = len(imgs) > 300
                for f in fs:
                    p = os.path.join(a, f)
                    try: sz = os.lstat(p).st_size
                    except OSError: continue
                    if os.path.islink(p): continue
                    if f in KEEP_NAMES: continue
                    low = f.lower()
                    if low.endswith(KEEP_EXT): continue
                    if seq and f in imgs: kill(p, sz); continue
                    if low.endswith(DEL_EXT) or ".so." in low or low.endswith(".so") or SHA.match(f): kill(p, sz); continue
                    if any(x in a for x in ("/pydeps", "/site-packages", "/node_modules", "/.cache", "/pip-", "/huggingface")) and sz > 1e5:
                        kill(p, sz)
print(("APPLIED" if APPLY else "DRY RUN"), f"{n} files, {freed/1e9:.1f} GB")
