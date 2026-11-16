# Environment

You are running fully autonomously inside a Linux container. Nobody will answer questions; do not wait for input.
This is a single non-interactive session: nobody will resume or notify you later. Do not end your turn while waiting for
background jobs; wait for or poll them yourself, and finish only when the deliverables are written.

* Working directory: `/work` (writable). Task inputs are in `/task` (read-only):
  `prompt.txt` (the brief), `audio.wav` (the music clip), and `lyrics.txt` when the task provides lyrics.
{budget_line}
* Resources: {cpus} CPU cores and {mem} of memory.
* Available: Python 3 (numpy, scipy, Pillow, pycairo, opencv, moviepy), Node 22, headless Chromium (`/usr/bin/chromium`),
  ffmpeg/ffprobe, ImageMagick, sox, Noto/DejaVu fonts (including CJK). You may install additional packages with pip/npm.
{tools_note}
* There is NO image, video, or music generation model. Every frame must be produced by your code.

## Deliverables (all in `/work/out/`)
1. `final.mp4` — H.264 video + AAC audio, **{w}x{h}, {fps} fps, duration {dur:.3f} s (±0.1 s)**. Its soundtrack must be
   `/task/audio.wav` exactly: not re-timed, trimmed, pitched, or mixed with other sound.
2. `project/` — the source code that renders the video (it should re-render `final.mp4` when run).
3. `timing.json` — what you believe the music contains, in seconds from the clip start:
   `{{"bpm": number|null, "beats": [...], "downbeats": [...], "sections": [boundary times], "lyrics": [{{"start": s, "end": s, "text": "..."}}]}}`
   Use empty lists for anything you did not determine. Report honestly; this is scored against reference annotations.
4. `notes.md` — a short production note: what you heard in the song, the MV style you chose (e.g. narrative, performance,
   rhythmic montage, abstract, lyric video) and why, how you aligned visuals to the music, and honest verification results.
