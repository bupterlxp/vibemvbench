"""Template-floor baselines and synthetic controls. All rendered with numpy + ffmpeg, no agent involved.

  uniform_cuts   : colour-field cuts every `every_s` seconds, phase unrelated to the music (lower anchor)
  beat_cuts      : colour-field cuts exactly on every k-th beat (mechanical upper anchor for cut metrics)
  spectrum       : ffmpeg showspectrum/showwaves visualiser (classic template)
  lyric_karaoke  : static background + lyric lines burned in at their reference times
"""
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from . import FFMPEG

W, H, FPS = 1920, 1080, 30
PALETTE = np.array([[230, 57, 70], [241, 250, 238], [168, 218, 220], [69, 123, 157], [29, 53, 87],
                    [255, 183, 3], [251, 133, 0], [33, 158, 188], [142, 202, 230], [2, 48, 71]], np.uint8)


def _encode(frames_iter, n, audio, out, s=0.0, dur=None, size=(W, H)):
    out = str(out); Path(out).parent.mkdir(parents=True, exist_ok=True)
    cmd = [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}", "-r", str(FPS),
           "-i", "-", "-ss", f"{s:.3f}", "-t", f"{dur or n / FPS:.3f}", "-i", str(audio),
           "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "23",
           "-c:a", "aac", "-b:a", "192k", "-shortest", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames_iter:
        p.stdin.write(f.tobytes())
    p.stdin.close(); p.wait()
    return out


def _cut_frames(cut_times, n, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.searchsorted(np.asarray(cut_times) + shift, np.arange(n) / FPS, side="right")
    colours = PALETTE[rng.permutation(len(PALETTE))]
    yy, xx = np.mgrid[0:H, 0:W]
    for i in range(n):
        k = idx[i]
        c = colours[k % len(colours)].astype(np.float32)
        # a slowly drifting gradient so frames are not perfectly static inside a shot
        g = 0.85 + 0.15 * np.sin((xx / W + i / (4 * FPS)) * np.pi)
        yield (c[None, None, :] * g[..., None]).clip(0, 255).astype(np.uint8)


def uniform_cuts(audio, ann, out, s, dur, every_s=1.7, offset=0.37):
    n = int(round(dur * FPS))
    cuts = np.arange(offset, dur, every_s)
    return _encode(_cut_frames(cuts, n), n, audio, out, s, dur)


def beat_cuts(audio, ann, out, s, dur, every=2, shift=0.0):
    n = int(round(dur * FPS))
    cuts = np.asarray(ann["beats"])[::every]
    return _encode(_cut_frames(cuts, n, shift), n, audio, out, s, dur)


def spectrum(audio, ann, out, s, dur):
    out = str(out); Path(out).parent.mkdir(parents=True, exist_ok=True)
    fc = (f"[0:a]showspectrum=s={W}x{H // 2}:mode=combined:color=intensity:slide=scroll,format=yuv420p[sp];"
          f"[0:a]showwaves=s={W}x{H // 2}:mode=cline:colors=white,format=yuv420p[wv];"
          f"[sp][wv]vstack,fps={FPS}[v]")
    subprocess.run([FFMPEG, "-v", "error", "-y", "-ss", f"{s:.3f}", "-t", f"{dur:.3f}", "-i", str(audio),
                    "-filter_complex", fc, "-map", "[v]", "-map", "0:a", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k", out], check=True)
    return out


def _ass_time(x):
    h, r = divmod(max(0, x), 3600); m, sec = divmod(r, 60)
    return f"{int(h)}:{int(m):02d}:{sec:05.2f}"


def lyric_karaoke(audio, ann, out, s, dur, shift=0.0):
    out = str(out); Path(out).parent.mkdir(parents=True, exist_ok=True)
    ass = tempfile.NamedTemporaryFile("w", suffix=".ass", delete=False, encoding="utf-8")
    ass.write("[Script Info]\nScriptType: v4.00+\nPlayResX: %d\nPlayResY: %d\n\n[V4+ Styles]\n"
              "Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BorderStyle, Outline, Alignment\n"
              "Style: L,Arial,54,&H00FFFFFF,&H00000000,1,3,2\n\n[Events]\n"
              "Format: Layer, Start, End, Style, Text\n" % (W, H))
    for l in ann["lyrics_lines"]:
        a, b = l["start"] + shift, l["end"] + shift
        if b > 0 and a < dur:
            ass.write(f"Dialogue: 0,{_ass_time(a)},{_ass_time(min(b, dur))},L,{l['text']}\n")
    ass.close()
    subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=0x1d3557:s={W}x{H}:r={FPS}:d={dur:.3f}",
                    "-ss", f"{s:.3f}", "-t", f"{dur:.3f}", "-i", str(audio), "-vf", f"ass={ass.name}",
                    "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", out],
                   check=True)
    return out
