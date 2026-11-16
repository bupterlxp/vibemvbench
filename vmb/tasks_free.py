"""Task definition v2 ("free"): the input is just a song clip; the agent chooses concept and style itself.

Hidden from the agent: reference annotation and a song profile + song-specific judging anchors written by Gemini
after LISTENING to the clip. Checks are style-agnostic (they must hold for a narrative MV and an abstract one alike).
"""
import json
import subprocess

from . import DATA, FFMPEG, TASKS_DIR
from .clips import clip_annotation, pick_clip
from .llm import GEMINI_PRO, ask_av, audio_as_mp4

PROMPT = "Make a music video for this song. Make it as good as you possibly can."

DIMENSIONS = {
    "UN": "Song understanding: the chosen concept and style (narrative, performance, rhythmic, abstract, lyric, ...) "
          "suit this song's genre, mood, lyrics and energy; the video shows it 'got' the song.",
    "EM": "Emotion: the picture conveys the song's mood and follows its emotional arc.",
    "RH": "Rhythm: visual events (cuts, hits, motion accents) feel locked to the beat and phrasing, without monotony.",
    "ST": "Structure: the picture responds when the music changes (section change, drop, breakdown, build).",
    "LY": "Lyrics: if words are shown or depicted, they are right, readable and timed to the singing; "
          "if the song is instrumental or the video uses no text, judge whether the imagery fits the lyric content (null only if truly N/A).",
    "VA": "Visual quality: clean rendering, composition, typography, colour; no glitches or placeholder look.",
    "CR": "Creativity: a distinctive idea beyond a template visualiser; surprising yet appropriate.",
}

GENERIC_CHECKS = [
    ("G1", "core", "audio_visual", "At each audible section change in the music, the picture changes in a clearly visible way that persists (not a single flash)."),
    ("G2", "core", "audio_visual", "Prominent beats or accents are answered by visual events (cuts, hits, motion accents) for most of the clip."),
    ("G3", "core", "audio_visual", "Visual intensity (motion, brightness, density) rises and falls with the music's energy."),
    ("G4", "detail", "audio_visual", "Visual rhythm is varied: not the same identical pulse on every beat for the whole clip."),
    ("G5", "core", "visual", "The video has a coherent visual identity (consistent palette, style, or world) rather than unrelated stock effects."),
    ("G6", "detail", "visual", "No rendering defects: no tearing, black/blank stretches, placeholder text, or broken frames."),
    ("G7", "detail", "text", "Any on-screen lyrics or words are spelled correctly and match what is sung at that moment."),
    ("G8", "core", "audio_visual", "The concept is recognisably about THIS song (its mood, lyrics or sound), not a generic visual that would fit any track."),
]

PROFILE_PROMPT = """LISTEN to the attached audio (the picture is black; only the audio matters). It is {what} of
"{title}" by {artist} (genre tag: {genre}), {dur:.0f} seconds long.{hints}

You are preparing the HIDDEN evaluation material for a benchmark in which an AI agent receives only this audio clip and
the instruction "Make a music video for this song." and must invent everything itself. Write:

1. "profile": what you hear -- instrumentation, vocals, mood, lyrical theme, and the energy arc.
2. "key_moments": 2-8 moments a strong music video should visibly respond to, each {{"t": seconds, "what": "..."}}.
   {moment_rule}
3. "fitting_styles": which MV approaches would suit this song (e.g. narrative, performance, rhythmic montage, abstract,
   lyric/kinetic type) and why; and approaches that would clash with it.
4. "song_checks": 3-5 extra pass/fail checks that are SPECIFIC TO THIS SONG but STYLE-AGNOSTIC (they must be fair to
   any concept the agent picks), e.g. "the picture marks the drop at ~14 s"; any time you cite must be one of the moments above. Each {{"id": "S1", "level": "core"|"detail",
   "modality": "audio_visual"|"visual"|"text", "check": "..."}}.
5. "anchors": for each dimension below, what scores 1, 3 and 5 look like FOR THIS SONG (one sentence each), written so
   that very different concepts can all score 5. Dimensions: {dims}

Return JSON with keys profile, key_moments, fitting_styles, song_checks, anchors."""


def build(lengths=(30,), song_ids=None, model=GEMINI_PRO, full=False, hint_annotation=True):
    """full=True builds one task per song on the WHOLE song (tid = <song>_full); otherwise one clip per length."""
    songs = json.load(open(DATA / "songs.json"))
    if song_ids:
        songs = [s for s in songs if s["song_id"] in song_ids]
    TASKS_DIR.mkdir(exist_ok=True)
    out = []
    for s in songs:
        v2 = DATA / "annotations_v2" / f"{s['song_id']}.json"   # ensemble beats (v1 had known tracker errors)
        ann = json.load(open(v2 if v2.exists() else DATA / "annotations" / f"{s['song_id']}.json"))
        for L in (["full"] if full else lengths):
            if L == "full":
                st, en = 0.0, float(ann["duration"])
            else:
                c = pick_clip(ann, L)
                if c is None:
                    continue
                st, en = c[0], c[1]
            ca = clip_annotation(ann, st, en)
            tid = f"{s['song_id']}_{L}"; d = TASKS_DIR / tid; d.mkdir(exist_ok=True)
            wav = d / "audio.wav"
            if not wav.exists():
                subprocess.run([FFMPEG, "-v", "error", "-y", "-ss", f"{st:.3f}", "-t", f"{en - st:.3f}", "-i",
                                str(DATA / "audio" / f"{s['song_id']}.wav"), "-c:a", "pcm_s16le", str(wav)], check=True)
            listen = d / "listen.mp4"
            if not listen.exists():
                audio_as_mp4(wav, listen)
            rp = d / "rubric.json"
            if not rp.exists():
                lines = [f'"{l["text"]}"' for l in ca["lyrics_lines"]]
                hints = ""
                timeline = None
                if L == "full" or L >= 60:
                    # Gemini's own timestamps drift on long audio (~1.4-1.5x after the first minute), so the TIMES come from
                    # the annotation + human lyric timings, and Gemini only describes what happens at each of them.
                    timeline = [{"t": round(b, 1), "kind": "section change"} for b in ca["section_boundaries"]]
                    timeline += [{"t": round(l["start"], 1), "kind": f'sung line starts: "{l["text"]}"'} for l in ca["lyrics_lines"]]
                    timeline.sort(key=lambda x: x["t"])
                    hints += (" Timeline of candidate moments (seconds; section changes are automatic and may be imprecise; lyric "
                              "times are human-annotated): " + json.dumps(timeline, ensure_ascii=False))
                if hint_annotation and timeline is None:
                    hints = (f" Reference annotation (may contain errors): tempo ~{ca['bpm']:.0f} BPM; section changes at "
                             f"{ca['section_boundaries']}.")
                if lines:
                    hints += " The sung lyrics, in order (no timing): " + " / ".join(lines)
                rule = ("Take every time from the timeline above (pick the moments that matter; do not invent other times) and "
                        "describe what you hear change there." if timeline is not None else
                        "Use times from what you hear.")
                r = ask_av(model, PROFILE_PROMPT.format(what="the full song" if L == "full" else "a clip", dur=en - st,
                                                        moment_rule=rule,
                                                        title=s["title"], artist=s["artist"], genre=s["genre"], hints=hints,
                                                        dims=json.dumps(DIMENSIONS)), media=[listen], json_mode=True)
                checks = [{"id": i, "level": lv, "modality": m, "check": c} for i, lv, m, c in GENERIC_CHECKS]
                if timeline is not None:  # keep only moments that are on the given timeline and inside the song
                    allowed = [x["t"] for x in timeline]
                    km = []
                    for m in r.get("key_moments", []):
                        try:
                            t = float(m["t"])
                        except Exception:
                            continue
                        if allowed and min(abs(t - a) for a in allowed) <= 1.0 and 0 <= t <= en - st:
                            km.append(m)
                    r["key_moments_dropped"] = len(r.get("key_moments", [])) - len(km); r["key_moments"] = km
                    r["timeline_given"] = timeline
                checks += r.get("song_checks", [])
                r["checks"] = checks; r["model"] = model
                r["facts"] = {"clip_seconds": round(en - st, 2), "approx_bpm": round(ann["bpm"]),
                              "section_changes_at_s": ca["section_boundaries"], "sung_lines": ca["lyrics_lines"]}
                json.dump(r, open(rp, "w"), ensure_ascii=False, indent=1)
            lyr = [l["text"] for l in ca["lyrics_lines"]]
            meta = {"task_id": tid, "song_id": s["song_id"], "clip_start": round(st, 3), "clip_end": round(en, 3),
                    "length_class": L, "difficulty_tier": None, "spec": {"resolution": [1920, 1080], "fps": 30, "duration_tol": 0.1, "duration": round(en - st, 3)},
                    "song": {k: s[k] for k in ("title", "artist", "genre", "language", "license", "url")},
                    "lyrics_text": lyr, "prompt": PROMPT, "free": True}
            json.dump(meta, open(d / "task.json", "w"), ensure_ascii=False, indent=1)
            json.dump(ca, open(d / "reference.json", "w"), ensure_ascii=False)
            out.append(tid); print(tid, s["title"], flush=True)
    return out
