"""Build benchmark tasks: (song clip, brief category, delivery spec, rubric). Briefs and rubrics are drafted by an
LLM from clip metadata (never from beat times, which the agent must find itself) and then reviewed."""
import json
import subprocess
from pathlib import Path

from . import DATA, FFMPEG, ROOT
from .clips import clip_annotation, pick_clip
from .llm import GEMINI_PRO, ask_av, audio_as_mp4

TASKS = ROOT / "tasks"
CATEGORIES = {
    "lyric": "Lyric video / kinetic typography: the sung words are the main visual material.",
    "visualizer": "Audio-reactive visualiser: abstract generative visuals driven by rhythm, energy and timbre.",
    "abstract": "Abstract or narrative mood piece: visuals carry the emotion and arc of the music, minimal text.",
    "dance": "Character / object choreography: a drawn character, creature or set of objects moves to the music.",
    "edit": "Rhythmic montage: a sequence of distinct code-rendered scenes cut together with hard editorial constraints.",
}
SPEC = {"resolution": [1920, 1080], "fps": 30, "duration_tol": 0.1}
DIMENSIONS = {
    "MF": "Music fit: visual style, palette and pacing match the genre, tempo and character of this music.",
    "EM": "Emotion: the visuals convey the mood of the music and follow its emotional arc.",
    "RH": "Rhythm: visual events (cuts, hits, motion accents) feel locked to the beat and phrasing, not mechanical.",
    "ST": "Structure: the visuals respond when the music changes section (e.g. verse to chorus, drop, breakdown).",
    "LY": "Lyrics: when lyrics are shown or referenced, they are correct, readable and timed to the singing (N/A if none required).",
    "VA": "Visual quality and aesthetics: clean rendering, composition, typography, no glitches or placeholder look.",
    "CR": "Creativity: a distinctive idea beyond a template; surprising yet appropriate choices.",
}

BRIEF_PROMPT = """LISTEN to the attached clip first (the picture is black; only its audio matters). You are writing one task for Vibe MV-Bench, a benchmark in which an autonomous coding agent must make a
music video for a clip of a real song ENTIRELY WITH CODE (no image/video generation models; programmatic 2D/3D
rendering, typography, shaders, procedural animation). The finished MP4 must use the provided audio clip unchanged.

Category: {cat} -- {cat_desc}
Song: "{title}" by {artist}; genre: {genre}; language: {lang}; tempo about {bpm:.0f} BPM.
Clip: {dur:.0f} s long; the clip contains at least one section change (the agent must find where).
{lyr}

Write a brief (120-220 words, English, second person) that a real creator could have sent: a concrete visual concept
fitted to what you HEAR in this clip (instrumentation, vocal delivery, energy arc, the moment it changes) and its lyrics, and 3-5 CHECKABLE requirements that a viewer could verify from the video alone.
Requirements must concern the music-video relationship (e.g. hard cuts land on beats, a visible change at the section
change, sung lines appear while sung, motion intensity follows loudness), plus at most one visual-content requirement.
Never state beat times, BPM values to use, or section timestamps -- discovering them is part of the task.
Do not mention tools, libraries, or file formats. Leave real creative freedom.

Return JSON: {{"title": short task title, "listening_notes": what you heard (instruments, vocals, mood, energy arc, where and
how it changes, approximate seconds), "brief": text, "requirements": [list of the checkable requirements, each one sentence]}}"""

RUBRIC_PROMPT = """LISTEN to the attached clip (black picture; audio only). You write the evaluation rubric for one Vibe MV-Bench task. Judges will watch the delivered music video
(with audio) and never see the agent's code or notes. Task brief:
---
{brief}
---
Ground truth about the clip (judges will receive these facts too): {facts}
Ground anchors in what is audible (e.g. "the drums drop out", "the vocal turns to a shout") as well as in these facts.

1) "checks": 10-18 atomic pass/fail checks. Each quotes the brief text it enforces ("basis"), is "core" or "detail",
   has "modality" in [visual, audio_visual, text] and is decidable from the video. Phrase checks so that claims
   shown on screen (e.g. a caption saying "synced to beat") are NOT evidence; only observed behaviour counts.
2) "anchors": for each quality dimension below, one sentence each describing what a 1, 3 and 5 look like FOR THIS TASK.
   Dimensions: {dims}
   For LY use "N/A" anchors if the task neither requires nor naturally involves lyrics.

Return JSON: {{"checks": [{{"id": "C1", "check": ..., "basis": ..., "level": "core"|"detail", "modality": ...}}],
"anchors": {{"MF": {{"1": ..., "3": ..., "5": ...}}, ...}}}}"""


def _assign(songs):
    """Deterministic category/length assignment balancing categories over sources."""
    order = ["lyric", "visualizer", "abstract", "dance", "edit"]
    lengths = [30, 30, 15, 45, 30]
    plan, i = [], 0
    for s in songs:
        has_lyr = s.get("n_lyric_lines", 0) > 0
        cats = order if has_lyr else [c for c in order if c != "lyric"]
        cat = cats[i % len(cats)]
        plan.append((s, cat, lengths[i % len(lengths)])); i += 1
    return plan


def build(model=GEMINI_PRO, limit=None):
    songs = json.load(open(DATA / "songs.json"))
    TASKS.mkdir(exist_ok=True)
    out = []
    for s, cat, L in _assign(songs)[:limit]:
        ann = json.load(open(DATA / "annotations" / f"{s['song_id']}.json"))
        c = pick_clip(ann, L)
        if c is None:
            continue
        st, en = c[0], c[1]
        ca = clip_annotation(ann, st, en)
        tid = f"{s['song_id']}_{cat}_{L}"
        d = TASKS / tid; d.mkdir(exist_ok=True)
        clip_wav = d / "audio.wav"
        if not clip_wav.exists():
            subprocess.run([FFMPEG, "-v", "error", "-y", "-ss", f"{st:.3f}", "-t", f"{en - st:.3f}", "-i",
                            str(DATA / "audio" / f"{s['song_id']}.wav"), "-c:a", "pcm_s16le", str(clip_wav)], check=True)
        lyr_lines = [l["text"] for l in ca["lyrics_lines"]]
        lyr = ("Sung lines in this clip: " + " / ".join(lyr_lines)) if lyr_lines else "The clip has no sung lyrics (or none annotated)."
        listen = d / "listen.mp4"
        if not listen.exists():
            audio_as_mp4(clip_wav, listen)
        bp = d / "brief.json"
        if not bp.exists() or json.load(open(bp)).get("model") != model:
            b = ask_av(model, BRIEF_PROMPT.format(cat=cat, cat_desc=CATEGORIES[cat], title=s["title"], artist=s["artist"],
                                               genre=s["genre"], lang=s.get("language"), bpm=ann["bpm"], dur=en - st, lyr=lyr),
                       media=[listen], json_mode=True)
            b["model"] = model
            json.dump(b, open(bp, "w"), ensure_ascii=False, indent=1)
            (d / "rubric.json").unlink(missing_ok=True)
        b = json.load(open(bp))
        facts = {"clip_seconds": round(en - st, 2), "approx_bpm": round(ann["bpm"]),
                 "section_changes_at_s": ca["section_boundaries"],
                 "sung_lines": [{"start": l["start"], "end": l["end"], "text": l["text"]} for l in ca["lyrics_lines"]]}
        rp = d / "rubric.json"
        if not rp.exists():
            r = ask_av(model, RUBRIC_PROMPT.format(brief=b["brief"], facts=json.dumps(facts, ensure_ascii=False),
                                                   dims=json.dumps(DIMENSIONS)), media=[listen], json_mode=True)
            r["facts"] = facts; r["model"] = model
            json.dump(r, open(rp, "w"), ensure_ascii=False, indent=1)
        spec = dict(SPEC, duration=round(en - st, 3))
        meta = {"task_id": tid, "song_id": s["song_id"], "category": cat, "clip_start": round(st, 3), "clip_end": round(en, 3),
                "length_class": L, "spec": spec, "song": {k: s[k] for k in ("title", "artist", "genre", "language", "license", "url")},
                "lyrics_text": lyr_lines}
        json.dump(meta, open(d / "task.json", "w"), ensure_ascii=False, indent=1)
        json.dump(ca, open(d / "reference.json", "w"), ensure_ascii=False)
        out.append(tid); print(tid, b["title"], flush=True)
    return out
