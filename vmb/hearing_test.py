"""Hearing test ("audiogram") for code-mediated listening: synthetic stimuli with exact ground truth.

Each stimulus is a 20-30 s WAV made from synthesised kick / snare / hat / bass / pad voices, so the timing is known to the
sample. Stimuli vary one factor at a time from an easy baseline (120 BPM, 4/4, steady, full kit):
  tempo        60, 90, 160, 200 BPM
  drift        accelerando 100->140, ritardando 140->100
  feel         swing (2:1), syncopated kick, half-time (backbeat at 2x)
  metre        3/4, 5/4, 7/8 (accent on each bar start)
  sparse       no kick (hats + pad only), pad only with soft pulse, kick drops out for 8 s
  structure    section change (texture + key) at a known time, stop-time break (2 s of silence) then resume
  words        a synthetic voice says a known word at known times (espeak-ng) over the beat
Ground truth per stimulus: beats, downbeats, section boundaries, word onsets.

  python -m vmb.hearing_test build          -> hearing_test/<id>/{audio.wav,truth.json}
  python -m vmb.hearing_test score <runs>   -> per-system beat F / tempo error / section F / word onset error per stimulus
"""
import json, math, os, subprocess, sys
from pathlib import Path
import numpy as np

SR = 44100
OUT = Path(__file__).resolve().parent.parent / "hearing_test"


def env(n, attack=0.002, decay=0.15):
    t = np.arange(n) / SR
    return np.minimum(t / attack, 1) * np.exp(-t / decay)


def kick(n=int(.35 * SR)):
    t = np.arange(n) / SR; f = 50 + 120 * np.exp(-t * 30)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, .001, .12)


def snare(n=int(.25 * SR), rng=np.random.default_rng(0)):
    t = np.arange(n) / SR
    return (.6 * rng.standard_normal(n) * env(n, .001, .07) + .4 * np.sin(2 * np.pi * 190 * t) * env(n, .001, .05))


def hat(n=int(.06 * SR), rng=np.random.default_rng(1)):
    x = rng.standard_normal(n); x = np.diff(np.concatenate([[0], x]))   # crude high-pass
    return .25 * x * env(n, .0005, .015)


def tone(freq, dur, amp=.15, decay=.6):
    n = int(dur * SR); t = np.arange(n) / SR
    return amp * (np.sin(2 * np.pi * freq * t) + .3 * np.sin(4 * np.pi * freq * t)) * env(n, .01, decay)


def place(buf, x, t):
    i = int(round(t * SR))
    if i >= len(buf): return
    j = min(len(buf), i + len(x)); buf[i:j] += x[:j - i]


def grid(dur, bpm_at, meter=4):
    """Beat times for a (possibly time-varying) tempo function bpm_at(t)."""
    beats, t = [], .5
    while t < dur - .3:
        beats.append(round(t, 4)); t += 60 / bpm_at(t)
    return beats, beats[::meter]


def render(spec):
    dur = spec.get("dur", 24.0); meter = spec.get("meter", 4)
    bpm = spec.get("bpm", 120); drift = spec.get("drift")
    bpm_at = (lambda t: drift[0] + (drift[1] - drift[0]) * t / dur) if drift else (lambda t: bpm)
    beats, downs = grid(dur, bpm_at, meter)
    buf = np.zeros(int(dur * SR) + SR)
    rest = spec.get("rest", [])          # [(start, end)] intervals with no sound at all
    kick_off = spec.get("kick_off", [])  # [(start, end)] intervals without kick
    silent = lambda t, iv: any(a <= t < b for a, b in iv)
    swing = spec.get("swing", False); synco = spec.get("synco", False); half = spec.get("half", False)
    no_kick = spec.get("no_kick", False); pad_only = spec.get("pad_only", False)
    key_change = spec.get("section_at")
    for k, b in enumerate(beats):
        if silent(b, rest): continue
        ib = 60 / bpm_at(b); pos = k % meter
        root = 55.0 if (key_change is None or b < key_change) else 55.0 * 2 ** (5 / 12)
        if pad_only:
            place(buf, tone(root * 2, ib * .95, .12, ib), b)        # soft pulse on every beat, no drums
            continue
        if not no_kick and not silent(b, kick_off):
            if synco and pos in (1, 3):
                place(buf, kick(), b + ib / 2)                       # kick on the "and" instead of the beat
            elif not (half and pos == 2):
                place(buf, kick() * (1.0 if pos == 0 else .8), b)
        snare_pos = (2,) if half else ((1, 3) if meter == 4 else (meter - 1,) if meter > 3 else ())
        if pos in snare_pos: place(buf, snare(), b)
        place(buf, hat() * (1.3 if pos == 0 else 1), b)
        off = ib * (2 / 3 if swing else 1 / 2)
        place(buf, hat() * .6, b + off)
        if pos == 0: place(buf, tone(root, ib * meter * .9, .18, ib * meter), b)
        if key_change is not None and b >= key_change:
            place(buf, tone(root * 4, ib * .5, .05, .2), b + ib / 4)   # new texture after the section change
    words = []
    for w, t in spec.get("words", []):
        wav = OUT / "_tmp_word.wav"
        subprocess.run(["espeak-ng", "-v", "en-us", "-s", "150", "-w", str(wav), w], check=True, capture_output=True)
        import soundfile as sf
        x, sr = sf.read(wav); x = x if x.ndim == 1 else x.mean(1)
        if sr != SR: x = np.interp(np.arange(0, len(x), sr / SR), np.arange(len(x)), x)
        on = np.argmax(np.abs(x) > .02 * np.abs(x).max()) / SR   # speech onset inside the synthesised file
        place(buf, .5 * x / np.abs(x).max(), t - on); words.append({"start": t, "text": w})
    buf = buf[:int(dur * SR)]; buf = .9 * buf / np.abs(buf).max()
    keep = lambda ts: [t for t in ts if not silent(t, rest)]
    beats_k = beats if pad_only else keep(beats)
    truth = {"bpm": None if drift else bpm, "beats": beats_k, "downbeats": keep(downs), "meter": meter,
             "sections": [key_change] if key_change else [b for iv in rest for b in iv], "lyrics": words, "spec": spec}
    return buf, truth


STIMULI = {
    "t120_base": {}, "t060": {"bpm": 60}, "t090": {"bpm": 90}, "t160": {"bpm": 160}, "t200": {"bpm": 200},
    "accel": {"drift": [100, 140]}, "rit": {"drift": [140, 100]},
    "swing": {"swing": True}, "synco": {"synco": True}, "halftime": {"half": True, "bpm": 140},
    "m34": {"meter": 3}, "m54": {"meter": 5}, "m78": {"meter": 7, "bpm": 210},
    "nokick": {"no_kick": True}, "padonly": {"pad_only": True}, "kickdrop": {"kick_off": [[8, 16]]},
    "section": {"section_at": 12.0}, "stoptime": {"rest": [[11.0, 13.0]]},
    "words": {"words": [["river", 4.0], ["golden", 9.5], ["tomorrow", 15.25], ["echo", 20.0]]},
    "words_fast": {"bpm": 150, "words": [["ninety", 3.2], ["seven", 7.7], ["lantern", 12.3], ["orbit", 16.6], ["velvet", 21.1]]},
}


def build():
    import soundfile as sf
    for sid, spec in STIMULI.items():
        d = OUT / sid; d.mkdir(parents=True, exist_ok=True)
        buf, truth = render(spec)
        sf.write(d / "audio.wav", buf, SR, subtype="PCM_16")
        json.dump(truth, open(d / "truth.json", "w"), indent=1)
    (OUT / "_tmp_word.wav").unlink(missing_ok=True)
    print(len(STIMULI), "stimuli in", OUT)


def score(runs_dir):
    from .eval.listening import f_measure
    rows = []
    for rd in sorted(Path(runs_dir).glob("*/*")):
        sid = rd.name; tp = rd / "work" / "out" / "timing.json"
        truth = OUT / sid / "truth.json"
        if not truth.exists(): continue
        t = json.load(open(truth))
        res = rd / "result.json"
        if not res.exists(): continue                      # still running
        rj = json.load(open(res))
        if rj.get("infra_failure") and not tp.exists(): continue
        try: s = json.load(open(tp))
        except Exception: s = {}
        r = {"system": rd.parent.name, "stimulus": sid, "delivered": bool(s)}
        if s.get("beats"): r["beat_f"] = round(f_measure(t["beats"], s["beats"]), 3)
        else: r["beat_f"] = 0.0
        if t["downbeats"] and s.get("downbeats"): r["downbeat_f"] = round(f_measure(t["downbeats"], s["downbeats"]), 3)
        if t["bpm"] and s.get("bpm"): r["bpm_rel_err"] = round(abs(s["bpm"] - t["bpm"]) / t["bpm"], 3)
        if t["sections"]:
            r["section_f"] = round(f_measure(t["sections"], [x for x in (s.get("sections") or []) if 0.5 < x < t["spec"].get("dur", 24) - .5], tol=1.0), 3)
        if t["lyrics"]:
            est = s.get("lyrics") or []
            errs = [min((abs(x.get("start", 1e9) - w["start"]) for x in est if w["text"].lower() in str(x.get("text", "")).lower()), default=None) for w in t["lyrics"]]
            r["word_found"] = round(sum(e is not None for e in errs) / len(errs), 2)
            got = [e for e in errs if e is not None]
            r["word_onset_err"] = round(float(np.median(got)), 3) if got else None
        rows.append(r)
    out = Path(runs_dir) / "hearing_scores.json"; json.dump(rows, open(out, "w"), indent=1)
    by = {}
    for r in rows: by.setdefault(r["system"], {})[r["stimulus"]] = r["beat_f"]
    ids = list(STIMULI)
    print("system".ljust(28) + " ".join(s[:8].rjust(8) for s in ids))
    for sysn, v in by.items():
        print(sysn[:28].ljust(28) + " ".join(f"{v[s]:8.2f}" if s in v else "       ." for s in ids))


if __name__ == "__main__":
    {"build": build, "score": lambda: score(sys.argv[2])}[sys.argv[1]]()
