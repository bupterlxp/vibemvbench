"""How well the agent 'heard' the music: its self-reported timing sheet vs reference."""
import numpy as np


def f_measure(ref, est, tol=0.07):
    ref, est = np.asarray(ref, float), np.asarray(est, float)
    if len(ref) == 0 or len(est) == 0:
        return 0.0
    used = np.zeros(len(est), bool); tp = 0
    for r in ref:
        d = np.abs(est - r); d[used] = np.inf
        k = int(np.argmin(d))
        if d[k] <= tol:
            used[k] = True; tp += 1
    p, rc = tp / len(est), tp / len(ref)
    return 0.0 if tp == 0 else 2 * p * rc / (p + rc)


def listening_metrics(sheet, clip_ann):
    out = {}
    if sheet.get("beats"):
        out["listen_beat_f"] = f_measure(clip_ann["beats"], sheet["beats"])
        est = np.asarray(sheet["beats"], float)
        if len(est) > 1:
            out["listen_bpm_err"] = abs(60 / np.median(np.diff(est)) - clip_ann["bpm"]) / clip_ann["bpm"]
    if sheet.get("downbeats"):
        out["listen_downbeat_f"] = f_measure(clip_ann["downbeats"], sheet["downbeats"])
    if sheet.get("sections") is not None and clip_ann["section_boundaries"]:
        out["listen_section_f"] = f_measure(clip_ann["section_boundaries"], sheet["sections"], tol=1.0)
    if sheet.get("lyrics") and clip_ann["lyrics_lines"]:
        errs = []
        for l in clip_ann["lyrics_lines"]:
            if not sheet["lyrics"]:
                break
            errs.append(min(abs(x.get("start", 1e9) - l["start"]) for x in sheet["lyrics"]))
        out["listen_lyric_onset_err_med"] = float(np.median(errs)) if errs else None
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()}
