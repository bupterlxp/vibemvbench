"""L1 program checks: decodable, spec (resolution, fps, duration), original audio present and aligned."""
import numpy as np
from scipy.signal import correlate

from .media import probe, read_audio


def audio_offset(delivered, reference, sr, max_lag_s=1.0):
    """Lag (s) of delivered vs reference via onset-envelope xcorr, plus normalised peak and level ratio."""
    import librosa
    n = min(len(delivered), len(reference))
    if n < sr:
        return None, 0.0
    a = librosa.onset.onset_strength(y=delivered[:n], sr=sr, hop_length=128)
    b = librosa.onset.onset_strength(y=reference[:n], sr=sr, hop_length=128)
    a = (a - a.mean()) / (a.std() + 1e-9); b = (b - b.mean()) / (b.std() + 1e-9)
    c = correlate(a, b, mode="full") / len(a)
    lags = np.arange(-len(b) + 1, len(a))
    m = np.abs(lags) <= int(max_lag_s * sr / 128)
    k = np.argmax(c[m])
    return float(lags[m][k] * 128 / sr), float(c[m][k])


def l1_checks(video, spec, ref_audio_path=None):
    info = probe(video)
    res = {"decodable": info is not None}
    if not info:
        res["pass"] = False
        return res
    vs = [s for s in info["streams"] if s["codec_type"] == "video"]
    as_ = [s for s in info["streams"] if s["codec_type"] == "audio"]
    res["has_video"] = bool(vs); res["has_audio"] = bool(as_)
    if vs:
        v = vs[0]
        num, den = map(float, v.get("avg_frame_rate", "0/1").split("/"))
        fps = num / den if den else 0
        res["resolution_ok"] = (int(v["width"]), int(v["height"])) == tuple(spec["resolution"])
        res["fps"] = round(fps, 3)
        res["fps_ok"] = abs(fps - spec["fps"]) <= 0.01
    dur = float(info["format"].get("duration", 0))
    res["duration"] = round(dur, 3)
    res["duration_ok"] = abs(dur - spec["duration"]) <= spec.get("duration_tol", 0.1)
    if as_ and ref_audio_path:
        import librosa
        y, sr = read_audio(video)
        r, _ = librosa.load(ref_audio_path, sr=sr, mono=True)
        lag, peak = audio_offset(y, r, sr)
        res["audio_offset_s"] = None if lag is None else round(lag, 3)
        res["audio_match"] = round(peak, 3)
        res["audio_ok"] = lag is not None and abs(lag) <= 0.04 and peak >= 0.5
    res["pass"] = all(res.get(k, False) for k in ["decodable", "has_video", "has_audio", "resolution_ok", "fps_ok",
                                                   "duration_ok"]) and res.get("audio_ok", True)
    return res
