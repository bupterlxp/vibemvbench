"""CLIP frame embeddings: per-video style vector (for cross-system diversity) and a semantic section-response score."""
import os

import numpy as np

from .media import read_frames

_M = None


def _model():
    global _M
    if _M is None:
        pass  # set HF_ENDPOINT yourself if you need a Hugging Face mirror
        import open_clip
        import torch
        m, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        _M = (m.eval().to(dev), pre, dev)
    return _M


def embed_video(video, fps=2):
    import torch
    from PIL import Image
    m, pre, dev = _model()
    t, fr = read_frames(video, fps=fps, width=336)
    with torch.no_grad():
        x = torch.stack([pre(Image.fromarray(f)) for f in fr]).to(dev)
        e = m.encode_image(x).float()
        e = e / e.norm(dim=-1, keepdim=True)
    return t, e.cpu().numpy()


def clip_metrics(video, ref, win=2.0, cache=None):
    if cache is not None and os.path.exists(cache):
        z = np.load(cache); t, e = z["t"], z["e"]
    else:
        t, e = embed_video(video)
        if cache is not None:
            np.savez(cache, t=t, e=e)
    out = {"clip_mean": e.mean(0)}
    # semantic diversity within the video: mean pairwise cosine distance of frames
    sim = e @ e.T
    out["visual_variety"] = float(1 - sim[np.triu_indices(len(e), 1)].mean()) if len(e) > 1 else 0.0

    def contrast(x):
        a = e[(t >= x - win) & (t < x)]; b = e[(t >= x) & (t < x + win)]
        if len(a) == 0 or len(b) == 0:
            return np.nan
        ma, mb = a.mean(0), b.mean(0)
        return float(1 - ma @ mb / (np.linalg.norm(ma) * np.linalg.norm(mb)))

    cb = [contrast(b) for b in ref["section_boundaries"]]
    others = [d for d in ref["downbeats"] if all(abs(d - b) > win for b in ref["section_boundaries"])]
    co = [contrast(d) for d in others]
    cb = [x for x in cb if not np.isnan(x)]; co = [x for x in co if not np.isnan(x)]
    out["clip_section_response"] = float(np.mean(cb) / (np.median(co) + 1e-4)) if cb and co else None
    return out
