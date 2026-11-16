"""Does a metric separate systems? Within-song permutation test of the system effect (eta^2 after removing song means)."""
from collections import defaultdict

import numpy as np


def eta2(rows, metric, systems):
    xs = [(r["system"], r["task"], r[metric]) for r in rows
          if r["system"] in systems and isinstance(r.get(metric), (int, float)) and not np.isnan(r[metric])]
    if len(xs) < 4:
        return None, xs
    tm = defaultdict(list)
    for _, t, v in xs:
        tm[t].append(v)
    res = [(s, v - np.mean(tm[t])) for s, t, v in xs]
    tot = sum(v * v for _, v in res)
    sm = defaultdict(list)
    for s, v in res:
        sm[s].append(v)
    return (sum(len(v) * np.mean(v) ** 2 for v in sm.values()) / tot if tot > 0 else None), xs


def perm_test(rows, metric, systems, n=2000, seed=0):
    """Returns (eta2, null mean, p). System labels are permuted within each song."""
    e, xs = eta2(rows, metric, systems)
    if e is None:
        return None
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for i, (s, t, v) in enumerate(xs):
        by[t].append(i)
    null = []
    for _ in range(n):
        lab = [s for s, _, _ in xs]
        for idx in by.values():
            perm = rng.permutation([lab[i] for i in idx])
            for i, p in zip(idx, perm):
                lab[i] = p
        fake = [{"system": l, "task": t, metric: v} for l, (_, t, v) in zip(lab, xs)]
        null.append(eta2(fake, metric, systems)[0])
    null = np.array([x for x in null if x is not None])
    return {"eta2": e, "null_mean": float(null.mean()), "p": float((np.sum(null >= e) + 1) / (len(null) + 1)), "n": len(xs)}
