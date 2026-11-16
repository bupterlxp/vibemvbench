"""Bradley-Terry strengths on an Elo scale with bootstrap CIs and anchored fitting.

Outcomes may be graded (score of a in [0, 1]); each game contributes fractional wins. `prior` adds that many virtual
ties against a fixed 1000-rated reference player, which keeps ratings finite when a system wins (or loses) every game.
"""
import numpy as np

REF = "__prior__"


def fit_bt(games, players=None, iters=2000, anchors=None, prior=0.0):
    players = list(players or sorted({g[0] for g in games} | {g[1] for g in games}))
    allp = players + ([REF] if prior else [])
    idx = {p: i for i, p in enumerate(allp)}; n = len(allp)
    W = np.zeros((n, n))
    for g in games:
        a, b, o = g[0], g[1], g[2]
        W[idx[a], idx[b]] += o; W[idx[b], idx[a]] += 1 - o
    if prior:
        r = idx[REF]
        for p in players:
            W[idx[p], r] += prior / 2; W[r, idx[p]] += prior / 2
    N = W + W.T
    fixed = {idx[REF]: 1.0} if prior else {}
    for k, v in (anchors or {}).items():
        if k in idx:
            fixed[idx[k]] = 10 ** ((v - 1000) / 400)
    p = np.ones(n)
    for i, v in fixed.items():
        p[i] = v
    for _ in range(iters):
        wins = W.sum(1)
        denom = (N / (p[:, None] + p[None, :])).sum(1)
        newp = np.where(denom > 0, wins / np.maximum(denom, 1e-12), p)
        newp = np.maximum(newp, 1e-6)
        for i, v in fixed.items():
            newp[i] = v
        if not fixed:
            newp /= np.exp(np.mean(np.log(newp)))
        if np.allclose(newp, p, rtol=1e-9):
            p = newp; break
        p = newp
    return {pl: float(1000 + 400 * np.log10(p[idx[pl]])) for pl in players}


def bootstrap(games, B=2000, seed=0, **kw):
    rng = np.random.default_rng(seed)
    base = fit_bt(games, **kw)
    samples = {k: [] for k in base}
    games = list(games)
    for _ in range(B):
        g = [games[i] for i in rng.integers(0, len(games), len(games))]
        r = fit_bt(g, players=list(base), **kw)
        for k in base:
            samples[k].append(r[k])
    return {k: {"elo": base[k], "lo": float(np.percentile(samples[k], 2.5)), "hi": float(np.percentile(samples[k], 97.5))}
            for k in base}


def bootstrap_by_task(games, B=2000, seed=0, **kw):
    """Song-clustered bootstrap: resample whole songs (all games of a song together), not individual games.
    games: (a, b, outcome, task, ...)."""
    rng = np.random.default_rng(seed)
    base = fit_bt(games, **kw)
    by = {}
    for g in games:
        by.setdefault(g[3], []).append(g)
    tasks = list(by)
    samples = {k: [] for k in base}
    for _ in range(B):
        g = [x for t in rng.choice(tasks, len(tasks)) for x in by[t]]
        r = fit_bt(g, players=list(base), **kw)
        for k in base:
            samples[k].append(r[k])
    return {k: {"elo": base[k], "lo": float(np.percentile(samples[k], 2.5)), "hi": float(np.percentile(samples[k], 97.5))}
            for k in base}
