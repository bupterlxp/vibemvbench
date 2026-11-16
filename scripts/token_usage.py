"""Token usage per rollout from the harness logs, priced with an official list-price table (data/prices.json).

Claude Code stream-json: sum `message.usage` of assistant events, de-duplicated by message id (a message is
streamed in several events). Codex --json: sum `turn.completed.usage`.
"""
import json, sys
from collections import defaultdict
from pathlib import Path


def claude_usage(traj):
    """Exact totals come from `result` events (one per session segment; summed). If the run was killed before any
    result event, fall back to per-message streamed usage, which UNDER-counts output tokens (flag `partial`)."""
    seen, u, results = {}, defaultdict(float), []
    for line in open(traj, errors="ignore"):
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("type") == "assistant":
            m = e.get("message", {})
            if m.get("id") and m.get("usage"):
                seen[m["id"]] = m["usage"]
        elif e.get("type") == "result" and e.get("modelUsage"):
            results.append(e)
    if results:
        for e in results:
            for mu in e["modelUsage"].values():
                u["input"] += mu.get("inputTokens", 0); u["output"] += mu.get("outputTokens", 0)
                u["cache_read"] += mu.get("cacheReadInputTokens", 0); u["cache_write"] += mu.get("cacheCreationInputTokens", 0)
                u["thinking"] += mu.get("thinkingTokens", 0) or 0
            u["harness_list_cost_usd"] += e.get("total_cost_usd") or 0
        u["partial"] = 0
    else:
        for us in seen.values():
            u["input"] += us.get("input_tokens", 0) or 0
            u["cache_write"] += us.get("cache_creation_input_tokens", 0) or 0
            u["cache_read"] += us.get("cache_read_input_tokens", 0) or 0
            u["output"] += us.get("output_tokens", 0) or 0
        u["partial"] = 1
    u["n_messages"] = len(seen)
    return dict(u)


def codex_usage(traj):
    u = defaultdict(int)
    for line in open(traj, errors="ignore"):
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("type") == "turn.completed" and e.get("usage"):
            us = e["usage"]
            u["input"] += (us.get("input_tokens", 0) or 0) - (us.get("cached_input_tokens", 0) or 0)
            u["cache_read"] += us.get("cached_input_tokens", 0) or 0
            u["output"] += us.get("output_tokens", 0) or 0
            u["reasoning_output"] += us.get("reasoning_output_tokens", 0) or 0
            u["turns"] += 1
    return dict(u)


def price(model, u, table):
    p = table.get(model) or table.get(model.split("/")[-1])
    if not p:
        return None
    m = 1e-6
    return round(m * (u.get("input", 0) * p["input"] + u.get("output", 0) * p["output"] +
                      u.get("cache_read", 0) * p.get("cache_read", p["input"]) +
                      u.get("cache_write", 0) * p.get("cache_write", p["input"])), 4)


if __name__ == "__main__":
    root = Path(sys.argv[1])  # e.g. runs_srv
    table = json.load(open(Path(__file__).resolve().parent.parent / "data" / "prices.json")) \
        if (Path(__file__).resolve().parent.parent / "data" / "prices.json").exists() else {}
    rows = []
    for rp in sorted(root.glob("*/*/*/result.json")):
        rd = rp.parent
        if rd.name.startswith("_"):
            continue
        r = json.load(open(rp)); harness, model = r["system"].split(":")[:2]
        if r.get("skipped"):
            continue
        u = claude_usage(rd / "trajectory.jsonl") if harness == "claude" else codex_usage(rd / "trajectory.jsonl")
        rows.append({"condition": r["condition"], "system": r["system"], "task": r["task_id"], "wall_min": round(r["wall_s"] / 60, 1),
                     "delivered": r["delivered"], **u, "list_price_usd": price(model, u, table)})
    json.dump(rows, open(root / "token_usage.json", "w"), indent=1)
    agg = defaultdict(lambda: defaultdict(float))
    for x in rows:
        a = agg[(x["condition"], x["system"])]
        a["n"] += 1
        a["partial"] += x.get("partial", 0) or 0
        a["harness_cost"] += x.get("harness_list_cost_usd", 0) or 0
        for k in ("input", "cache_read", "cache_write", "output", "reasoning_output", "list_price_usd", "wall_min"):
            a[k] += x.get(k) or 0
    for (c, s), a in sorted(agg.items()):
        n = a["n"]
        print(f"{c:7s} {s:32s} n={int(n)} per-run: in {a['input']/n/1e3:.0f}k  cache_rd {a['cache_read']/n/1e3:.0f}k  "
              f"cache_wr {a['cache_write']/n/1e3:.0f}k  out {a['output']/n/1e3:.0f}k  min {a['wall_min']/n:.1f}  "
              f"${a['list_price_usd']/n:.2f}" + (f"  [Claude Code list cost ${a['harness_cost']/n:.2f}]" if a['harness_cost'] else "")
              + (f"  partial={int(a['partial'])}" if a['partial'] else "") + ("" if table else " (no price table yet)"))
