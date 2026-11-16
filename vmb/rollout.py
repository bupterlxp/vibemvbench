"""Run one agent rollout for one task inside the Docker sandbox.

Conditions: bare (no music tools, no lyrics text), tools (mvtools CLI + lyrics text), oracle (reference annotation JSON).
Harnesses: codex (Codex CLI, OpenAI-compatible responses endpoint), claude (Claude Code).
"""
import json, re, shlex
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

from . import ROOT, TASKS_DIR, load_env

RUNS = ROOT / os.environ.get("VMB_RUNS", "runs")
TOOLS_NOTE = {
    "bare": "",
    "tools": ("* Music analysis CLI: `mvtools beats|sections|energy|all AUDIO` prints JSON (beat/downbeat times, BPM, section "
              "boundaries, RMS envelope, onsets). The sung lyrics of the clip are in `/task/lyrics.txt` (one line per sung "
              "line, in order, without timestamps)."),
    "ears": "",
    "anon": "",
    "oracle": ("* `/task/reference.json` contains reference annotations of the clip: beats, downbeats, section boundaries, "
               "and timed lyric lines (seconds from clip start). You may rely on them."),
}
INSTRUCTION = ("Read /task/ENVIRONMENT.md and /task/prompt.txt. Complete only that task autonomously before the deadline. "
               "Deliver everything listed in ENVIRONMENT.md to /work/out/. Do the best work you possibly can.")


SAFETY_CAP_MIN = int(os.environ.get("VMB_SAFETY_CAP_MIN", "720"))  # unlimited runs are still killed after this


def BUDGET_LINE(budget_min):
    if not budget_min:
        return ("* There is no time limit. Take as long as you need to make the best video you can; stop when you are done.")
    return (f"* Wall-clock budget: **{budget_min} minutes** from start, including thinking, tool calls, previews and the final "
            "render.\n  At the deadline the container is killed; anything not written to `/work/out/` by then is lost.")


EARS_SRC = {"sol": "codex-gpt-6.1-sol-max", "astra": "codex-gpt-6-astra-max", "grok": "codex-grok-4.7-max",
            "dspro": "codex-deepseek-v4-pro-0813-max"}   # swapped ears: whose delivered timing sheet the director hears through
EARS_NOTE = ("\n\nThe song has already been listened to for you. /task/ears/timing.json holds what the listener heard: tempo, "
             "beats, downbeats, section boundaries and timed lyric lines (it may be incomplete). Use it as your hearing of the "
             "song; do not analyse /task/audio.wav yourself, and use the audio file only as the soundtrack.")


def ears_sheet(task_dir, tag):
    """Timing sheet for the swapped-ears condition: a main-run system's delivered timing.json, or the reference ('ref')."""
    if tag == "ref":
        r = json.load(open(task_dir / "reference.json"))
        return {"bpm": r["bpm"], "beats": r["beats"], "downbeats": r["downbeats"],
                "sections": [0.0] + list(r["section_boundaries"]) + [r["duration"]],
                "lyrics": [{k: l[k] for k in ("start", "end", "text")} for l in r["lyrics_lines"]]}
    return json.load(open(ROOT / "runs_disc" / "bare" / EARS_SRC[tag] / task_dir.name / "work" / "out" / "timing.json"))


HEAR_PROMPT = (
    "Listen to this audio and write down exactly what it contains, as precisely as you can. There is no video to make.\n\n"
    "The audio is /task/audio.wav ({dur:.1f} s). Write /work/out/timing.json with, in seconds from the start of the file:\n"
    '{{"bpm": number|null, "beats": [every beat], "downbeats": [first beat of every bar], "meter": beats per bar|null, '
    '"sections": [times where the music clearly changes or stops], "lyrics": [{{"start": s, "end": s, "text": "..."}} for any '
    "spoken or sung words]}}\n"
    "Use empty lists for anything absent. Then write /work/out/notes.md with how you determined each item. "
    "You may install any tools you like.\n")


def stage_hearing(sid, dest):
    """Hearing-test stimulus: audio only, no title, no criteria; deliverable is timing.json."""
    from .hearing_test import OUT as HT
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(HT / sid / "audio.wav", dest / "audio.wav")
    dur = json.load(open(HT / sid / "truth.json"))["spec"].get("dur", 24.0)
    (dest / "prompt.txt").write_text(HEAR_PROMPT.format(dur=dur), encoding="utf-8")
    (dest / "HEAR").write_text("")
    (dest / "ENVIRONMENT.md").write_text(
        "# Environment\n\nYou are running fully autonomously inside a Linux container. Nobody will answer questions. This is a "
        "single non-interactive session; finish only when /work/out/timing.json is written.\n\n"
        f"* Working directory: `/work`. Input: `/task/audio.wav`. Resources: {os.environ.get('VMB_CPUS', '2')} CPU cores, "
        f"{os.environ.get('VMB_MEM', '3500m')} memory, network access.\n"
        "* Available: Python 3 (numpy, scipy, Pillow, opencv), Node 22, ffmpeg/ffprobe, sox. You may pip/npm install.\n")


def stage_task(task_dir, cond, budget_min, dest, tag=None):
    if cond == "hear":
        return stage_hearing(task_dir.name, dest)
    t = json.load(open(task_dir / "task.json"))
    dest.mkdir(parents=True, exist_ok=True)
    if cond == "anon":   # retrieval ablation: no title/artist anywhere, including the WAV's metadata tags
        subprocess.run([str(ROOT / "bin" / "ffmpeg"), "-v", "error", "-y", "-i", str(task_dir / "audio.wav"), "-map_metadata", "-1",
                        "-fflags", "+bitexact", "-c", "copy", str(dest / "audio.wav")], check=True)
    else:
        shutil.copy(task_dir / "audio.wav", dest / "audio.wav")
    if t.get("free"):
        disclose = os.environ.get("VMB_DISCLOSE") == "1"
        extra = ("\n\nRead /task/CRITERIA.md first: it says exactly how the video will be evaluated. Aim to do well on all of it."
                 if disclose else "") + (EARS_NOTE if cond == "ears" else "")
        title, artist = ("Untitled", "an unknown artist") if cond == "anon" else (t['song']['title'], t['song']['artist'])
        (dest / "prompt.txt").write_text(f"{t['prompt']}\n\nSong: \"{title}\" by {artist} "
                                         f"(file: /task/audio.wav, {t['spec']['duration']:.1f} s).{extra}\n", encoding="utf-8")
        if disclose:
            shutil.copy(ROOT / "sandbox" / "CRITERIA.md", dest / "CRITERIA.md")
    else:
        b = json.load(open(task_dir / "brief.json"))
        req = "\n".join(f"- {r}" for r in b["requirements"])
        (dest / "prompt.txt").write_text(f"# {b['title']}\n\nMusic: \"{t['song']['title']}\" by {t['song']['artist']} "
                                         f"({t['song']['genre']}), clip of {t['spec']['duration']:.1f} s.\n\n{b['brief']}\n\n"
                                         f"Requirements:\n{req}\n", encoding="utf-8")
    sp = t["spec"]
    env = (ROOT / "sandbox" / "ENVIRONMENT.md").read_text().format(
        budget_line=BUDGET_LINE(budget_min), cpus=os.environ.get("VMB_CPUS", "2"), mem=os.environ.get("VMB_MEM", "3500m"), tools_note=TOOLS_NOTE[cond], w=sp["resolution"][0], h=sp["resolution"][1], fps=sp["fps"],
        dur=sp["duration"])
    (dest / "ENVIRONMENT.md").write_text(env)
    if cond == "ears":
        (dest / "ears").mkdir(exist_ok=True)
        json.dump(ears_sheet(task_dir, tag), open(dest / "ears" / "timing.json", "w"), ensure_ascii=False, indent=1)
    if cond in ("tools", "oracle") and t["lyrics_text"]:
        (dest / "lyrics.txt").write_text("\n".join(t["lyrics_text"]) + "\n", encoding="utf-8")
    if cond == "oracle":
        ref = json.load(open(task_dir / "reference.json")); ref.pop("rms", None); ref.pop("rms_hop_s", None)
        json.dump(ref, open(dest / "reference.json", "w"), ensure_ascii=False)


CODEX_EFFORT = {}  # the endpoint accepts "max" directly


TOKENWAVE_MODELS = {"gpt-6.1-sol", "grok-4.7", "gpt-6-astra", "gpt-6-luna", "gpt-6.1-terra"}  # served by TokenWave; other GPT models go through hapi
ZGC_MODELS = {"kimi-k3", "glm-5.3", "deepseek-v4-pro-0813", "deepseek-v4.1-flash", "qwen3.8-max"}  # zgc gateway (responses API)
TOKENWAVE_KEY_ENV = {"grok-4.7": "TOKENWAVE_GROK_KEY"}  # models that need their own TokenWave key


ZGC_CONTEXT, ZGC_COMPACT_AT = 256000, 100000   # 150k still hit 413 on screenshot-heavy Flash/Qwen sessions
# Codex ends a session whenever the model replies without a tool call; gateway models often do this mid-task ("Let me check
# the frames.") or after a gateway refusal. If the session ends without /work/out/final.mp4, the harness resumes it with this
# fixed message, up to NUDGES times. Systems that deliver in their first session are unaffected.
NUDGES = 8
NUDGE = ("Your session ended but /work/out/final.mp4 does not exist yet. Continue working on the task until all deliverables "
         "in /task/ENVIRONMENT.md are in /work/out.")
NUDGE_HEAR = "Your session ended but /work/out/timing.json does not exist yet. Continue until it is written."
MOSS_MODELS = {"nex-n2.6"}
ALIYUN_MODELS = set(filter(None, os.environ.get("VMB_ALIYUN_MODELS", "deepseek-v4.1-flash").split(",")))   # Aliyun MaaS (responses API)
TEXT_ONLY_MODELS = {"glm-5.3", "deepseek-v4-pro-0813", "deepseek-v4.1-flash", "nex-n2.6"}   # text-only on the gateway; only Kimi and Qwen (and the native GPT/Grok/Claude systems) view images


def codex_cmd(model, effort):
    """Codex CLI over the OpenAI responses API: TokenWave for TOKENWAVE_MODELS (Bearer auth), otherwise the hapi gateway
    (x-api-key auth)."""
    effort = CODEX_EFFORT.get(effort, effort)
    if model in ALIYUN_MODELS and os.environ.get("ALIYUN_BASE_URL"):
        provider = ('model_provider="ali"', 'model_providers.ali.name="ali"',
                    f'model_providers.ali.base_url="{os.environ["ALIYUN_BASE_URL"]}"',
                    'model_providers.ali.env_key="ALIYUN_API_KEY"', 'model_providers.ali.wire_api="responses"',
                    'model_providers.ali.requires_openai_auth=false', 'model_providers.ali.request_max_retries=30',
                    'model_providers.ali.stream_max_retries=30')
    elif model in MOSS_MODELS:
        provider = ('model_provider="moss"', 'model_providers.moss.name="moss"',
                    f'model_providers.moss.base_url="{os.environ.get("MOSS_PROXY", os.environ["MOSS_API_ORIGIN"])}/v1"',
                    'model_providers.moss.env_key="MOSS_API_KEY"', 'model_providers.moss.wire_api="responses"',
                    'model_providers.moss.requires_openai_auth=false', 'model_providers.moss.request_max_retries=30',
                    'model_providers.moss.stream_max_retries=30')
    elif model in ZGC_MODELS:
        provider = ('model_provider="zgc"', 'model_providers.zgc.name="zgc"',
                    f'model_providers.zgc.base_url="{os.environ["ZGC_BASE_URL"]}/v1"',
                    'model_providers.zgc.env_key="ZGC_API_KEY"', 'model_providers.zgc.wire_api="responses"',
                    'model_providers.zgc.requires_openai_auth=false', 'model_providers.zgc.request_max_retries=30',
                    'model_providers.zgc.stream_max_retries=30')
    elif model in TOKENWAVE_MODELS:
        provider = ('model_provider="tw"', 'model_providers.tw.name="tw"',
                    f'model_providers.tw.base_url="{os.environ["TOKENWAVE_BASE_URL"]}/v1"',
                    f'model_providers.tw.env_key="{TOKENWAVE_KEY_ENV.get(model, "TOKENWAVE_API_KEY")}"', 'model_providers.tw.wire_api="responses"',
                    'model_providers.tw.requires_openai_auth=false')
    else:
        provider = ('model_provider="hc"', 'model_providers.hc.name="hc"',
                    f'model_providers.hc.base_url="{os.environ["HC_BASE_URL"]}/v1"', 'model_providers.hc.env_key="HC_API_KEY"',
                    'model_providers.hc.wire_api="responses"', 'model_providers.hc.requires_openai_auth=false',
                    'model_providers.hc.env_http_headers={"x-api-key"="HC_API_KEY"}')
    cfg = (f'model="{model}"', f'model_reasoning_effort="{effort}"') + provider
    if model in ZGC_MODELS or model in MOSS_MODELS:   # gateway rejects request bodies past ~2 MB (413); Codex lacks metadata for these models
        cfg += (f'model_context_window={ZGC_CONTEXT}', f'model_auto_compact_token_limit={ZGC_COMPACT_AT}')
    if model in MOSS_MODELS:   # Nex accepts function tools only: drop the hosted web_search tool and the multi-agent namespace
        cfg += ('web_search="disabled"', "features.multi_agent=false", "features.multi_agent_v2=false")
    if model in TEXT_ONLY_MODELS:   # the view_image tool makes the gateway reject the whole session; agents inspect frames with code
        cfg += ("features.view_image=false",)
    args = ["codex", "exec", "--json", "--skip-git-repo-check", "--dangerously-bypass-approvals-and-sandbox", "-C", "/work"]
    for c in cfg:
        args += ["-c", c]
    return args + [INSTRUCTION]


def with_nudges(cmd, msg=NUDGE):
    """Wrap a codex exec command so the session is resumed (same container, same history) until final.mp4 exists."""
    first = shlex.join(cmd)
    resume = shlex.join(cmd[:-1] + ["resume", "--last", msg])
    return ["bash", "-c", f"{first}; for i in $(seq {NUDGES}); do {{ [ -f /work/out/final.mp4 ] || {{ [ -f /task/HEAR ] && [ -f /work/out/timing.json ]; }}; }} && break; "
                          f"echo '{{\"type\":\"harness.nudge\",\"n\":'$i'}}'; {resume}; done"]


# models served through the hapi endpoint's Anthropic-compatible /v1/messages, run in the Claude Code harness
HAPI_MODELS = {"bigmodel/glm-5.3", "deepseek/deepseek-v4-pro-0813", "deepseek/deepseek-v4.1-flash", "kimi-k3", "glm-5.3", "emily/glm-5.3", "sophia/glm-5.3", "sophia/deepseek-v4.1-flash", "deepseek-v4.1-flash", "deepseek-v4-pro-0813", "sophia/deepseek-v4-pro-0813",
               "gemini-3.1-pro-preview"}


def claude_cmd(model, effort):
    return ["claude", "-p", INSTRUCTION, "--model", model, "--effort", effort, "--output-format", "stream-json", "--verbose",
            "--dangerously-skip-permissions", "--add-dir", "/task"]


INFRA_MARKERS = ("余额不足", "insufficient balance", "403 Forbidden", "USER_INACTIVE", "User account is not active", "401 Unauthorized", "Failed to authenticate",
                 "insufficient_quota", "billing", "ECONNRESET", "server_error", "model_not_found", "overloaded", "stream disconnected",
                 "Connection to the API was lost", "An error occurred while processing your request", "502 Bad Gateway",
                 "503 Service", "rate_limit_error", "Rate limit reached", "429 Too Many Requests",
                 "empty or malformed response", "Payload Too Large", "No healthy", "upstream_error", "All available accounts exhausted")


def rmtree_any(path):
    """Remove a run dir even if the sandbox user created files the host user cannot delete."""
    try:
        shutil.rmtree(path)
    except PermissionError:
        path = Path(path).resolve()
        subprocess.run(["docker", "run", "--rm", "-v", f"{path.parent}:/p", "vmb-sandbox:base", "rm", "-rf", f"/p/{path.name}"],
                       check=True)


def mp4_ok(path):
    """Delivered = a final.mp4 that decodes; a session cut mid-encode leaves a file without a moov atom."""
    if not Path(path).exists():
        return False
    ff = ROOT / "bin" / "ffprobe"
    r = subprocess.run([str(ff) if ff.exists() else "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                        str(path)], capture_output=True, text=True)
    return r.returncode == 0 and bool(r.stdout.strip())


def infra_failure(rd):
    """A run that failed to deliver AND whose *API error events* name an endpoint/connection fault is an infrastructure
    fault, not a model failure. Only error/result events are inspected: tool outputs (pip/HF download banners such as
    "rate limits and faster downloads") must not count."""
    if mp4_ok(rd / "work" / "out" / "final.mp4"):
        return None
    hits = []
    for f in ("trajectory.jsonl", "stderr.log"):
        p = rd / f
        if not p.exists():
            continue
        for line in p.read_text(errors="ignore").splitlines():
            if f == "trajectory.jsonl":
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                if e.get("type") not in ("error", "turn.failed", "result") and not (
                        e.get("type") == "system" and "api_error" in str(e.get("subtype", ""))):
                    continue
                if e.get("type") == "result" and not e.get("is_error"):
                    continue
                line = json.dumps(e)
            hits += [m for m in INFRA_MARKERS if m in line]
    return sorted(set(hits)) or None


def wait_for_endpoint(system, max_wait_s=6 * 3600):
    """Block until the model endpoint answers a trivial request (polls every 5 min), so outages are waited out."""
    import urllib.request
    harness, model = system.split(":")[:2]
    t0 = time.time()
    while time.time() - t0 < max_wait_s:
        try:
            if harness == "codex" and model in ALIYUN_MODELS and os.environ.get("ALIYUN_BASE_URL"):
                req = urllib.request.Request(os.environ["ALIYUN_BASE_URL"] + "/responses",
                                             data=json.dumps({"model": model, "input": "Say OK"}).encode(),
                                             headers={"Authorization": "Bearer " + os.environ["ALIYUN_API_KEY"],
                                                      "Content-Type": "application/json"})
            elif harness == "codex" and model in MOSS_MODELS:
                req = urllib.request.Request(os.environ["MOSS_API_ORIGIN"] + "/v1/responses",
                                             data=json.dumps({"model": model, "input": "Say OK"}).encode(),
                                             headers={"Authorization": "Bearer " + os.environ["MOSS_API_KEY"],
                                                      "Content-Type": "application/json"})
            elif harness == "codex" and model in ZGC_MODELS:
                req = urllib.request.Request(os.environ["ZGC_BASE_URL"] + "/v1/responses",
                                             data=json.dumps({"model": model, "input": "Say OK"}).encode(),
                                             headers={"Authorization": "Bearer " + os.environ["ZGC_API_KEY"],
                                                      "Content-Type": "application/json"})
            elif harness == "codex" and model in TOKENWAVE_MODELS:
                req = urllib.request.Request(os.environ["TOKENWAVE_BASE_URL"] + "/v1/responses",
                                             data=json.dumps({"model": model, "input": "Say OK"}).encode(),
                                             headers={"Authorization": "Bearer " + os.environ[TOKENWAVE_KEY_ENV.get(model, "TOKENWAVE_API_KEY")],
                                                      "Content-Type": "application/json"})
            elif harness == "codex":
                req = urllib.request.Request(os.environ["HC_BASE_URL"] + "/v1/responses",
                                             data=json.dumps({"model": model, "input": "Say OK"}).encode(),
                                             headers={"x-api-key": os.environ["HC_API_KEY"], "Content-Type": "application/json"})
            else:
                hapi = model in HAPI_MODELS
                base = os.environ["HC_BASE_URL"] if hapi else os.environ["ANTHROPIC_BASE_URL"]
                key = os.environ["HC_API_KEY"] if hapi else os.environ["ANTHROPIC_AUTH_TOKEN"]
                req = urllib.request.Request(base + "/v1/messages", data=json.dumps(
                    {"model": model, "max_tokens": 8, "messages": [{"role": "user", "content": "Say OK"}]}).encode(),
                    headers={"x-api-key": key, "Authorization": "Bearer " + key, "Content-Type": "application/json",
                             "anthropic-version": "2023-06-01"})
            urllib.request.urlopen(req, timeout=120).read()
            return True
        except Exception as e:
            print(f"[wait_for_endpoint] {system}: {repr(e)[:120]}; retry in 5 min", flush=True)
            time.sleep(300)
    return False


def _claim(rd):
    """Per-run lock (atomic mkdir) so two runner processes can never work on the same run dir."""
    lock = Path(str(rd) + ".lock")
    try:
        lock.parent.mkdir(parents=True, exist_ok=True); lock.mkdir()
        (lock / "pid").write_text(f"{os.getpid()} {socket.gethostname()}")
        return lock
    except FileExistsError:
        try:
            pid = int((lock / "pid").read_text().split()[0]); os.kill(pid, 0)
            return None            # owner alive
        except Exception:
            shutil.rmtree(lock, ignore_errors=True)   # stale lock from a dead process
            return _claim(rd)


def run(task_id, system, cond="tools", budget_min=30, image=None, tag=None, infra_retries=3):
    harness, model, effort = (system.split(":") + ["high"])[:3]
    rd = RUNS / cond / f"{harness}-{model}-{effort}".replace("/", "_") / (task_id + (f"__{tag}" if tag else ""))
    lock = _claim(rd)
    if lock is None:
        return {"task_id": task_id, "system": system, "condition": cond, "delivered": None, "skipped": "locked by another runner"}
    try:
        return _run_locked(task_id, system, cond, budget_min, image, tag, infra_retries)
    finally:
        shutil.rmtree(lock, ignore_errors=True)


def _run_locked(task_id, system, cond, budget_min, image, tag, infra_retries):
    """Runs lost to infrastructure faults are re-run (up to `infra_retries`); attempts are kept for the record."""
    for attempt in range(infra_retries + 1):
        res = _run_once(task_id, system, cond, budget_min, image, tag)
        if not res.get("infra_failure") or attempt == infra_retries:
            return res
        wait_for_endpoint(system)
        rd = Path(res["run_dir"])
        archive = rd.parent / f"_infra_attempt{attempt}_{rd.name}"
        if archive.exists():
            rmtree_any(archive)
        rd.rename(archive)
    return res


def _run_once(task_id, system, cond="tools", budget_min=30, image=None, tag=None):
    """system: e.g. 'codex:gpt-6-astra:high' or 'claude:claude-opus-5-5:max'."""
    load_env()
    harness, model, effort = (system.split(":") + ["high"])[:3]
    task_dir = TASKS_DIR / task_id
    name = f"{harness}-{model}-{effort}".replace("/", "_")
    rd = RUNS / cond / name / (task_id + (f"__{tag}" if tag else ""))
    if (rd / "result.json").exists():
        r = json.load(open(rd / "result.json"))
        r.setdefault("run_dir", str(rd))
        if "infra_failure" not in r:
            r["infra_failure"] = infra_failure(rd)
        return r
    if rd.exists():
        rmtree_any(rd)
    (rd / "work").mkdir(parents=True); stage_task(task_dir, cond, budget_min, rd / "task", tag)
    os.chmod(rd / "work", 0o777)
    image = image or os.environ.get("VMB_IMAGE_TOOLS" if cond == "tools" else "VMB_IMAGE_BASE",
                                    "vmb-sandbox:tools" if cond == "tools" else "vmb-sandbox:base")
    env = []
    if harness == "codex":
        cmd = with_nudges(codex_cmd(model, effort), NUDGE_HEAR if cond == "hear" else NUDGE)
        kv = TOKENWAVE_KEY_ENV.get(model, "TOKENWAVE_API_KEY")
        env = ["-e", f"ALIYUN_API_KEY={os.environ['ALIYUN_API_KEY']}"] if (model in ALIYUN_MODELS and os.environ.get("ALIYUN_BASE_URL")) else \
              ["-e", f"MOSS_API_KEY={os.environ['MOSS_API_KEY']}"] if model in MOSS_MODELS else \
              ["-e", f"ZGC_API_KEY={os.environ['ZGC_API_KEY']}"] if model in ZGC_MODELS else \
              ["-e", f"{kv}={os.environ[kv]}"] if model in TOKENWAVE_MODELS else \
              ["-e", f"HC_API_KEY={os.environ['HC_API_KEY']}"]
    elif model in HAPI_MODELS:
        cmd = claude_cmd(model, effort)
        env = ["-e", f"ANTHROPIC_BASE_URL={os.environ['HC_BASE_URL']}", "-e", f"ANTHROPIC_API_KEY={os.environ['HC_API_KEY']}"]
    else:
        cmd = claude_cmd(model, effort); env = ["-e", "ANTHROPIC_BASE_URL", "-e", "ANTHROPIC_AUTH_TOKEN"]
    # optional outbound proxy (dev machine): web/pip/HF through the proxy, model APIs direct
    net = []
    if os.environ.get("VMB_PROXY"):
        px = os.environ["VMB_PROXY"]
        no = ",".join(filter(None, [os.environ.get(k, "").split("/")[2] if "://" in os.environ.get(k, "") else "" for k in ("TOKENWAVE_BASE_URL", "HC_BASE_URL", "ZGC_BASE_URL", "MOSS_API_ORIGIN")] + ["localhost", "127.0.0.1"]))
        if os.environ.get("ALIYUN_BASE_URL"):
            no += "," + os.environ["ALIYUN_BASE_URL"].split("/")[2]
        net = ["--network", "host"] + [x for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")
                                       for x in ("-e", f"{k}={px}")] + ["-e", f"NO_PROXY={no}", "-e", f"no_proxy={no}"]
    cname = f"vmb-{abs(hash(str(rd))) % 10**10}"
    docker = ["docker", "run", "--rm", "--name", cname, "-u", "agent", "--cpus", os.environ.get("VMB_CPUS", "2"), "--memory", os.environ.get("VMB_MEM", "3500m"),
              "-v", f"{rd / 'task'}:/task:ro", "-v", f"{rd / 'work'}:/work", "-e", "HOME=/home/agent", *net, *env, image,
              "timeout", "-s", "KILL", str((budget_min or SAFETY_CAP_MIN) * 60), *cmd]
    t0 = time.time()
    with open(rd / "trajectory.jsonl", "w") as out, open(rd / "stderr.log", "w") as err:
        p = subprocess.run(docker, stdout=out, stderr=err)
    res = {"task_id": task_id, "system": system, "condition": cond, "budget_min": budget_min, "exit": p.returncode,
           "wall_s": round(time.time() - t0, 1), "delivered": mp4_ok(rd / "work" / "out" / "final.mp4"),
           "infra_failure": infra_failure(rd), "run_dir": str(rd),
           **({"delivered": (rd / "work" / "out" / "timing.json").exists()} if cond == "hear" else {}),
           "nudges": (rd / "trajectory.jsonl").read_text(errors="ignore").count('"type":"harness.nudge"'),
           "provider": (re.search(r'model_provider="(\w+)"', " ".join(map(str, cmd))) or [None, None])[1]}
    # killed (SIGKILL, exit 137) well before our own deadline = OOM or external kill, not the agent's fault
    cap_s = (budget_min or SAFETY_CAP_MIN) * 60
    if not res["delivered"] and p.returncode == 137 and res["wall_s"] < cap_s - 60:
        res["infra_failure"] = (res["infra_failure"] or []) + ["killed_before_deadline (likely OOM)"]
    json.dump(res, open(rd / "result.json", "w"), indent=1)
    return res
