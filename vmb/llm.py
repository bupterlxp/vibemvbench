"""Thin client over the OpenAI-compatible responses endpoint (text + images), with retries and a disk cache."""
import base64
import hashlib
import json
import os
import time
from pathlib import Path

from . import ROOT, load_env

CACHE = ROOT / "data" / "llm_cache"


def _client():
    load_env()
    from openai import OpenAI
    return OpenAI(base_url=os.environ["TOKENWAVE_BASE_URL"] + "/v1", api_key=os.environ["TOKENWAVE_API_KEY"],
                  timeout=600, max_retries=0)


def img_part(path_or_bytes, mime="image/jpeg"):
    b = Path(path_or_bytes).read_bytes() if not isinstance(path_or_bytes, bytes) else path_or_bytes
    return {"type": "input_image", "image_url": f"data:{mime};base64,{base64.b64encode(b).decode()}"}


def ask(model, content, instructions="You are a careful assistant.", json_mode=False, effort=None, cache=True, tries=5):
    """content: str or list of parts ({'type':'input_text',...}/img_part). Returns text (or parsed JSON)."""
    if isinstance(content, str):
        content = [{"type": "input_text", "text": content}]
    key = hashlib.sha256(json.dumps([model, instructions, content, effort], sort_keys=True).encode()).hexdigest()
    cp = CACHE / f"{key}.json"
    if cache and cp.exists():
        txt = json.load(open(cp))["text"]
    else:
        kw = {"reasoning": {"effort": effort}} if effort else {}
        for i in range(tries):
            try:
                r = _client().responses.create(model=model, instructions=instructions,
                                               input=[{"role": "user", "content": content}], **kw)
                txt = r.output_text
                break
            except Exception as e:
                if i == tries - 1:
                    raise
                time.sleep(min(5 * 2 ** i, 120))
        CACHE.mkdir(parents=True, exist_ok=True)
        json.dump({"model": model, "text": txt}, open(cp, "w"))
    if json_mode:
        s = txt[txt.find("{"): txt.rfind("}") + 1]
        return json.loads(s)
    return txt


# ---- Gemini (hears audio): media is sent as an MP4 data URL through the OpenAI-compatible chat endpoint ----
GEMINI_PRO = os.environ.get("VMB_GEMINI_PRO", "gemini-3.1-pro-preview")
GEMINI_FLASH = os.environ.get("VMB_GEMINI_FLASH", "gemini-3.6-flash")


def audio_as_mp4(wav, out=None, width=320):
    """Wrap an audio file in a tiny black-frame MP4 (the endpoint accepts video, not raw audio)."""
    import subprocess
    import tempfile
    from . import FFMPEG
    out = out or tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=black:s={width}x{width * 3 // 4}:r=2",
                    "-i", str(wav), "-shortest", "-c:v", "libx264", "-tune", "stillimage", "-c:a", "aac", "-b:a", "128k",
                    out], check=True)
    return out


def _parse_json(txt):
    return json.loads(txt[txt.find("{"): txt.rfind("}") + 1])


TOKENWAVE_AV = {"gemini-3.8-flash-high"}
ZGC_AV = {"gemini-3.8-flash"}   # zgc gateway (Bearer auth)


JUDGE_USD_PER_DAY = float(os.environ.get("VMB_JUDGE_USD_PER_DAY", "0") or 0)   # 0 = no cap
JUDGE_LEDGER = CACHE / "judge_spend.jsonl"
JUDGE_PRICE = {"gemini-3.8-flash": (0.5, 3.0)}   # $/M input, output (list price, conservative)


def _judge_spent_today():
    if not JUDGE_LEDGER.exists():
        return 0.0
    day = time.strftime("%Y-%m-%d")
    return sum(json.loads(l)["usd"] for l in open(JUDGE_LEDGER) if l.startswith('{"day": "' + day))


def ask_av(model, text, media=(), instructions=None, json_mode=False, cache=True, tries=5, sample=0):
    if model in TOKENWAVE_AV:
        tries = max(tries, 12)   # TokenWave routes to a pool of upstream accounts; some answer "User location is not supported"
    """Ask a model that can watch/hear: media is a list of MP4 paths (video with sound)."""
    load_env()
    import urllib.request
    parts = [{"type": "text", "text": text}]
    digest = []
    for m in media:
        b = Path(m).read_bytes(); digest.append(hashlib.sha256(b).hexdigest())
        parts.append({"type": "image_url", "image_url": {"url": "data:video/mp4;base64," + base64.b64encode(b).decode()}})
    msgs = ([{"role": "system", "content": instructions}] if instructions else []) + [{"role": "user", "content": parts}]
    key = hashlib.sha256(json.dumps([model, instructions, text, digest] + ([sample] if sample else [])).encode()).hexdigest()
    cp = CACHE / f"av_{key}.json"
    if cache and cp.exists():
        txt = json.load(open(cp))["text"]
    else:
        if model in ZGC_AV:
            url = os.environ["ZGC_BASE_URL"] + "/v1/chat/completions"
            hdr = {"Authorization": "Bearer " + os.environ["ZGC_API_KEY"], "Content-Type": "application/json"}
        elif model in TOKENWAVE_AV:   # TokenWave serves Gemini 3.8 Flash (Bearer auth)
            url = os.environ["TOKENWAVE_BASE_URL"] + "/v1/chat/completions"
            hdr = {"Authorization": "Bearer " + os.environ["TOKENWAVE_GEMINI_KEY"], "Content-Type": "application/json"}
        else:
            url = os.environ["HC_BASE_URL"] + "/v1/chat/completions"
            hdr = {"x-api-key": os.environ["HC_API_KEY"], "Content-Type": "application/json"}
        for i in range(tries):
            try:
                req = urllib.request.Request(url, data=json.dumps({"model": model, "messages": msgs}).encode(), headers=hdr)
                while JUDGE_USD_PER_DAY and _judge_spent_today() >= JUDGE_USD_PER_DAY:
                    time.sleep(900)   # daily judge budget used up: wait for the next day
                d = json.loads(urllib.request.urlopen(req, timeout=600).read())
                if model in JUDGE_PRICE:
                    u = d.get("usage") or {}; pi, po = JUDGE_PRICE[model]
                    usd = (u.get("prompt_tokens", 0) * pi + u.get("completion_tokens", 0) * po) / 1e6
                    CACHE.mkdir(parents=True, exist_ok=True)
                    with open(JUDGE_LEDGER, "a") as f:
                        f.write(json.dumps({"day": time.strftime("%Y-%m-%d"), "model": model, "usd": usd}) + "\n")
                txt = d["choices"][0]["message"]["content"]
                if json_mode:
                    _parse_json(txt)  # raise (and retry) instead of caching a garbage answer
                break
            except Exception:
                if i == tries - 1:
                    raise
                time.sleep(min(5 * 2 ** i, 120))
        CACHE.mkdir(parents=True, exist_ok=True)
        json.dump({"model": model, "text": txt, "served_model": d.get("model") if not (cache and cp.exists()) else None},
                  open(cp, "w"))
    if json_mode:
        return _parse_json(txt)
    return txt
