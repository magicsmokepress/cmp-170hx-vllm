#!/usr/bin/env python3
"""Measure prefill throughput on a vLLM endpoint.

Prefill rate is prompt_tokens / TTFT. Two rules make the number honest:

  * The prompt must be freshly randomised every run, or prefix caching serves
    the second run and hands back a fictional number.
  * Measure TTFT from a streamed request, not total latency, or you are timing
    prefill plus decode.

Usage:  prefill.py URL MODEL [KEY] [approx_prompt_tokens] [reps]
        URL is the base, e.g. http://127.0.0.1:8000
"""
import json, random, sys, time, urllib.request

url = sys.argv[1].rstrip("/")
model = sys.argv[2]
key = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "none" else ""
target = int(sys.argv[4]) if len(sys.argv) > 4 else 10000
reps = int(sys.argv[5]) if len(sys.argv) > 5 else 3

WORDS = ("memory portal speaker harness cognition recall session thread ledger "
         "anchor quiet lantern orbit granite velvet compass meridian thicket "
         "bellows kestrel").split()

for rep in range(reps):
    # fresh randomness per rep: no seed, so prefix caching cannot serve this
    rnd = random.Random()
    doc = " ".join(rnd.choice(WORDS) for _ in range(int(target / 1.35)))
    body = json.dumps({"model": model, "prompt": doc + "\n\nSummarise the above.",
                       "max_tokens": 16, "min_tokens": 16, "ignore_eos": True,
                       "temperature": 0.8, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(
        url + "/v1/completions", data=body,
        headers={"Content-Type": "application/json",
                 **({"Authorization": "Bearer " + key} if key else {})})
    t0 = time.time()
    ttft = None
    usage = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            ch = d.get("choices") or []
            if ch and ch[0].get("text") and ttft is None:
                ttft = time.time() - t0
    pt = usage["prompt_tokens"] if usage else 0
    print(f"rep{rep}: prompt={pt} ttft={ttft:.3f}s prefill={pt / ttft:.0f} tok/s")
