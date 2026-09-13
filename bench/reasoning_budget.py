#!/usr/bin/env python3
"""Measure how much a reasoning model thinks, and what limiting it costs.

For each prompt with a checkable answer, records reasoning tokens, answer
tokens, finish_reason, empty-answer failures, and correctness, under a set of
conditions: unbounded thinking, thinking disabled, and several hard
thinking_token_budget values. Also probes the max_tokens trap: a cap that lands
mid-think returns an EMPTY answer with a normal-looking response.

Requires the server to run with --reasoning-parser (so reasoning comes back in
its own field) and, for thinking_token_budget, VLLM_USE_V2_MODEL_RUNNER=0.

Usage: reasoning_budget.py URL MODEL OUT.jsonl [reps]
"""
import json, re, subprocess, sys, tempfile, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL, MODEL, OUT = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3]
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 3
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}  # model card, general tasks

def num(s):
    m = re.findall(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    return m[0] if m else None  # first number: answers lead with it, checks follow

def check_prime(ans):
    code = re.search(r"```(?:python)?\n(.*?)```", ans, re.S)
    src = code.group(1) if code else ans
    test = src + "\nassert [n for n in range(-5,60) if is_prime(n)]==[2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59]\nassert is_prime(7919) and not is_prime(7917)\nprint('PASS')\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(test)
    try:
        r = subprocess.run([sys.executable, f.name], capture_output=True, text=True, timeout=10)
        return "PASS" in r.stdout
    except subprocess.TimeoutExpired:
        return False

PROMPTS = [
    ("trivial_capital", "What is the capital of France? Answer with one word.",
     lambda a: "paris" in a.lower()),
    ("trivial_ok", "Say OK.", lambda a: "ok" in a.lower()),
    ("arith", "What is 17 * 23? Answer with just the number.", lambda a: num(a) == "391"),
    ("time", "A train leaves at 14:35 and the trip takes 2 h 47 min. What time does it arrive? Answer HH:MM only.",
     lambda a: "17:22" in a),
    ("apples", "I have 3 apples. I eat 2, buy 5 more, then give away half. How many apples do I have? Just the number.",
     lambda a: num(a) == "3"),
    ("strawberry", "How many times does the letter r appear in the word strawberry? Just the number.",
     lambda a: num(a) == "3"),
    ("crt", "What is the smallest positive integer that leaves remainder 1 when divided by 2, 3, 4, 5 and 6, and is divisible by 7? Just the number.",
     lambda a: num(a) == "301"),
    ("code_prime", "Write a Python function is_prime(n) that returns True for primes and False otherwise, including for n < 2. Reply with only the code in a python code block.",
     check_prime),
    ("chitchat", "Good morning. How are you today?", None),
    ("oneline", "In one sentence, why do lighthouses still matter in the age of GPS?", None),
]

CONDITIONS = [
    ("unbounded", {"max_tokens": 16384}),
    ("thinking_off", {"max_tokens": 16384, "chat_template_kwargs": {"enable_thinking": False}}),
    ("budget_1024", {"max_tokens": 16384, "thinking_token_budget": 1024}),
    ("budget_256", {"max_tokens": 16384, "thinking_token_budget": 256}),
    ("budget_64", {"max_tokens": 16384, "thinking_token_budget": 64}),
    ("maxtok_256_trap", {"max_tokens": 256}),
]

def tokcount(text):
    if not text:
        return 0
    body = json.dumps({"model": MODEL, "prompt": text}).encode()
    req = urllib.request.Request(URL + "/tokenize", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["count"]

def run(job):
    cond, extra, (pid, prompt, check), rep = job
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], **SAMPLING, **extra}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        return {"cond": cond, "prompt": pid, "rep": rep, "error": e.read().decode()[:300]}
    dt = time.time() - t0
    msg = d["choices"][0]["message"]
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    answer = msg.get("content") or ""
    ct = d["usage"]["completion_tokens"]
    ans_tok = tokcount(answer)
    return {"cond": cond, "prompt": pid, "rep": rep,
            "completion_tokens": ct, "answer_tokens": ans_tok, "reasoning_tokens": ct - ans_tok,
            "reasoning_chars": len(reasoning), "finish": d["choices"][0]["finish_reason"],
            "empty_answer": not answer.strip(),
            "correct": (check(answer) if (check and answer.strip()) else (False if check else None)),
            "sec": round(dt, 2), "answer_head": answer.strip()[:80]}

jobs = [(c, e, p, r) for c, e in CONDITIONS for p in PROMPTS for r in range(REPS)]
print(f"{len(jobs)} requests, 8 concurrent", flush=True)
with open(OUT, "w") as f, ThreadPoolExecutor(8) as ex:
    for res in ex.map(run, jobs):
        f.write(json.dumps(res) + "\n"); f.flush()
print("RB-DONE", flush=True)
