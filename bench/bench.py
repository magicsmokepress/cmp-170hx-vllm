#!/usr/bin/env python3
"""Decode throughput + power sampler for a vLLM endpoint pinned to one GPU."""
import argparse, json, subprocess, sys, threading, time, urllib.request

def post(url, key, model, prompt, ntok, timeout=600):
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": ntok,
                       "min_tokens": ntok, "ignore_eos": True,
                       "temperature": 0.8, "stream": False}).encode()
    req = urllib.request.Request(url + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": "Bearer " + key} if key else {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

class PowerSampler(threading.Thread):
    def __init__(self, gpu):
        super().__init__(daemon=True); self.gpu = gpu; self.samples = []; self.stop = False
    def run(self):
        p = subprocess.Popen(["nvidia-smi", "-i", str(self.gpu), "-lms", "100",
                              "--query-gpu=power.draw,clocks.sm,temperature.gpu",
                              "--format=csv,noheader,nounits"],
                             stdout=subprocess.PIPE, text=True)
        self.proc = p
        for line in p.stdout:
            if self.stop: break
            try:
                w, c, t = [float(x) for x in line.split(",")]
                self.samples.append((w, c, t))
            except ValueError:
                pass
        p.terminate()

def run_case(url, key, model, gpu, ntok, conc, prompt):
    # warm-up
    post(url, key, model, prompt, 32)
    time.sleep(3.0)                      # let clocks settle to idle-ish
    s = PowerSampler(gpu); s.start(); time.sleep(0.6)
    base = len(s.samples)
    results = [None] * conc
    def worker(i):
        results[i] = post(url, key, model, prompt, ntok)
    t0 = time.time()
    ths = [threading.Thread(target=worker, args=(i,)) for i in range(conc)]
    [t.start() for t in ths]; [t.join() for t in ths]
    dt = time.time() - t0
    s.stop = True; time.sleep(0.4)
    win = s.samples[base:]
    got = sum(r["usage"]["completion_tokens"] for r in results)
    watts = sum(x[0] for x in win) / max(len(win), 1)
    clk = sum(x[1] for x in win) / max(len(win), 1)
    temp = max((x[2] for x in win), default=0)
    tps = got / dt
    return {"tokens": got, "sec": round(dt, 2), "tok_s": round(tps, 2),
            "watts": round(watts, 1), "j_per_tok": round(watts / tps, 3),
            "tok_s_per_100w": round(tps / watts * 100, 2),
            "sm_mhz": round(clk), "max_temp": round(temp), "psamples": len(win)}

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--url"); a.add_argument("--model"); a.add_argument("--gpu", type=int)
    a.add_argument("--key", default=""); a.add_argument("--ntok", type=int, default=1500)
    a.add_argument("--conc", type=int, default=1); a.add_argument("--label", default="")
    args = a.parse_args()
    prompt = "Write a detailed technical description of how a modern GPU schedules work across streaming multiprocessors. Be thorough and specific.\n\n"
    r = run_case(args.url, args.key, args.model, args.gpu, args.ntok, args.conc, prompt)
    r["label"] = args.label
    print(json.dumps(r))
