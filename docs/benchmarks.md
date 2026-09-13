# Benchmarks

Every number here was measured on the hardware listed in the README. Raw data
is in [`bench/data/`](../bench/data/); the harness is
[`bench/bench.py`](../bench/bench.py).

## Method

`bench.py` posts to `/v1/completions` with `ignore_eos` and `min_tokens` set to
the same value, so generation length is pinned exactly and the throughput
figure is not a function of when the model decided to stop. It samples
`nvidia-smi` at 100 ms in a background thread for the duration of the request
and reports the average draw over exactly that window.

Three rules that keep these numbers honest:

1. **Benchmark with prose, not repetitive text.** If the server uses
   speculative decoding, a predictable prompt inflates the result by more than
   2x. On the 27B we measured 364 tok/s on trivially predictable output and 161
   tok/s on real prose; that gap *is* the draft acceptance rate.
2. **Read `usage.completion_tokens`, never a count of stream chunks.** With
   speculative decoding, 400 tokens arrive in ~54 chunks. Counting chunks
   undercounts badly.
3. **Randomise long prompts per run.** Otherwise prefix caching serves the
   second run and hands you a fictional prefill number.

Single-stream points carry roughly 5% run-to-run spread because these are live
services and real traffic lands mid-benchmark. Trust the curve across many
points, not any one pair.

## Serving throughput

Idle server, one request at a time, generation pinned to 400 tokens, prefill
prompts ~9-10k tokens of freshly randomised words.

| model | GPU | prefill tok/s | decode tok/s (prose) |
|---|---|---|---|
| **Qwen3-Next-80B-A3B W4A16** | **170HX** | **6800-7950** | **112-121** |
| Qwen3.8-27B W4A16 + spec decode | RTX 4090 | 2070 | 161 |
| Cosmos-Reason2-8B fp8 | RTX 3090 | 3400 | 80 |

Two things here surprise people:

**The 80B prefills 3.6x faster than the 27B despite being three times the
size.** It is an MoE with ~3B active parameters. For long-context single-shot
work the big model on the slow card is the faster server end to end.

**The 27B's decode rate is not one number.** 364 tok/s on predictable output,
161 on prose. See rule 1 above.

## Both models on the same card

Measured 2026-09-13, same harness, same day, both at the 150 W cap.

| | Qwen3.8-27B + DFlash2 | Qwen3-Next-80B-A3B |
|---|---|---|
| decode, single stream | 127 tok/s | 116 tok/s |
| decode, 4 concurrent | 204 tok/s | 352 tok/s |
| decode, 8 concurrent | 235 tok/s | 352 tok/s (saturated at 4 slots) |
| decode at 24k context | 83 tok/s | 112 tok/s |
| prefill, ~8.9k tokens | 1768 tok/s | 6800-7950 tok/s |
| TTFT, 24k prefix, cold | 14.3 s | 3.4 s |
| TTFT, 24k prefix, warm | 0.56 s | 0.14 s |
| J/token, single stream | 1.12 | 1.18 |
| J/token, 8 concurrent | 0.63 | 0.41 |
| weight load, warm | 9.1 s for 15.8 GB | 68 s for 40.9 GB |

Raw data: `bench/data/27b_170hx_conc.jsonl`, `bench/data/80b_170hx_conc.jsonl`,
`bench/data/longctx_170hx.txt`.

The 80B wins everywhere except short-prompt single-stream decode, where the 27B
leads by 9%. It prefills nearly 4x faster, decodes 35% faster at 24k context,
and reaches 3x the concurrent throughput. The reason is that the 80B is an MoE
with roughly 3B active parameters, so it does less work per token than a 27B
dense model. On this card, the bigger model is the faster server.

The concurrency rows also show a configuration effect worth copying: the 80B is
run with `--max-num-seqs 4`, and its C4 and C8 numbers are identical within
noise (351.6 and 352.4 tok/s). Requests past the slot count queue. Raising
client concurrency above `--max-num-seqs` buys nothing at all.

### Load rate is model-dependent, not just link-dependent

The PCIe link caps host transfer at about 1.5 GB/s (see below), but that is an
upper bound, not a prediction:

| model | size | page cache | load time | effective rate |
|---|---|---|---|---|
| Qwen3.8-27B W4A16 | 15.8 GB | warm | 9.13 s | 1.73 GB/s |
| Qwen3-Next-80B W4A16 | 40.9 GB | 98% resident (`mincore`) | 68.07 s | 0.60 GB/s |

The 27B loads at roughly the link ceiling. The 80B, with its page cache
residency verified rather than assumed, loads at a third of that, so something
other than PCIe dominates its load path. Do not derive an expected load time
from the link speed alone; measure the model you actually serve.

## The 27B on the 170HX, head to head

The same 27B W4A16 stack (vLLM with speculative decoding and prefix caching)
run on three different cards, so the card is the only variable:

| | 170HX | RTX 3090 | RTX 4090 |
|---|---|---|---|
| decode, single stream | **152 tok/s** | 133-135 | 157-163 |
| decode at 24k context | **102 tok/s** | 88 | not measured |
| concurrency 8, end to end | 278 tok/s | not measured | 373-451 |
| 24k prefix TTFT, cold | 11.8 s | 19.5 s | not measured |
| 24k prefix TTFT, warm | 0.49 s | 0.75 s | not measured |
| power under load | 245 W (250 W cap) | not measured | not measured |
| temperature | 72 C | not measured | not measured |

For reference, the same model on llama.cpp on the 4090 does 39 tok/s at 24k
context. The 170HX's **102 tok/s at 24k is 2.6x that**, and its HBM2e makes it
the best of the three cards for long-context decode. It also leaves ~40 GB free
for a much larger KV pool, which none of the 24 GB cards can offer.

The 4090 is faster at short context. The 170HX is faster where it counts for
long prompts, and it is the only card that can hold the model plus a large KV
cache at the same time.

## Power sweep

`bench/data/170hx_c1.jsonl`, two repetitions per cap, 1200 tokens per run,
serving the 80B.

| cap | run 1 | run 2 | draw | J/token | SM clock |
|---|---|---|---|---|---|
| 250 W | 117.6 tok/s | 116.5 | 143-145 W | 1.22-1.25 | 1400 MHz |
| 200 W | 117.5 | 117.8 | 147-151 W | 1.26-1.28 | 1400 MHz |
| 175 W | 117.6 | 117.7 | 151 W | 1.28-1.29 | 1400 MHz |
| 150 W | 116.6 | 116.2 | 146-147 W | 1.25-1.26 | 1380 MHz |
| 125 W | 109.8 | 108.8 | 122 W | 1.12 | 1271 MHz |

Throughput is flat from 250 W down to 150 W. The card never approached its cap,
so the cap was never the limiting factor. Only at 125 W does the SM clock
finally drop enough to cost throughput, and even then only 7%.

**This result does not generalise to every model.** The same sweep against the
27B with speculative decoding has a completely different shape: the card draws
194 W, the SM clock tracks the cap all the way down, and 150 W costs about 7%
of throughput at short context and 16% at 24k. Speculative decoding verifies a
block of drafts in one pass, which is compute-heavy, and compute is what a cap
restricts. Both sweeps are tabulated in
[hardware.md](hardware.md#power).

### Cross-card efficiency

Same harness, same workload shape, each card serving its own model:

| card | J/token | peak temp |
|---|---|---|
| **CMP 170HX** | **1.22** | 59 C |
| RTX 4090 | 1.82 | 70 C |
| RTX 3090 | 4.11 | 83 C |

The 4090 tells the same story from the other side: it holds ~148 tok/s from a
450 W cap all the way down to 200 W while draw falls from 264 W to 185 W, with
the SM clock collapsing from 2730 MHz to 1416 MHz. **Single-stream LLM decode
does not care about clock.** It is memory-bandwidth bound. Below 200 W the 4090
stops saving anything, because its draw floors near 158 W in memory and VRAM,
which the cap cannot reach.

Chosen caps on this host: 4090 250 W, 170HX 150 W, 3090 275 W. About 95 W less
heat into the room, and the 3090 runs 9 C cooler.

## Raw memory bandwidth

Measured with `nvidia_bench` from
[bayley/cmpunlocker](https://github.com/bayley/cmpunlocker) (build for
`sm_80`):

```
READ   1325 GB/s
TRIAD  1292 GB/s
```

That is **88.7%** of the 1493 GB/s theoretical at this memory clock, which is a
good result, HBM2e delivering close to spec.

For contrast, llama.cpp on the same card extracts only ~728 GB/s effective. For
that engine the card is compute-bound, not bandwidth-bound, which is the
argument against overclocking it (see [tuning.md](tuning.md)).

## PCIe host transfer

With a matmul kernel held running in a background thread so the card is in P0:

```
H2D  1.53 GB/s
D2H  1.51 GB/s
link Gen2 x8, unchanged under load
```

RTX 3090, same host, same test: 24.1 GB/s H2D, 23.7 GB/s D2H at Gen4 x16.

If you measure this yourself, drive a compute kernel first. A pure DMA copy
leaves the shaders idle, the card stays in a low power state, and you will
measure the idle link and conclude the card cannot train up. That is a test
artifact, not a finding.

## One card that does not fit: gpt-oss-120b

63.4 GB of MXFP4 weights on a 64 GB card does **not** fit. Split across the
170HX and a 3090 under llama.cpp (`--tensor-split 64,24`, 32k context) it uses
46 GB + 16 GB, loads in ~60 s, and measures:

| | |
|---|---|
| decode, short prompt | 110 tok/s |
| decode after an 18k prompt | 101 tok/s |
| prefill | 2200 tok/s (18k tokens in 8.1 s) |

Comparable to the 80B on the 170HX alone. Given it costs a second card, the
80B is the better use of the hardware.

## Reproducing

```bash
# single point
python3 bench/bench.py --url http://localhost:8000 --model your-model \
        --gpu <nvidia-smi index> --ntok 1500 --conc 1 --label baseline

# power sweep: gpu, url, model, stock_watts, concurrency, ntokens, caps...
./bench/sweep.sh 1 http://localhost:8000 your-model 250 1 1200 250 200 175 150 125
```

`sweep.sh` needs passwordless `sudo nvidia-smi` to set caps, and restores the
stock limit on exit including on interrupt. Set `VLLM_API_KEY` if your endpoint
requires auth.
