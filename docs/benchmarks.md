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

The PCIe link caps host transfer at about 3.2 GB/s (see below). Neither model
comes close to it:

| model | size | page cache | load time | effective rate |
|---|---|---|---|---|
| Qwen3.8-27B W4A16 | 15.8 GB | warm | 9.13 s | 1.73 GB/s |
| Qwen3-Next-80B W4A16 | 40.9 GB | 98% resident (`mincore`) | 68.07 s | 0.60 GB/s |

The 27B reaches 54% of the link ceiling and the 80B 19%, with the 80B's page
cache residency verified rather than assumed. **Neither is link-bound**, so
something in the loader dominates in both cases, and it dominates differently
per model. Do not derive an expected load time from link speed; measure the
model you actually serve.

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

`bench/pcie_bw.py`, 512 MiB buffers, 20 iterations per direction.

| card | link | H2D uncontended | D2H uncontended | H2D contended |
|---|---|---|---|---|
| CMP 170HX | Gen2 x8 | **3.18 GB/s** | 3.12 GB/s | 2.50 GB/s |
| RTX 3090 | Gen4 x16 | 24.90 GB/s | 22.11 GB/s | 14.57 GB/s |

3.18 GB/s is ~79% of Gen2 x8's 4 GB/s theoretical, a normal efficiency. The
170HX is about 7.8x slower to feed than the 3090.

Note the link state this is measured at is **not stock**: Gen2 comes from
cmpunlocker's software retrain and x8 from a hardware capacitor mod on this
board. See [hardware.md](hardware.md#the-pcie-link-and-what-it-takes-to-make-it-usable).

### How to get this measurement wrong

An earlier version of this document reported 1.53 GB/s, less than half the real
figure, from a script that made two mistakes at once:

1. **It timed copies while a matmul was running.** The kernel was there to keep
   the card in P0, which is genuinely necessary (a pure DMA copy does not wake
   an idle GPU, and benchmarking that way measures the idle link). But leaving
   it running costs about 20%. Warm the card, stop the kernel, then time.
2. **It used 256 MiB buffers and 10 iterations.** Per-copy launch and
   synchronisation overhead dominated. 512 MiB and 20 iterations is enough to
   make it negligible.

The current script reports both contended and uncontended figures, because a
real serving workload is contended and a spec sheet is not.

## Ornith-1.5-35B-A3B

[ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)
is a hybrid-attention MoE (256 experts, 8 active, ~3B active parameters) with
a 256k context, a vision encoder, and a multi-token-prediction head. Measured
on the same card, the same day, with the same harness and the same 150 W cap
as the Qwen3-Next-80B above. Raw data: `bench/data/ornith35b_*`.

### Against the 80B, engine-matched

Both on vLLM 0.27.1. Ornith uses the official FP8 checkpoint (36.7 GiB).

| | Ornith FP8 | Ornith FP8 + MTP k=2 | Qwen3-Next-80B W4A16 |
|---|---|---|---|
| decode, single stream | 122.5 tok/s | **150.5** | 115.7 |
| decode, 4 concurrent | **345.4** | 273.1 | 351.6 |
| decode, 8 concurrent | **597.3** (8 slots) | 252.0 | 352.4 (4 slots) |
| prefill, ~8.9k tokens | **9,145** | 8,917 | 6,800-7,950 |
| decode at 24k context | **117.8** | 74.4 | 111.9 |
| TTFT, 24k prefix, cold / warm | **3.06 s / 0.13 s** | 3.16 s / 0.32 s ¹ | 3.39 s / 0.14 s |
| J/token, 8 concurrent | **0.241** | 0.321 | 0.412 |
| weights | 36.7 GiB | 36.7 GiB | 40.9 GiB |
| 4-turn recall probe | 4/4 | not run | 3/4 |

¹ From the 24k decode run's two requests. The other two columns come from the
four-turn shared-prefix run. Both are a cold request followed by a warm prefix
hit, but they are not the same test.

Without MTP, Ornith matches or beats the 80B on every row except 4-concurrent
decode, where it is 2% behind, and it does so with 4.2 GiB less in weights.

### FP8 runs on this card

The checkpoint is `compressed-tensors` W8A8 float. The 170HX has no FP8 tensor
cores, and vLLM handles that correctly: it routes the checkpoint to the
weight-only `w8a16` scheme, so the weights stay FP8 and the maths runs in bf16.

Two environment faults stopped it loading at first. Neither is an FP8 or
`sm_80` limitation:

| error | cause | fix |
|---|---|---|
| `failed to open libnvrtc-builtins.so.13.3` | the JIT loads nvrtc from `CUDA_HOME` but its matching builtins library is not on the loader path | `LD_LIBRARY_PATH=$CUDA_HOME/lib` |
| `Ninja is required to load C++ extensions` | ninja is installed in the venv, but launching `vllm` by absolute path leaves the venv's `bin` off `PATH` | `PATH=$VENV/bin:$PATH` |

### The engine mattered more than the quantization

The same model on llama.cpp (`qwen35moe`) looks far worse, and almost all of
the gap is the engine:

| | llama.cpp Q4_K_M | llama.cpp Q8_0 | vLLM FP8 |
|---|---|---|---|
| decode, single stream | 114.5 | 104.7 | 122.5 |
| decode, 8 concurrent | 268.6 | 274.7 | 597.3 |
| prefill | 2,333 | 2,333 | 9,145 |
| TTFT, 24k prefix, cold | 11.5 s | 11.5 s | 3.06 s |

Halving the weight bits on llama.cpp (Q8_0 to Q4_K_M) bought 9% single-stream
and nothing at concurrency or prefill. Changing engine quadrupled prefill and
doubled 8-concurrent throughput. llama.cpp also ignores the checkpoint's MTP
tensors (`model has unused tensor blk.40.nextn.*`), so there is no
speculative decoding on that path.

If you compare models across engines, you are mostly measuring the engines.

### MTP speculative decoding

The MTP head ships in the checkpoint and needs no separate drafter. vLLM
builds the draft config from `model_type: qwen3_5_moe` on its own:

```
--speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

| | no MTP | MTP k=2 | MTP k=1 + batch 8192 |
|---|---|---|---|
| decode, single stream | 122.5 | **150.5** | 141.8 |
| decode, 4 concurrent | **345.4** | 273.1 | 260.7 |
| decode, 8 concurrent | **597.3** | 252.0 | 572.4 |
| prefill | 9,145 | 8,917 | **11,183** |
| decode at 24k context | **117.8** | 74.4 | 69.4 |
| KV cache | 625,916 tokens | ~400,000 | 411,420 |

**Raise the batch budget whenever you enable MTP.** Speculative decoding makes
vLLM set `max_num_scheduled_tokens` to 2048. At k=2 that dropped 8-concurrent
decode to 252 tok/s while the card drew only 80.9 W, which means it was
stalled, not busy. Adding `--max-num-batched-tokens 8192` brought it back to
572 and gave 11,183 tok/s prefill, the fastest prefill measured on this card.

**MTP costs about 40% at long context**, in both k=1 and k=2, so it is not a
tuning artefact. 4-concurrent decode also stayed about 25% down in both
configurations, and we do not have an explanation for why 8-concurrent
recovered and 4-concurrent did not.

There is no single best setting, so choose by workload:

| workload | configuration |
|---|---|
| single stream, short prompts | MTP k=2 |
| long context | no MTP |
| high concurrency | no MTP, or MTP k=1 with a raised batch budget |
| fastest prefill | MTP k=1 with a raised batch budget |

### Reasoning budget

Ornith is a reasoning model: every reply opens with a `<think>` block.
`bench/reasoning_budget.py` measures how much it thinks and what limiting it
costs, across 10 prompts (8 with checkable answers), 6 conditions and 3
repetitions, at the card's recommended temperature 0.6.

| condition | correct | empty answers | reasoning tokens, median / max |
|---|---|---|---|
| unbounded | **24/24** | 0 | 96 / 447 |
| thinking disabled | 20/24 | 0 | 1 / 1 |
| `thinking_token_budget` 1024 | **24/24** | 0 | 101 / 634 |
| `thinking_token_budget` 256 | 22/24 | 0 | 102 / 257 |
| `thinking_token_budget` 64 | 22/24 | 0 | 65 / 65 |
| `max_tokens` 256 | 19/24 | **7** | 100 / 256 |

**Left alone, it kept its reasoning short.** The median was 96 tokens, under a
second at single-stream speed, and nothing came near the 16k cap. The longest
reasoning was on the number-theory prompt and on an open-ended one-sentence
prose prompt (339 tokens).

**A tight budget gives confident wrong answers, not hedges.** At 256 and 64,
the hardest prompt (answer 301) came back as a flat "121" in 4 of 6 runs.
Disabling thinking was worse than any budget: it got a three-step word problem
wrong every time.

**Never use `max_tokens` as the limit.** A 256 cap with no thinking budget
returned 7 empty answers out of 30, each with `finish_reason=length`, because
the cap landed mid-thought. The response looks normal and contains nothing.

If you want a limit, use `thinking_token_budget`. It is an exact hard cap (64
gave at most 65 tokens: the budget plus the forced end-of-thinking token), and
1024 cost nothing here. It needs `--reasoning-parser qwen3` and
`VLLM_USE_V2_MODEL_RUNNER=0`, because the V2 runner rejects the parameter.

**Caveat:** these are short, single-turn prompts. Ornith is trained for long
agentic coding, which this test does not exercise, so it says nothing about
reasoning length on those tasks.

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
# reasoning length and budget cost (needs --reasoning-parser on the server)
python3 bench/reasoning_budget.py http://localhost:8000 your-model out.jsonl 3

# prefill, freshly randomised prompt per rep
python3 bench/prefill.py http://localhost:8000 your-model none 10000 3

# single point
python3 bench/bench.py --url http://localhost:8000 --model your-model \
        --gpu <nvidia-smi index> --ntok 1500 --conc 1 --label baseline

# power sweep: gpu, url, model, stock_watts, concurrency, ntokens, caps...
./bench/sweep.sh 1 http://localhost:8000 your-model 250 1 1200 250 200 175 150 125
```

`sweep.sh` needs passwordless `sudo nvidia-smi` to set caps, and restores the
stock limit on exit including on interrupt. Set `VLLM_API_KEY` if your endpoint
requires auth.
