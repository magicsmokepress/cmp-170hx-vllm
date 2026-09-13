# CMP 170HX + vLLM

**64 GB of HBM2e, serving Ornith-1.5-35B at 150 tok/s and Qwen3-Next-80B at 116 tok/s under a 150 W cap.**

The NVIDIA CMP 170HX is a cryptocurrency mining card built on the GA100 die,
the same silicon as the A100. It has 64 GB of HBM2e at 1.5 TB/s. It also has no
display outputs, no video encoders, a PCIe link pinned to Gen1 x4 in firmware,
and a locked VBIOS.

**Read [the PCIe section](docs/hardware.md#the-pcie-link-and-what-it-takes-to-make-it-usable)
before you buy one.** The numbers here come from a card running a patched
driver that retrains the link to Gen2, on a board with a hardware capacitor
modification that widens it to x8. Out of the box you get neither.

This repository documents a working, measured setup: what the card actually
does under vLLM, the configuration that gets there, the traps that cost time,
and the raw benchmark data behind every number.

Everything here was measured on real hardware, not estimated. The harness and
the raw JSONL are in [`bench/`](bench/) so you can check the arithmetic.

---

## Headline numbers

Three models, all on one 170HX, all under vLLM, all measured on the same day
with the same harness at the same 150 W cap.

| | **Qwen3.8-27B** W4A16 + DFlash2 | **Qwen3-Next-80B-A3B** W4A16 | **Ornith-1.5-35B-A3B** FP8 |
|---|---|---|---|
| architecture | 27B dense | 80B MoE, ~3B active | 35B MoE, ~3B active |
| weights on disk | 15.8 GB | 40.9 GB | 36.7 GiB |
| **decode, single stream** | **127 tok/s** | **116 tok/s** | **122.5 tok/s** (150.5 with MTP) |
| **decode at 24k context** | **83 tok/s** | **112 tok/s** | **117.8 tok/s** |
| **prefill (~8.9k tokens)** | **1768 tok/s** | **6800-7950 tok/s** | **9145 tok/s** |
| decode, 4 concurrent | 204 tok/s | 352 tok/s | 345 tok/s |
| decode, 8 concurrent | 235 tok/s | 352 tok/s (4 slots, saturated) | 597 tok/s (8 slots) |
| TTFT, 24k prefix, cold | 14.3 s | 3.4 s | 3.1 s |
| TTFT, 24k prefix, warm | 0.56 s | 0.14 s | 0.13 s |
| efficiency, single stream | 1.12 J/token | 1.18 J/token | 1.13 J/token |
| efficiency, 8 concurrent | 0.63 J/token | 0.41 J/token | 0.24 J/token |
| power drawn | 143 W (capped), 194 W uncapped | 137-145 W, never hits the cap | 138-144 W, never hits the cap |
| temperature | 50-56 C | 49-51 C | 46-49 C |
| weight load time | 9.1 s (1.73 GB/s) | 68 s (0.60 GB/s) | 24 s (1.51 GiB/s), cache state not checked |
| KV cache | 57.7k tokens at `--gpu-memory-utilization 0.65` | 242.5k tokens at 0.75 | 625.9k tokens at 0.80 |
| 4-turn recall accuracy | 4/4 | 3/4 | 4/4 |

All rows measured 2026-09-13 at the 150 W cap unless noted, so the columns
are directly comparable. The 27B's "uncapped" power figure is from the cap
sweep below. Ornith's main column is without speculative decoding; its MTP
figures, which trade long-context speed for single-stream speed, are in
[docs/benchmarks.md](docs/benchmarks.md#ornith-15-35b-a3b).

### What the table says

**Ornith leads on almost every row.** It prefills fastest, decodes fastest at
24k context, reaches a cold 24k prompt soonest, and at 8 concurrent requests
delivers 597 tok/s at 0.24 J/token, the highest throughput measured on this
card. It trails the 27B by 4.5 tok/s on single-stream decode without MTP and the 80B by 2% at 4 concurrent requests. With MTP k=2 it
reaches 150.5 tok/s single-stream, the fastest of the three, at the cost of
about 40% at long context.

**Among the Qwen models, the 80B beats the 27B almost everywhere except
short-prompt decode.** It prefills nearly 4x faster, decodes 35% faster at 24k
context, and reaches a cold 24k prompt in 3.4 s against the 27B's 14.3 s. The
27B's advantage is a narrow 9% on single-stream short-prompt decode.

That is not the ranking most people expect from parameter counts, and the
reason is architectural: the 80B and Ornith are MoEs with ~3B active
parameters, so each does far less work per token than a 27B dense model. On
this card the MoE models are the fast models.

**Concurrency is where the card earns its keep.** Going from 1 to 8
concurrent requests multiplies throughput by 1.9x for the 27B, 3.0x for the
80B and 4.9x for Ornith, and efficiency improves with it: Ornith goes from 1.13
to 0.24 J/token. The 80B
saturates at 352 tok/s because it is configured with `--max-num-seqs 4`;
raising concurrency past the slot count buys exactly nothing, which its C4 and
C8 rows show precisely.

**Speculative decoding makes decode rates noisy.** Six repetitions of the 27B
at a fixed cap spread from 109 to 136 tok/s. That spread is the
speculative-decoding acceptance rate moving with sampling, not measurement
error, and it is why the 27B needs medians over many runs where the 80B does
not. On trivially predictable output the same stack reaches 364 tok/s, so any
benchmark of this model using a repetitive prompt overstates it by more than
2x.

**Recall was not equal.** Asked four questions against a 24k-token document,
the 27B and Ornith answered 4/4 correctly and the 80B miscounted the paragraphs
(265 against 266). One probe is not an evaluation, but it is a reminder that
these throughput numbers say nothing about quality.

**Ornith is a reasoning model, and how you limit that matters.** Left
unbounded on short prompts it reasoned for a median 96 tokens and got 24/24
right. `thinking_token_budget` caps it exactly; tight budgets turned the hardest
prompt into a confident wrong answer, and using `max_tokens` as the limit
returned empty answers. See
[the reasoning budget results](docs/benchmarks.md#reasoning-budget).

For comparison on the same host and harness: an RTX 4090 runs at 1.82 J/token
and an RTX 3090 at 4.11 J/token single-stream, against the 170HX's 1.12-1.18.
The 170HX is the most efficient card in the machine by a wide margin, and the
only one that can hold an 80B model at all.

Full detail and methodology: [docs/benchmarks.md](docs/benchmarks.md).

---

## What the card is good at

**Holding a big model.** 64 GB is more than two 3090s and needs no tensor
parallelism, no NVLink, no second PSU rail. A 43 GB W4A16 80B fits with 20 GB
left over.

**Memory bandwidth.** 1325 GB/s read, 1292 GB/s triad measured, 88.7% of the
1493 GB/s theoretical at its clock. Single-stream LLM decode is
bandwidth-bound, and this card has A100-class bandwidth.

**Performance per watt.** 1.12-1.18 J/token single-stream, and 0.41 J/token
at concurrency 8. The best in the machine, against 1.82 for a 4090 and 4.11
for a 3090.

Whether you can cap it for free depends on the workload. Serving the 80B it
never drew more than 152 W against a 250 W cap, so a 150 W cap costs nothing.
Serving the 27B with speculative decoding it draws 194 W and a 150 W cap costs
about 7% of single-stream throughput and 16% at 24k context. Speculative
decoding adds a compute-heavy verify step, and compute is the one thing a cap
actually restricts. Measure your own workload before capping.

## What the card is bad at

**Getting data in and out.** Even modified, the link is the weak point:

| | this card (Gen2 x8) | RTX 3090 (Gen4 x16) |
|---|---|---|
| H2D, uncontended | 3.18 GB/s | 24.90 GB/s |
| D2H, uncontended | 3.12 GB/s | 22.11 GB/s |

Same script, same chassis, same day: about **7.8x slower to feed**. And that is
the modified card. Stock, the link is Gen1 x4, roughly a quarter of the lanes
and half the clock.

Consequences:

- Tensor parallelism across two of these would be miserable. Use one card
  per model.
- Anything that streams tensors from host RAM per token (CPU offload,
  layer swapping) is off the table. Fit the model in VRAM or pick another card.
- Model loading is slow, though **not because of the link**: both models here
  load well under the link ceiling, so the loader dominates.
- Inference itself is unaffected. Once loaded, the link carries only prompts
  and tokens.

**Compute, relatively.** llama.cpp extracts only ~728 GB/s effective against
the 1325 GB/s the memory can deliver, so for that workload the card is
compute-bound, not bandwidth-bound. Prefill is slower than a 4090 per FLOP.
It is a big-memory card, not a fast card.

**No display output, no NVENC/NVDEC.** You need another GPU or onboard video
to boot. It is a pure compute device.

**Cooling.** Passive heatsink designed for a mining rig's wind tunnel. It
needs forced air. Ours sits behind a blower and holds 52-59 C in short runs;
expect hotter under sustained load.

---

## Start here

1. [docs/hardware.md](docs/hardware.md): identifying the card, PCIe, power,
   cooling, and what "Device Max: 1" costs you.
2. [docs/setup.md](docs/setup.md): the full vLLM setup. Driver, CUDA, getting
   a model that fits, a working invocation with every flag explained, how to
   size the memory knob, how to verify it is really serving, and the two
   pinning rules that will otherwise send your job to the wrong GPU.
3. [docs/benchmarks.md](docs/benchmarks.md): every number above, with method,
   including the Ornith engine, MTP and reasoning budget results.
4. [docs/tuning.md](docs/tuning.md): why we do not overclock this card, with
   the measurement that closed the question.
5. [configs/](configs/): the systemd units, ready to adapt.
6. [bench/](bench/): the harness and raw data. Reproduce it on your card.

---

## Test system

| | |
|---|---|
| Card | CMP 170HX, `10de:20c2`, 64 GB HBM2e, VBIOS 92.00.67.00.01, compute capability 8.0 |
| Card mods | capacitor mod for PCIe x8; [cmpunlocker](https://github.com/bayley/cmpunlocker) patched driver for the Gen2 retrain |
| Host | AMD Ryzen Threadripper PRO 3945WX, 128 GB DDR4 ECC |
| OS | Ubuntu 26.04 LTS, kernel 7.0 |
| Driver | 610.43.02, patched by cmpunlocker (runs at every driver init) |
| Stack | vLLM 0.27.1, torch 2.13.0+cu130, CUDA 13.0 |
| Also in the box | RTX 4090, 2x RTX 3090 (used as comparison baselines) |

---

## License

MIT. See [LICENSE](LICENSE).

Related third-party work referenced in these docs:
[bayley/cmpunlocker](https://github.com/bayley/cmpunlocker) (GPL-2) for the
VBIOS/clock-unlocking research and the `nvidia_bench` micro-benchmark. Nothing
from it is vendored here.

This is AI-assisted work.
