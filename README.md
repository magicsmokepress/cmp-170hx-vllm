# CMP 170HX + vLLM

**64 GB of HBM2e on a dead mining card, serving an 80B model at 121 tok/s and 150 watts.**

The NVIDIA CMP 170HX is a cryptocurrency mining card built on the GA100 die,
the same silicon as the A100. It has 64 GB of HBM2e at 1.5 TB/s. It has no
display outputs, no video encoders, a PCIe link hobbled to Gen2 x8, and a
locked VBIOS. Nobody wants it. That is the point.

This repository documents a working, measured setup: what the card actually
does under vLLM, the configuration that gets there, the traps that cost time,
and the raw benchmark data behind every number.

Everything here was measured on real hardware, not estimated. The harness and
the raw JSONL are in [`bench/`](bench/) so you can check the arithmetic.

---

## Headline numbers

Two models, both on one 170HX, both under vLLM, both re-measured on the same
day with the same harness. The 27B is the fast one; the 80B is the one that
only fits because the card has 64 GB.

| | **Qwen3.8-27B** W4A16 + DFlash2 | **Qwen3-Next-80B-A3B** W4A16 |
|---|---|---|
| architecture | 27B dense | 80B MoE, ~3B active |
| weights on disk | 15.8 GB | 40.9 GB |
| **decode, single stream** | **127 tok/s** | **116 tok/s** |
| **decode at 24k context** | **83 tok/s** | **112 tok/s** |
| **prefill (~8.9k tokens)** | **1768 tok/s** | **6800-7950 tok/s** |
| decode, 4 concurrent | 204 tok/s | 352 tok/s |
| decode, 8 concurrent | 235 tok/s | 352 tok/s (4 slots, saturated) |
| TTFT, 24k prefix, cold | 14.3 s | 3.4 s |
| TTFT, 24k prefix, warm | 0.56 s | 0.14 s |
| efficiency, single stream | 1.12 J/token | 1.18 J/token |
| efficiency, 8 concurrent | 0.63 J/token | 0.41 J/token |
| power drawn | 143 W (capped), 194 W uncapped | 137-145 W, never hits the cap |
| temperature | 50-56 C | 49-51 C |
| weight load time | 9.1 s (1.73 GB/s) | 68 s (0.60 GB/s) |
| KV cache | 57.7k tokens at `--gpu-memory-utilization 0.65` | 242.5k tokens at 0.75 |
| 4-turn recall accuracy | 4/4 | 3/4 |

All rows measured 2026-09-13 at the 150 W cap unless noted, so the two columns
are directly comparable. The 27B's "uncapped" power figure is from the cap
sweep below.

### What the table says

**The 80B wins almost everywhere except short-prompt decode.** It prefills
nearly 4x faster, decodes 35% faster at 24k context, hits 3x the concurrent
throughput, and reaches a cold 24k prompt in 3.4 s against the 27B's 14.3 s.
The 27B's advantage is a narrow 9% on single-stream short-prompt decode.

That is not the ranking most people expect from "27B vs 80B", and the reason is
architectural: the 80B is an MoE with ~3B active parameters, so it does far
less work per token than a 27B dense model, while still getting the quality of
a much larger network. On this card the big model is the fast model.

**Concurrency is where the card earns its keep.** Both models roughly triple
their throughput from 1 to 8 concurrent requests, and efficiency improves
correspondingly: the 80B goes from 1.18 to 0.41 J/token. The 80B saturates at
352 tok/s because it is configured with `--max-num-seqs 4`; raising
concurrency past the slot count buys exactly nothing, which the C4 and C8 rows
show precisely.

**The 27B's decode rate is unusually noisy.** Six repetitions at a fixed cap
spread from 109 to 136 tok/s. That spread is the speculative-decoding
acceptance rate moving with sampling, not measurement error, and it is why the
27B needs medians over many runs where the 80B does not. On trivially
predictable output the same stack reaches 364 tok/s, so any benchmark of this
model using a repetitive prompt overstates it by more than 2x.

**Recall was not equal.** Asked four questions against a 24k-token document,
the 27B answered 4/4 correctly and the 80B miscounted the paragraphs (265
against 266). One probe is not an evaluation, but it is a reminder that these
throughput numbers say nothing about quality.

For comparison on the same host and harness: an RTX 4090 runs at 1.82 J/token
and an RTX 3090 at 4.11 J/token, against the 170HX's 1.12-1.18. The 170HX is
the most efficient card in the machine by a wide margin, and the only one that
can hold an 80B model at all.

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

**Getting data in and out.** The PCIe link is the defining constraint:

```
$ nvidia-smi -q -i <170hx> | grep -A6 "PCIe Generation"
    PCIe Generation
        Max          : 2
        Current      : 2
        Device Max   : 1     <- the card advertises Gen1
        Host Max     : 4
    Link Width
        Max          : 16x
        Current      : 8x
```

Measured host transfer with a compute kernel held running so the card cannot
downclock: **1.53 GB/s H2D, 1.51 GB/s D2H.** A 3090 in the same chassis does
24 GB/s. The link never trains higher, at any load.

Consequences:

- Loading 41 GB of weights takes **68 seconds** with 98% of the file already
  in page cache (verified with `mincore`), and up to 130 s cold from disk.
  Budget two minutes for a service restart, and set
  `TimeoutStartSec=900` in systemd or the unit will be killed mid-load.
- Tensor parallelism across two of these would be miserable. Use one card
  per model.
- Anything that streams tensors from host RAM per token (CPU offload,
  layer swapping) is off the table. Fit the model in VRAM or pick another card.

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
2. [docs/setup.md](docs/setup.md): driver, CUDA, vLLM, and the two pinning
   rules that will otherwise send your job to the wrong GPU.
3. [docs/benchmarks.md](docs/benchmarks.md): every number above, with method.
4. [docs/tuning.md](docs/tuning.md): why we do not overclock this card, with
   the measurement that closed the question.
5. [configs/](configs/): the systemd units, ready to adapt.
6. [bench/](bench/): the harness and raw data. Reproduce it on your card.

---

## Test system

| | |
|---|---|
| Card | CMP 170HX, `10de:20c2`, 64 GB HBM2e, VBIOS 92.00.67.00.01, compute capability 8.0 |
| Host | AMD Ryzen Threadripper PRO 3945WX, 128 GB DDR4 ECC |
| OS | Ubuntu 26.04 LTS, kernel 7.0 |
| Driver | 610.43.02 |
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
