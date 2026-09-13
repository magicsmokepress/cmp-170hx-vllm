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

Two models, both on one 170HX, both under vLLM. The 27B is the fast one; the
80B is the one that only fits because the card has 64 GB.

| | **Qwen3.8-27B** W4A16 + DFlash2 | **Qwen3-Next-80B-A3B** W4A16 |
|---|---|---|
| architecture | 27B dense | 80B MoE, ~3B active |
| weights on disk | 22 GB + 1.8 GB drafter | 41 GB |
| **decode, single stream** | **152 tok/s** | **121 tok/s** |
| **decode at 24k context** | **102 tok/s** | not measured |
| **prefill (~10k tokens)** | not measured on this card | **7400-7950 tok/s** |
| concurrency 8, end to end | 278 tok/s | not measured (4 slots) |
| TTFT, 24k prefix, cold | 11.8 s | not measured |
| TTFT, 24k prefix, warm | 0.49 s | not measured |
| power under load | 245 W (at a 250 W cap) | 143-151 W (at a 150 W cap) |
| efficiency | not measured | 1.22 J/token |
| temperature | 72 C | 52-59 C |
| context served | 56k, 4 slots | 64k, 4 slots |
| VRAM left over | ~40 GB | ~20 GB |
| measured | 2026-08-23 | 2026-09-09 / 2026-09-12 |

Empty cells are honestly empty: the two models were benchmarked at different
times for different reasons, and neither run covered the other's axes. The
power figures are not comparable either: the 27B run predates the power sweep
that set the 150 W cap, so it was free to draw 245 W. Do not read the power
rows as a model-to-model comparison.

### What the table says

**The 27B decodes 26% faster; the 80B prefills.** For short prompts and chatty
turns the 27B wins. For long single-shot work, a document to summarise or a
large context to answer over, the 80B is the faster server end to end, because
an MoE with ~3B active parameters prefills at nearly 8000 tok/s. Pick by prompt
shape, not by parameter count.

**Neither number is the card's ceiling.** Both are single-stream. The 170HX has
bandwidth and VRAM to spare in both configurations, and the 27B at concurrency
8 more than doubles to 278 tok/s.

**The 27B's decode rate depends on speculative decoding accepting drafts.** On
trivially predictable output the same stack measures 364 tok/s; on real prose
it settles near 152-161. Any benchmark of this model using a repetitive prompt
overstates it by more than 2x. The 80B has no drafter, so its 121 tok/s has no
such caveat.

For comparison on the same host and harness: an RTX 4090 runs at 1.82 J/token
and an RTX 3090 at 4.11 J/token, against the 170HX's 1.22. The 170HX is the
most efficient card in the machine by a wide margin, and the only one that can
hold an 80B model at all.

The same 27B stack on other cards, so the card is the only variable: 133-135
tok/s on a 3090, 157-163 on a 4090, against the 170HX's 152. At 24k context it
is 88 on the 3090 against the 170HX's 102. Full three-way table in
[docs/benchmarks.md](docs/benchmarks.md#the-27b-on-the-170hx-head-to-head).

Full detail and methodology: [docs/benchmarks.md](docs/benchmarks.md).

---

## What the card is good at

**Holding a big model.** 64 GB is more than two 3090s and needs no tensor
parallelism, no NVLink, no second PSU rail. A 43 GB W4A16 80B fits with 20 GB
left over.

**Memory bandwidth.** 1325 GB/s read, 1292 GB/s triad measured, 88.7% of the
1493 GB/s theoretical at its clock. Single-stream LLM decode is
bandwidth-bound, and this card has A100-class bandwidth.

**Performance per watt.** It never drew more than 152 W against its 250 W
stock cap. Capping it to 150 W costs nothing measurable.

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

- Loading 43 GB of weights takes **79-130 seconds**, even from page cache.
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
