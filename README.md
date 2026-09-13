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

Serving **Qwen3-Next-80B-A3B (W4A16)** on one 170HX under vLLM 0.27.1:

| metric | measured |
|---|---|
| prefill | **7400-7950 tok/s** (~10k-token prompt, no prefix cache) |
| decode, single stream | **121 tok/s** on prose |
| power under load | **143-151 W** |
| efficiency | **1.22 J/token** |
| temperature | 52-59 C (short runs, blower shroud) |
| KV cache at 64k ctx | 242k tokens in 5.85 GiB |

For comparison on the same host, same harness: an RTX 4090 runs at 1.82
J/token and an RTX 3090 at 4.11 J/token. The 170HX is the most efficient card
in the machine by a wide margin, and the only one that can hold an 80B model
at all.

Serving **Qwen3.8-27B (W4A16 + speculative decode)** on the same card:
152 tok/s single-stream, **102 tok/s at 24k context**, 278 tok/s at
concurrency 8, 24k-token prefix cold TTFT 11.8 s / warm 0.49 s.

Full detail and methodology: [docs/benchmarks.md](docs/benchmarks.md).

---

## What the card is good at

**Holding a big model.** 64 GB is more than two 3090s and needs no tensor
parallelism, no NVLink, no second PSU rail. A 43 GB W4A16 80B fits with 20 GB
left over.

**Memory bandwidth.** 1325 GB/s read, 1292 GB/s triad measured — 88.7% of the
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

1. [docs/hardware.md](docs/hardware.md) — identifying the card, PCIe, power,
   cooling, and what "Device Max: 1" costs you.
2. [docs/setup.md](docs/setup.md) — driver, CUDA, vLLM, and the two pinning
   rules that will otherwise send your job to the wrong GPU.
3. [docs/benchmarks.md](docs/benchmarks.md) — every number above, with method.
4. [docs/tuning.md](docs/tuning.md) — why we do not overclock this card, with
   the measurement that closed the question.
5. [configs/](configs/) — the systemd units, ready to adapt.
6. [bench/](bench/) — the harness and raw data. Reproduce it on your card.

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
