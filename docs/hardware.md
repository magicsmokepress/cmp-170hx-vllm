# The hardware

## Identifying the card

The 170HX does not report a marketing name. `nvidia-smi` calls it
"NVIDIA Graphics Device". Identify it by PCI id and memory size:

```
$ lspci -nn | grep -i nvidia
03:00.0 3D controller [0302]: NVIDIA Corporation Device [10de:20c2]

$ nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,memory.total,vbios_version \
             --format=csv
1, NVIDIA Graphics Device, GPU-aec84db3-..., 00000000:03:00.0, 65536 MiB, 92.00.67.00.01
```

Note `3D controller`, not `VGA compatible controller`. It has no display
engine. It reports compute capability **8.0** (GA100), so build for `sm_80`.

Useful consequence: because it is `sm_80` and not `sm_86`/`sm_89`, wheels
built only for consumer Ampere/Ada will not have a kernel for it. Most
mainstream builds (PyTorch, vLLM) do ship sm_80 because that is the A100, so
in practice this is a non-issue, but check if you build anything yourself.

## The PCIe link, and what it takes to make it usable

This is the part of the card most write-ups get wrong, including an earlier
version of this one. **The link you get out of the box is not the link
measured here.**

### Three different states

| state | speed | width | notes |
|---|---|---|---|
| stock card, stock driver | Gen1 | x4 (from an x16 capability) | the card's CYA bits pin the link to Gen1 |
| stock card, cmpunlocker | Gen2 | x4 | software retrain, no hardware change |
| **this card** | **Gen2** | **x8** | cmpunlocker **plus** a hardware capacitor mod |
| fully-wired card, cmpunlocker | Gen2 | x16 | upstream's logs show this; not reached here |

The Gen1 pin is a firmware decision, not a physical limit.
[cmpunlocker](https://github.com/bayley/cmpunlocker) clears the CYA bits inside
its unlock window, selects Gen2 in `XP_CFG0` and `XP_LCTRL2`, sets the root
port's target speed (both ends must agree), and triggers a directed speed
change. It runs at every driver initialisation. On this host you can watch it
happen:

```
$ journalctl -b -k | grep _cmpRetrainGen2
NVRM: GPU3 _cmpRetrainGen2: CMPUNLOCK: PCIe pre: CFG0=0x800c4c00 LCTRL2=0x00140036 CYA0=0x068731b7
NVRM: GPU3 _cmpRetrainGen2: CMPUNLOCK: PCIe root port LnkCtl2 @ 0x88: 0x0004 -> 0x0002
NVRM: GPU3 _cmpRetrainGen2: CMPUNLOCK: PCIe retrain done (polls=0): LinkCtrlStat=0x10820040 speed=2 width=x8
```

The **width** is a separate problem with a separate fix. cmpunlocker retrains
whatever lanes are physically present; upstream's own log shows `width=x16` on
a card where all of them are. This card reaches x8 because of a capacitor
modification to the board. x16 was not achievable on it.

So if you are pricing one of these: budget for the software unlock (free, but
it means running a patched driver with Secure Boot off) and understand that
the link width depends on a hardware modification you may or may not want to
do.

### What it reports

```
$ nvidia-smi -q -i <idx> | grep -A9 "GPU Link Info"
    PCIe Generation
        Max                  : 2
        Current              : 2
        Device Current       : 2
        Device Max           : 1     <- the card still advertises Gen1
        Host Max             : 4
    Link Width
        Max                  : 16x
        Current              : 8x
```

`Device Max: 1` with `Current: 2` is the signature of the software retrain: the
device's advertised capability is untouched, the live link is not. `Link Width
Max: 16x` is the capability, `8x` is what this board's lanes support.

### What it measures

`bench/pcie_bw.py`, 512 MiB buffers, 20 iterations:

| | H2D | D2H |
|---|---|---|
| uncontended | **3.18 GB/s** | 3.12 GB/s |
| contended (matmul running) | 2.50 GB/s | 2.46 GB/s |

3.18 GB/s is about 79% of Gen2 x8's 4 GB/s theoretical, which is a normal
efficiency. An RTX 3090 in the same chassis, measured with the same script,
does 24.90 GB/s at Gen4 x16. So the 170HX is roughly **7.8x slower to feed**,
not the 16x an earlier version of this document claimed from a broken
measurement.

Two ways to get this wrong, both of which we did:

- **Measuring with a compute kernel running** costs about 20%. You need one to
  warm an idle card into P0, but stop it before timing the copies.
- **Buffers under about 512 MiB** let per-copy overhead dominate. At 256 MiB
  with 10 iterations this same card read 1.53 GB/s, half the real figure.

### What it costs you

| operation | effect |
|---|---|
| Model load | **not link-bound** (see below) |
| systemd unit startup | still set `TimeoutStartSec=900`, for other reasons |
| Tensor parallel across 2 cards | avoid; one model per card |
| CPU offload / layer streaming | not viable, fit in VRAM |
| Inference itself | **unaffected**, weights and KV live in VRAM |

Model loading is slower than the link, not limited by it. The 80B loads 40.9 GB
at 0.60 GB/s and the 27B loads 15.8 GB at 1.73 GB/s, against a 3.18 GB/s
ceiling, so something in the loader dominates in both cases. See
[benchmarks.md](benchmarks.md#load-rate-is-model-dependent-not-just-link-dependent).

The last row is why the card is worth using at all. Once loaded, the link
carries only prompts and tokens, which are kilobytes.

## Power

Stock limit 250 W. **Whether the card approaches it depends entirely on the
workload**, and this is the one place where a single number would mislead you:

- Serving Qwen3-Next-80B (MoE, ~3B active), it never exceeded **152 W**.
- Serving Qwen3.8-27B with DFlash2 speculative decoding, it draws **194 W**.

Speculative decoding verifies a block of draft tokens in one pass, which is
compute-heavy, and compute is what a power cap restricts. A memory-bound
decode workload does not notice the cap; a speculative one does.

Swept caps from 250 W down to 125 W against both models. Raw data in
`bench/data/170hx_c1.jsonl` (80B) and `bench/data/27b_170hx_capsweep_c1.jsonl`
(27B).

**Qwen3-Next-80B, memory-bound, two reps per cap:**

| cap | tok/s | draw | J/token | SM clock |
|---|---|---|---|---|
| 250 W | 117.6 | 143 W | 1.22 | 1400 MHz |
| 200 W | 117.6 | 149 W | 1.27 | 1400 MHz |
| 175 W | 117.6 | 151 W | 1.29 | 1400 MHz |
| 150 W | 116.4 | 146 W | 1.25 | 1380 MHz |
| 125 W | 109.3 | 122 W | 1.12 | 1271 MHz |

Flat from 250 W to 150 W, because the card never got near the cap.

**Qwen3.8-27B with DFlash2 speculative decoding, medians (n per row):**

| cap | median tok/s | range | draw | SM clock |
|---|---|---|---|---|
| 250 W | 137.0 (n=9) | 121.6-156.1 | 194 W | 1382 MHz |
| 200 W | 133.7 (n=3) | 128.2-135.4 | 188 W | 1349 MHz |
| 175 W | 128.5 (n=3) | 120.5-137.2 | 166 W | 1277 MHz |
| 150 W | 127.1 (n=9) | 108.8-136.2 | 143 W | 1178 MHz |
| 125 W | 114.6 (n=3) | 110.9-120.2 | 121 W | 1005 MHz |

A different shape entirely. The card draws 194 W when allowed, the SM clock
tracks the cap all the way down, and 250 W to 150 W costs about 7% of median
throughput for 50 W. At 24k context the same cut costs more, roughly 16% (101.6
and 96.0 tok/s at 250 W against 81.5 and 85.3 at 150 W).

Note the ranges: individual runs overlap heavily between caps, because
speculative-decoding acceptance varies with sampling. Nine repetitions
separate 250 W from 150 W; three would not have. If you sweep this workload,
take medians over many runs.

**We set 150 W** on this host because the 80B is what it serves. That choice
would be wrong for a machine running the 27B, where 200 W is the knee. 125 W
is the efficiency optimum for both, if you care more about watts than latency.

Apply at boot with [`configs/gpu-power-caps.service`](../configs/gpu-power-caps.service).
Power limits reset on reboot and on every driver reload.

## Cooling

Passive heatsink, no fan, designed for a mining frame's front-to-back airflow.
It will thermally throttle on a desk. With a blower shroud pushing air through
it, short benchmark runs held 52-59 C. Our runs are 8-45 seconds, so those are
not steady-state numbers. A card held at load for an hour sits hotter.

`nvidia-smi` reports no fan speed (`Fan Speed: N/A`); there is no fan to
report. Watch `temperature.gpu` and the `HW Thermal Slowdown` clock event
reason instead.

## What is missing versus an A100

- No display outputs (3D controller, not VGA).
- No NVENC / NVDEC. Do not plan on video transcoding.
- No NVLink.
- No MIG.
- No ECC (`ecc.mode.current` reads `N/A`).
- VBIOS-locked clocks: SM tops out at 1410 MHz, memory at 1458 MHz (NDIV 54,
  the only supported value). See [tuning.md](tuning.md).
- PCIe pinned to Gen1 x4 in firmware, against the A100's Gen4 x16. Gen2 needs
  a patched driver and any width above x4 needs a hardware mod; see above.

What you keep: the GA100 die, 64 GB of HBM2e, and ~1.3 TB/s of real measured
bandwidth.
