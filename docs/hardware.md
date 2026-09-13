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

## The PCIe link is the headline constraint

```
$ nvidia-smi -q -i <idx> | grep -A9 "GPU Link Info"
    PCIe Generation
        Max                  : 2
        Current              : 2
        Device Current       : 2
        Device Max           : 1
        Host Max             : 4
    Link Width
        Max                  : 16x
        Current              : 8x
```

The host slot is Gen4 x16. The card negotiates Gen2 x8 and stays there.

Measured, with a matmul kernel held running in a background thread so the card
cannot be sitting in a low power state (a pure DMA copy does **not** wake the
GPU, and benchmarking an idle card measures the idle link):

```
H2D: 1.53 GB/s
D2H: 1.51 GB/s
```

An RTX 3090 in the same chassis measures 24.1 GB/s H2D on the same test. The
170HX is roughly **16x slower to feed**.

What this costs you in practice:

| operation | effect |
|---|---|
| Loading a 41 GB W4A16 model | 68 s warm from page cache, up to 130 s cold |
| systemd unit startup | set `TimeoutStartSec=900` or it gets killed mid-load |
| Tensor parallel across 2 cards | avoid; one model per card |
| CPU offload / layer streaming | not viable, fit in VRAM |
| Inference itself | **unaffected**, weights and KV live in VRAM |

The last row is why the card is still worth using. Once loaded, the PCIe link
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
- PCIe Gen2 x8 as above, against the A100's Gen4 x16.

What you keep: the GA100 die, 64 GB of HBM2e, and ~1.3 TB/s of real measured
bandwidth.
