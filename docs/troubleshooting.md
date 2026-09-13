# Troubleshooting

## "vLLM crashes on startup" / the unit keeps restarting

Almost always the start timeout. 41 GB of W4A16 weights take **68 seconds** to
load even with the file already in page cache, up to 130 s cold, and the
service needs **104 seconds** total before it answers. A default 90-second
timeout kills it mid-load. Set `TimeoutStartSec=900`.

Confirm by looking for the loader line in the journal:

```
INFO [default_loader.py] Loading weights took 111.61 seconds
```

If that line never appears before the kill, it is the timeout.

## The job landed on the wrong GPU

**CUDA orders devices by capability, not by PCI bus.** `CUDA_VISIBLE_DEVICES=1`
is not `nvidia-smi -i 1`. On our machine torch's index 3 was the card that
`nvidia-smi` calls index 1.

Pin by UUID, always:

```bash
nvidia-smi --query-gpu=uuid,pci.bus_id,memory.total --format=csv
export CUDA_VISIBLE_DEVICES=GPU-aec84db3-...
```

Verify from inside the process with
`torch.cuda.get_device_properties(i).pci_bus_id`. It is decimal, so bus `0x42`
prints as `66`.

The symptom that gives it away: you sample `nvidia-smi -i N` (which orders by
bus) while loading a card CUDA chose by speed, and the power and clock readings
belong to a completely different GPU than the one doing the work.

## The card reads Gen1, or a narrow width

Three different causes, and they need telling apart.

**Idle downtraining** affects every modern NVIDIA card. It trains up on the
first compute kernel. Harmless.

**The 170HX's firmware Gen1 pin** is not downtraining: the card's CYA bits hold
the link at Gen1 regardless of load. Only a patched driver
([cmpunlocker](https://github.com/bayley/cmpunlocker)) clears them. If you have
it installed, confirm it actually ran:

```
$ journalctl -b -k | grep _cmpRetrainGen2
NVRM: ... PCIe retrain done (polls=0): LinkCtrlStat=0x10820040 speed=2 width=x8
```

No such line means no retrain, and you are on Gen1 whatever else is true.

**Width** is physical. All stock CMP 170HX report `x4 (downgraded)` from an x16
capability. Getting x8 or x16 requires a hardware modification to the board;
this card has a capacitor mod that reaches x8. cmpunlocker retrains whatever
lanes are present and cannot add any.

`nvidia-smi -q` separates the ends of the negotiation, which is how you tell a
firmware pin from a slot problem:

```
PCIe Generation
    Max          : 2
    Current      : 2
    Device Max   : 1     <- advertised capability, untouched by the retrain
    Host Max     : 4
Link Width
    Max          : 16x
    Current      : 8x
```

`Device Max: 1` alongside `Current: 2` is the signature of a successful
software retrain, not a fault. Check `Replays Since Reset` too: a nonzero and
climbing count would point at signal integrity, and ours reads 0.

Measure the result with [`bench/pcie_bw.py`](../bench/pcie_bw.py), and read its
header first: it is easy to under-measure this by 2x.

## Serving one request at a time / terrible concurrency

Read the two KV lines in the startup log:

```
INFO [gpu_worker.py]     Available KV cache memory: 0.35 GiB
INFO [kv_cache_utils.py] GPU KV cache size: 14,000 tokens
```

`0.35 GiB` means the model plus overhead consumed almost the whole pool and
there is nothing left for batching. Raise `--gpu-memory-utilization`, or lower
`--max-model-len` *and* raise utilization. The server starts cleanly either
way, which is what makes this easy to miss.

## Throttling / thermals

The card has no fan and reports `Fan Speed: N/A`. It is designed for a mining
frame's wind tunnel and will throttle on an open bench. Watch:

```bash
nvidia-smi -q -i <idx> | grep -A9 "Clocks Event Reasons"
```

All reasons should read `Not Active`. `HW Thermal Slowdown: Active` means you
need more air, not a different config.

## The server answers but produces nothing

A wedged vLLM process still responds to `/v1/models` with HTTP 200 and exits 0
on a health check. Health-check on an actual completion and assert on
`usage.completion_tokens`, not on the endpoint being reachable.

If a completion hangs and `dmesg` shows an `Xid 79` (GPU fallen off the bus),
every *new* CUDA process on that card will fail while existing ones appear
fine. Only a reboot recovers it.

## Benchmark numbers that look too good

See the three rules in [benchmarks.md](benchmarks.md#method). The usual
culprits are a repetitive prompt inflating a speculative-decoding server by
more than 2x, prefix caching serving your "cold" prefill run, and counting
stream chunks instead of `usage.completion_tokens`.

## Someone told you vLLM cannot prefix-cache hybrid models

It can. It is opt-in:

```
--enable-prefix-caching --mamba-cache-mode align
```

We took the "cannot cache" claim at face value, migrated a service to
llama.cpp over it, and paid a 2.2x decode regression at long context for
months before measuring.
