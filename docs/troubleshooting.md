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

## The card reads Gen1 x16 at idle

That is normal power-state downtraining and affects every modern NVIDIA card,
not just this one. It trains up on the first compute kernel.

The 170HX is different in that it genuinely will not go past **Gen2 x8** even
under full load. Distinguish the two with `nvidia-smi -q`, which separates the
ends of the negotiation:

```
PCIe Generation
    Max          : 2
    Current      : 2
    Device Max   : 1
    Host Max     : 4
```

`Device Max` below `Host Max` means the card is the limit, not the slot or a
signal-integrity fallback. Check `Replays Since Reset` too: a nonzero and
climbing count would point at signal integrity, and ours reads 0.

Measure it properly with [`bench/pcie_bw.py`](../bench/pcie_bw.py), which holds
a compute kernel running so the card is in P0.

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
