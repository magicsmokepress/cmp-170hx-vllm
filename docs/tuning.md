# Do not overclock this card

Short version: we measured it, there is no headroom to unlock, and the known
overclocking path has a documented silent-corruption failure mode. The
performance headroom on a 170HX is in the serving stack, not the silicon.

## There is no constraint to relieve

Under sustained load:

```
SM clock          1410 MHz   <- the VBIOS maximum
Memory clock      1458 MHz   <- NDIV 54, the only supported value
Power             179 W of a 250 W limit
Temperature       52 C
Throttle reasons  all inactive
```

Every clock event reason reads `Not Active`: no SW power cap, no HW slowdown,
no thermal, no sync boost. The card is already running at its ceiling with
thermal and power headroom to spare. There is nothing being held back.

## Raising the memory clock would not help anyway

Measured bandwidth is 1325 GB/s read, 1292 GB/s triad, which is 88.7% of the
1493 GB/s theoretical at the stock clock. Meanwhile llama.cpp extracts only ~728 GB/s
effective. The engine is leaving 45% of the available bandwidth unused, so the
card is **compute-bound for that workload, not bandwidth-bound**, and more
memory clock has nothing to push into.

This is the same conclusion the power sweep reached from the other direction:
throughput is flat while the SM clock varies from 1400 MHz down to 1380 MHz,
and on the 4090 flat across a 2730 to 1416 MHz range. Clock is not what
produces tokens in single-stream decode.

[bayley/cmpunlocker](https://github.com/bayley/cmpunlocker)'s own README says
the same thing: "single-stream decode is not bandwidth-bound, so even the
stable OC buys ~0 there."

## The failure mode is silent

`170tune`'s published matrices document VRAM corruption at operating points
that pass every benchmark cleanly:

- `+325/1400` corrupts memory **while completing benchmarks without error**.
- `+300/1350` passed a 4/4 validation gate, then served for a full day, then
  threw an Xid 13 on an hour-long soak.

An overclock that computes wrong answers and reports success is worse than no
overclock, especially under an inference server where a corrupted weight shows
up as degraded output quality rather than a crash.

Real downside, zero measured upside.

## If you revisit it anyway

**Do not run `cmpunlocker install.sh --mclk-ndiv=N`.** It bakes a non-stock
clock into the driver's devinit, so it applies at every boot. `170tune` then
hard-refuses to tune on top of it via its mclk misclassification guard, and you
have a machine that comes up wrong with no easy way back.

Use `170tune`'s live BAR0 path instead. The card always boots stock, so a bad
profile is masked by a reboot rather than bricking a headless machine you reach
over ssh. That path also needs `iomem=relaxed` on the kernel command line,
which is not set by default.

## Where the headroom actually is

Everything that moved the numbers on this card was software:

- Enabling prefix caching on a hybrid model (`--mamba-cache-mode align`):
  24k-token TTFT from 11.8 s to 0.49 s.
- Choosing vLLM over llama.cpp for long context: 102 vs 39 tok/s at 24k.
- Sizing `--gpu-memory-utilization` to the real prompt distribution rather than
  the maximum context.
- Speculative decoding, where the model supports a drafter.

None of that requires touching a clock.
