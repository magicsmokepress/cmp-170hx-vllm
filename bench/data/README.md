# Raw benchmark data

One JSON object per line, produced by `../bench.py`. Fields:

| field | meaning |
|---|---|
| `tokens` | completion tokens generated, from `usage.completion_tokens` |
| `sec` | wall-clock seconds for the request(s) |
| `tok_s` | `tokens / sec` |
| `watts` | mean `power.draw` sampled at 100 ms across exactly that window |
| `j_per_tok` | `watts / tok_s` |
| `tok_s_per_100w` | throughput per 100 W |
| `sm_mhz` | mean SM clock over the window |
| `max_temp` | peak GPU temperature over the window |
| `psamples` | number of power samples in the window |
| `label` | `gpu<N>_cap<W>_c<concurrency>_r<repetition>` |

| file | card | model | what |
|---|---|---|---|
| `170hx_c1.jsonl` | CMP 170HX | Qwen3-Next-80B W4A16 | power sweep 250-125 W, single stream |
| `4090_c1.jsonl` | RTX 4090 | Qwen3.8-27B W4A16 | power sweep, single stream |
| `4090_c1_low.jsonl` | RTX 4090 | Qwen3.8-27B W4A16 | sweep extended to very low caps |
| `4090_c4.jsonl` | RTX 4090 | Qwen3.8-27B W4A16 | power sweep at concurrency 4 |
| `3090_c1.jsonl` | RTX 3090 | Cosmos-Reason2-8B fp8 | power sweep, single stream |

Runs are 8-45 seconds, so temperatures are not steady-state. The power figures
are averages over the full window and are the heat.

Single-stream points carry roughly 5% run-to-run spread: these were live
services and real traffic landed mid-benchmark. One 275 W point on the 4090 was
discarded as contended. Conclusions rest on the shape of the curve across eight
cap values, not on any single pair.

## Added 2026-09-13

| file | card | model | what |
|---|---|---|---|
| `27b_170hx_capsweep_c1.jsonl` | CMP 170HX | Qwen3.8-27B W4A16 + DFlash2 | power sweep 250-125 W, single stream, 27 runs |
| `27b_170hx_conc.jsonl` | CMP 170HX | Qwen3.8-27B W4A16 + DFlash2 | concurrency 1/4/8 at the 150 W cap |
| `80b_170hx_conc.jsonl` | CMP 170HX | Qwen3-Next-80B W4A16 | concurrency 1/4/8 at the 150 W cap |
| `longctx_170hx.txt` | CMP 170HX | both | 24k-context decode, 4-turn shared-prefix TTFT, prefill, load times |

The 27B sweep carries nine runs at each of 250 W and 150 W because
speculative-decoding acceptance varies with sampling: single runs at those two
caps overlap, and only the medians separate. Three runs would have supported
the wrong conclusion. Read medians, not individual rows.

| `pcie_bw.txt` | 170HX + RTX 3090 | n/a | host transfer bandwidth, link state, and the cmpunlocker retrain log |

The 170HX link state is not stock. Gen2 comes from cmpunlocker's software
retrain and x8 from a hardware capacitor mod on this board; a stock card on a
stock driver is Gen1 x4.
