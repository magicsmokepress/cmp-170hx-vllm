# Setup

## Driver and CUDA

For basic CUDA work, nothing special: the card is a standard GA100 to the
driver, and any recent NVIDIA driver that supports an A100 supports it.

**This host does not run a stock driver.** It runs
[cmpunlocker](https://github.com/bayley/cmpunlocker), which patches the NVIDIA
kernel modules and, among other things, retrains the PCIe link from the card's
firmware-pinned Gen1 up to Gen2. That costs you Secure Boot (the patched
modules are unsigned) and a rebuild hook on every driver update, and it is the
reason the bandwidth figures in these docs are what they are. If you skip it,
expect roughly half the host bandwidth. It is not required to serve a model.

See [hardware.md](hardware.md#the-pcie-link-and-what-it-takes-to-make-it-usable)
for what is stock, what is software, and what is a hardware modification.

Tested on:

```
driver  610.43.02
CUDA    13.0
torch   2.13.0+cu130
vLLM    0.27.1
OS      Ubuntu 26.04 LTS, kernel 7.0
```

You need a second GPU or onboard video to get a console, because the 170HX has
no display output. Make sure your BIOS is set to boot from the other adapter.

## Installing vLLM

```bash
python3 -m venv ~/.venv-vllm
~/.venv-vllm/bin/pip install vllm
```

Stock upstream vLLM is fine. The 170HX is `sm_80`, which every mainstream wheel
ships because it is the A100 target. No custom build, no patches.

Verify the card is visible and reports the right capability:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID ~/.venv-vllm/bin/python - <<'PY'
import torch
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(i, p.name, f"sm_{p.major}{p.minor}", p.total_memory >> 20, "MiB")
PY
```

You are looking for `sm_80` and `65536 MiB`.

## Two pinning rules, both learned the hard way

### 1. Pin the GPU by UUID, not by index

**CUDA orders devices by capability, not by PCI bus.** In a mixed box, `cuda:0`
is whichever card CUDA thinks is fastest. On our machine `nvidia-smi` index 1
is the 170HX, but torch's index 3 was. Setting `CUDA_VISIBLE_DEVICES=1` sent a
job to the wrong card twice, and once sampled `nvidia-smi -i 3` (which orders
by bus) while loading a completely different GPU.

Use the UUID:

```bash
nvidia-smi --query-gpu=uuid,pci.bus_id,memory.total --format=csv
export CUDA_VISIBLE_DEVICES=GPU-aec84db3-b8b3-9a07-41a7-32ac25ed2b8c
```

A UUID is unambiguous and survives reboots, driver reloads, and adding cards.

For anything you write yourself, also set `CUDA_DEVICE_ORDER=PCI_BUS_ID` and
verify with `torch.cuda.get_device_properties(i).pci_bus_id`. It is decimal,
so bus `0x42` reads as `66`.

vLLM will warn you about this itself in a mixed box:

```
WARNING [cuda.py] Detected different devices in the system: ...
Please make sure to set `CUDA_DEVICE_ORDER=PCI_BUS_ID`
```

### 2. Give the loader time

41 GB takes 68 seconds to load with the weights already in page cache, and up
to 130 s cold from disk, before vLLM even starts profiling. Total time to a
listening socket was 104 s. Any supervisor with a default 90-second start timeout will kill it
mid-load and look like a crash. In systemd:

```ini
TimeoutStartSec=900
TimeoutStopSec=180
```

## Getting a model that fits

64 GB is the whole reason to use this card, so the sizing question is "what can
I fit" rather than "what can I squeeze". Weights plus KV cache plus a little
overhead have to land under 64 GB, and you want the KV cache to be generous
rather than minimal.

The model benchmarked throughout these docs:

```bash
pip install huggingface_hub[cli]
hf download RedHatAI/Qwen3-Next-80B-A3B-Instruct-quantized.w4a16 \
   --local-dir ~/models/qwen3next-w4a16
```

40.9 GB on disk, leaving ~20 GB for KV cache and overhead. It is an MoE with
~3B active parameters, which is why it prefills so fast (see
[benchmarks.md](benchmarks.md)).

Rough guidance for a 64 GB card:

| weights | fits | notes |
|---|---|---|
| up to ~45 GB | comfortably | leaves 15+ GB of KV cache |
| 45-55 GB | yes, tightly | KV cache gets thin; check the startup log |
| over ~58 GB | no | gpt-oss-120b at 63.4 GB does not fit, see benchmarks.md |

W4A16 (`compressed-tensors`) is the sweet spot. The card is `sm_80`, so it has
no FP8 tensor cores; do not reach for FP8 quantization expecting Ada-class
speedups.

## A working vLLM invocation

This is what runs Qwen3-Next-80B-A3B W4A16 on the card. The full unit is in
[`configs/vllm-80b.service`](../configs/vllm-80b.service).

```bash
CUDA_VISIBLE_DEVICES=GPU-aec84db3-... \
vllm serve /path/to/qwen3next-w4a16 \
  --served-model-name qwen3-next-80b \
  --max-model-len 65536 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.75 \
  --enable-prefix-caching --mamba-cache-mode align \
  --mamba-ssm-cache-dtype float16 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --host 0.0.0.0 --port 8000
```

What each flag is doing, and which ones are 170HX-specific:

| flag | why |
|---|---|
| `--served-model-name` | the id clients pass as `model`. Give several aliases if existing callers use different names |
| `--max-model-len 65536` | longest single request. Does **not** control memory on its own, see below |
| `--max-num-seqs 4` | concurrent slots. Requests past this queue, and client concurrency above it buys nothing (measured: C4 and C8 are identical) |
| `--gpu-memory-utilization 0.75` | **the memory knob.** 0.75 of 64 GB, not of what is free |
| `--enable-prefix-caching` | reuse the KV of a shared prefix across requests. Large win for multi-turn |
| `--mamba-cache-mode align` | required for prefix caching on a hybrid attention/SSM model. Without it, caching is silently off |
| `--mamba-ssm-cache-dtype float16` | halves the recurrent-state cache |
| `--enable-auto-tool-choice --tool-call-parser hermes` | OpenAI-style tool calling. Parser must match the model's template |
| `--host 0.0.0.0` | listens on all interfaces. Use `127.0.0.1` unless you intend LAN exposure; **vLLM has no authentication by default** |

Nothing here is 170HX-specific except by implication: the card's size is what
lets you set `--max-num-seqs` and `--gpu-memory-utilization` generously in the
first place. The card-specific parts of the setup are the UUID pinning and the
start timeout above.

### Verify it is actually serving

A vLLM process can answer `/v1/models` with HTTP 200 while being unable to
generate a single token, so check a real completion and assert on the token
count:

```bash
curl -s http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-next-80b","prompt":"Say OK.","max_tokens":5}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["usage"]["completion_tokens"])'
```

Expect a non-zero number. Then read the two KV lines from the startup log,
because a server that came up with almost no KV cache starts cleanly and then
serves one request at a time:

```bash
journalctl --user -u vllm-80b | grep -E "KV cache (memory|size)"
```

### Sizing `--gpu-memory-utilization`

This is the knob that matters, and it is not the one people reach for.

Lowering `--max-model-len` alone frees **nothing**. vLLM claims the pool from
`gpu_memory_utilization` regardless, and any surplus just becomes more KV
cache. We ran at `0.92` and vLLM reported:

```
Available KV cache memory: 16.63 GiB
GPU KV cache size: 708,425 tokens
```

for a service whose prompts have a p50 of 417 tokens. KV cache usage read
0.0%. At `0.75` it reports:

```
Available KV cache memory: 5.85 GiB
GPU KV cache size: 242,534 tokens
```

which is still 3.7x concurrency at the full 64k context and far more at real
prompt sizes, and it hands ~11 GiB of the card back for other work. With 64 GB
it is easy to leave a lot of memory doing nothing.

Watch out for the other direction too: at one point a bad configuration left
`Available KV cache memory: 0.35 GiB`, which starts cleanly and then serves
one request at a time. Always read the two KV lines in the startup log.

### Prefix caching on hybrid models is opt-in, not unavailable

Qwen3-Next is a hybrid attention/SSM model. A widely repeated claim is that
vLLM cannot prefix-cache these. It can:

```
--enable-prefix-caching --mamba-cache-mode align
```

The recurrent state resumes exactly. Measured on a 24k-token shared prefix over
four turns: TTFT 11.8 s cold, **0.49 s warm**, all answers correct. We migrated
a service to llama.cpp on the strength of that "cannot cache" claim and it cost
us a 2.2x decode regression at long context. Do not repeat that.

## Speculative decoding, and why there is no config for it here

Several benchmarks in this repo cover Qwen3.8-27B with DFlash2 speculative
decoding, which is the fastest short-prompt configuration measured on this
card. **That stack is not stock vLLM and there is deliberately no unit file for
it in `configs/`.**

It comes from [syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090),
which ships its own vLLM build with patches (split-KV attention for the
multi-query verify step, a sort-free sampler), a requantized draft model, and a
launch script. Reproducing it means using that repo, not copying a command line
out of this one. Its README covers the setup; the 170HX-relevant parts are:

```bash
CUDA_VISIBLE_DEVICES=GPU-<uuid> \
  SPEC=dflash2 DFLASH_TOKENS=15 PREFIX_CACHE=1 GPU_UTIL=0.65 \
  bash single-user/start_qwen.sh
```

`GPU_UTIL` is lowered from the script's 0.93 default only because this card
also hosts another service. On a dedicated 170HX leave it alone.

Two things worth taking away even if you never run that stack:

- **Speculative decoding changes the power picture.** It adds a compute-heavy
  verify step, so unlike a memory-bound MoE it draws 194 W and loses
  throughput under a tight power cap. See [hardware.md](hardware.md#power).
- **It makes benchmarks lie.** Decode rate becomes a function of draft
  acceptance, so a repetitive prompt overstates the server by more than 2x and
  single runs vary by 25%. See [benchmarks.md](benchmarks.md#method).

## Power caps at boot

Power limits reset on reboot and on driver reload. Install the unit:

```bash
sudo cp configs/gpu-power-caps.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now gpu-power-caps
```

Edit the PCI bus ids inside it first. It pins by bus id deliberately, because
`nvidia-smi` indices follow enumeration order and can move.
