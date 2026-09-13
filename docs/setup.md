# Setup

## Driver and CUDA

Nothing special. The card is a standard GA100 to the driver; any recent
NVIDIA driver that supports A100 supports it.

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
verify with `torch.cuda.get_device_properties(i).pci_bus_id` — it is decimal,
so bus `0x42` reads as `66`.

vLLM will warn you about this itself in a mixed box:

```
WARNING [cuda.py] Detected different devices in the system: ...
Please make sure to set `CUDA_DEVICE_ORDER=PCI_BUS_ID`
```

### 2. Give the loader time

43 GB over a 1.5 GB/s link takes 79-130 seconds before vLLM even starts
profiling. Any supervisor with a default 90-second start timeout will kill it
mid-load and look like a crash. In systemd:

```ini
TimeoutStartSec=900
TimeoutStopSec=180
```

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

## Power caps at boot

Power limits reset on reboot and on driver reload. Install the unit:

```bash
sudo cp configs/gpu-power-caps.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now gpu-power-caps
```

Edit the PCI bus ids inside it first. It pins by bus id deliberately, because
`nvidia-smi` indices follow enumeration order and can move.
