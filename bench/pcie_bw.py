#!/usr/bin/env python3
"""Measure host<->device PCIe bandwidth on a specific GPU.

Two mistakes make this measurement read low, and the first version of this
script made both:

  * **Competing compute.** A matmul running on the card steals bandwidth from
    the copy engine and costs ~20% here. But a card with nothing running at all
    sits in a low power state with a downtrained link, which reads even lower.
    So: warm the card, let it settle, then copy with nothing else running.
  * **Buffers that are too small.** Per-copy launch and synchronisation
    overhead dominates at 256 MiB. Use 512 MiB or more and enough iterations
    to time at least a second of transfer.

Reports both the contended and uncontended figures, because a real serving
workload is contended and the spec sheet is not.

Usage:  pcie_bw.py <gpu-uuid-or-index> [smi-index]

Pass the UUID (nvidia-smi --query-gpu=uuid,pci.bus_id --format=csv). CUDA
orders devices by capability, not by PCI bus, so an index means something
different to torch than it does to nvidia-smi. The optional second argument is
the nvidia-smi index of the same card, used only to print link state.
"""
import os, sys, threading, time

dev = sys.argv[1] if len(sys.argv) > 1 else "0"
smi = sys.argv[2] if len(sys.argv) > 2 else None

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = dev

import torch

d = torch.device("cuda:0")
p = torch.cuda.get_device_properties(0)
print(f"card: {p.name} sm_{p.major}{p.minor} {p.total_memory >> 20} MiB")

LINK = ("nvidia-smi -i %s --query-gpu=pcie.link.gen.current,"
        "pcie.link.width.current,clocks.sm --format=csv,noheader" % smi) if smi else None

N = 512 * 2**20          # 512 MiB
ITERS = 20
host = torch.empty(N // 4, dtype=torch.float32).pin_memory()
gpu = torch.empty(N // 4, dtype=torch.float32, device=d)

def measure(tag):
    for name, fn in (("H2D", lambda: gpu.copy_(host, non_blocking=True)),
                     ("D2H", lambda: host.copy_(gpu, non_blocking=True))):
        fn()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(ITERS):
            fn()
        torch.cuda.synchronize()
        print(f"  {tag:<12} {name}: {ITERS * N / 2**30 / (time.time() - t0):.2f} GB/s")

# Warm the card into P0 with a compute kernel, and measure under that load too:
# that is what a copy competing with inference actually gets.
stop = False
def spin():
    a = torch.randn(2048, 2048, device=d)
    b = torch.randn(2048, 2048, device=d)
    while not stop:
        a = (a @ b).clamp(-1, 1)

t = threading.Thread(target=spin, daemon=True)
t.start()
time.sleep(3)
if LINK:
    os.system(LINK)
measure("contended")

stop = True
t.join(timeout=10)
time.sleep(2)          # clocks stay up briefly after the burst
if LINK:
    os.system(LINK)
measure("uncontended")
