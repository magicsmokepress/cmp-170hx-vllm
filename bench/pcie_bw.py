#!/usr/bin/env python3
"""Measure host<->device PCIe bandwidth on a specific GPU, under compute load.

A pure DMA copy does not wake the GPU: the shaders stay idle, the card stays in
a low power state, and the link stays downtrained. Benchmarking that way
measures the idle link and looks like proof the card cannot train up. So this
holds a matmul kernel running in a background thread for the duration.

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

stop = False
def spin():
    a = torch.randn(2048, 2048, device=d)
    b = torch.randn(2048, 2048, device=d)
    while not stop:
        a = (a @ b).clamp(-1, 1)

t = threading.Thread(target=spin, daemon=True)
t.start()
time.sleep(3)  # let the card reach P0 and the link train

link = ("nvidia-smi -i %s --query-gpu=pcie.link.gen.current,"
        "pcie.link.width.current,clocks.sm,power.draw "
        "--format=csv,noheader" % smi) if smi else None
if link:
    os.system(link)

N = 256 * 2**20  # 256 MiB
host = torch.empty(N // 4, dtype=torch.float32).pin_memory()
gpu = torch.empty(N // 4, dtype=torch.float32, device=d)

for name, fn in (("H2D", lambda: gpu.copy_(host, non_blocking=True)),
                 ("D2H", lambda: host.copy_(gpu, non_blocking=True))):
    fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    print(f"{name}: {10 * N / 2**30 / (time.time() - t0):.2f} GB/s")

if link:
    os.system(link)
stop = True
t.join(timeout=5)
