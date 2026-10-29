"""A5 — 1-qadam. DDP: gloo backend bilan 2 jarayonli taqsimlangan o'qitish

torch.multiprocessing orqali 2 ta jarayon (rank 0 va 1) ishga tushiriladi, har biri jarayonlar guruhiga qo'shiladi, DistributedSampler ma'lumotni kesishmaydigan ikki qismga bo'ladi, DDP o'rami gradientlarni har qadamda all-reduce qiladi. Oxirida parametrlar barcha ranklarda bir xil ekani (max farq 0) va jami ko'rilgan namunalar soni tekshiriladi. Talaba `EPOCHS`, `BS`, LR masshtablash qoidasini (`0.05 * world`) va `set_epoch` chaqiruvini o'zgartirib ko'rishi mumkin.

Kutilgan natija: Ikkala rank o'z 4096 namunasini ko'radi, jami 8192; aniqlik ~0.95; ranklar orasidagi parametr farqi 0.00e+00 (DDP sinxron); o'tkazuvchanlik bir necha o'n ming namuna/s.
"""
# 1-qadam. DDP: 2 jarayonli taqsimlangan o'qitish (gloo backend, CPU)
# GPU va NCCL yo'q — shuning uchun CPU uchun mo'ljallangan gloo backend ishlatiladi.
import os, time, random
import torch, torch.nn as nn, torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

N_SAMPLES, N_FEAT, N_CLS = 4096, 64, 8
EPOCHS, BS = 2, 64          # BS — har bir jarayon uchun paket

def make_data():
    """Sintetik tasniflash ma'lumoti — barcha ranklarda bir xil urug' bilan (o'zgartirmang)."""
    g = torch.Generator().manual_seed(0)
    centers = torch.randn(N_CLS, N_FEAT, generator=g) * 0.6
    y = torch.randint(0, N_CLS, (N_SAMPLES,), generator=g)
    X = centers[y] + torch.randn(N_SAMPLES, N_FEAT, generator=g) * 1.5
    return TensorDataset(X, y)

def make_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(N_FEAT, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, N_CLS))

def worker(rank, world, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    torch.set_num_threads(1)
    # TODO 1: jarayonlar guruhiga qo'shiling (gloo, rank, world_size)
    # TODO 2: DistributedSampler + DataLoader, modelni DDP ga o'rang, SGD (lr chiziqli masshtablash qoidasi bilan)
    # TODO 3: EPOCHS davomida o'qiting (har epoxa sampler.set_epoch), ko'rilgan namunalarni sanang
    # TODO 4: all_reduce bilan jami namunalar va yo'qotishni yig'ing; parametrlarning MAX va MIN all_reduce farqini toping
    # TODO 5: rank 0 da "=== DDP natijasi (gloo, 2 jarayon) ===" va parametr farqi (BIR XIL / FARQ BOR) ni chiqaring; guruhni yoping
    raise NotImplementedError("TODO 1-5: DDP worker")

if __name__ == "__main__":
    port = random.randint(20000, 60000)
    world = 2
    print(f"Ishga tushirish: world_size={world}, backend=gloo, MASTER_PORT={port}")
    mp.start_processes(worker, args=(world, port), nprocs=world, start_method="fork", join=True)
