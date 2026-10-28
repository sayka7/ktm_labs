"""A5 — 1-qadam. DDP: gloo backend bilan 2 jarayonli taqsimlangan o'qitish

torch.multiprocessing orqali 2 ta jarayon (rank 0 va 1) ishga tushiriladi, har biri jarayonlar guruhiga qo'shiladi, DistributedSampler ma'lumotni kesishmaydigan ikki qismga bo'ladi, DDP o'rami gradientlarni har qadamda all-reduce qiladi. Oxirida parametrlar barcha ranklarda bir xil ekani (max farq 0) va jami ko'rilgan namunalar soni tekshiriladi. Talaba `EPOCHS`, `BS`, LR masshtablash qoidasini (`0.05 * world`) va `set_epoch` chaqiruvini o'zgartirib ko'rishi mumkin.

Kutilgan natija: Ikkala rank o'z 4096 namunasini ko'radi, jami 8192; aniqlik ~0.95; ranklar orasidagi parametr farqi 0.00e+00 (DDP sinxron); o'tkazuvchanlik bir necha o'n ming namuna/s.
"""
# 1-qadam. DDP: 2 jarayonli taqsimlangan o'qitish (gloo backend, CPU)
# GPU va NCCL yo'q — shuning uchun CPU uchun mo'ljallangan gloo backend ishlatiladi.
# Mantiq (process group, DistributedSampler, DDP o'rami, all-reduce) haqiqiy GPU klasteridagi bilan bir xil.
import os, time, random
import torch, torch.nn as nn, torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

N_SAMPLES, N_FEAT, N_CLS = 4096, 64, 8
EPOCHS, BS = 2, 64          # BS — har bir jarayon uchun paket (kuchsiz masshtablash rejimi)

def make_data():
    """Sintetik tasniflash ma'lumoti — barcha ranklarda bir xil urug' bilan yaratiladi."""
    g = torch.Generator().manual_seed(0)
    centers = torch.randn(N_CLS, N_FEAT, generator=g) * 0.6
    y = torch.randint(0, N_CLS, (N_SAMPLES,), generator=g)
    X = centers[y] + torch.randn(N_SAMPLES, N_FEAT, generator=g) * 1.5
    return TensorDataset(X, y)

def make_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(N_FEAT, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, N_CLS))

def worker(rank, world, port):
    """Har bir jarayon shu funksiyani bajaradi — torchrun'dagi bitta 'rank' ga teng."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"      # tarmoq yo'q — faqat loopback interfeysi
    torch.set_num_threads(1)                      # har jarayonga 1 yadro (2 yadro = 2 jarayon)
    dist.init_process_group("gloo", rank=rank, world_size=world)

    ds = make_data()
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, seed=0)
    dl = DataLoader(ds, batch_size=BS, sampler=sampler, drop_last=True)
    model = DDP(make_model())                     # 0-rank parametrlari hammaga broadcast qilinadi
    opt = torch.optim.SGD(model.parameters(), lr=0.05 * world, momentum=0.9)   # chiziqli LR qoidasi
    lossf = nn.CrossEntropyLoss()

    t0 = time.time(); seen = 0
    for ep in range(EPOCHS):
        sampler.set_epoch(ep)                     # har epoxada boshqa aralashtirish
        for X, y in dl:
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(X), y)
            loss.backward()                       # gradientlar chelaklarda avtomatik all-reduce qilinadi
            opt.step(); seen += X.size(0)
    dt = time.time() - t0

    # Global ko'rsatkichlarni yig'amiz: namunalar soni va oxirgi yo'qotish
    stats = torch.tensor([seen, loss.item()], dtype=torch.float64)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    # Parametrlar barcha ranklarda bir xilmi? (DDP to'g'ri ishlaganining isboti)
    flat = torch.cat([p.detach().flatten() for p in model.parameters()])
    mx = flat.clone(); mn = flat.clone()
    dist.all_reduce(mx, op=dist.ReduceOp.MAX); dist.all_reduce(mn, op=dist.ReduceOp.MIN)
    diff = (mx - mn).abs().max().item()
    # Aniqlik — to'liq to'plamda (har rank hisoblaydi, 0-rank chiqaradi)
    with torch.no_grad():
        acc = (model(ds.tensors[0]).argmax(1) == ds.tensors[1]).float().mean().item()
    print(f"rank={rank} world={world}  namunalar(shu rank)={seen}  vaqt={dt:.2f}s", flush=True)
    if rank == 0:
        print("=== DDP natijasi (gloo, 2 jarayon) ===")
        print(f"jami ko'rilgan namunalar : {int(stats[0].item())}  ({EPOCHS} epoxa x {N_SAMPLES})")
        print(f"o'rtacha oxirgi yo'qotish: {stats[1].item() / world:.4f}")
        print(f"aniqlik (to'liq to'plam) : {acc:.3f}")
        print(f"ranklar orasidagi parametr farqi: {diff:.2e}  -> {'BIR XIL (DDP sinxron)' if diff < 1e-6 else 'FARQ BOR!'}")
        print(f"o'tkazuvchanlik: {stats[0].item() / dt:.0f} namuna/s")
    dist.destroy_process_group()

if __name__ == "__main__":
    port = random.randint(20000, 60000)           # port band bo'lmasligi uchun tasodifiy
    world = 2
    print(f"Ishga tushirish: world_size={world}, backend=gloo, MASTER_PORT={port}")
    mp.start_processes(worker, args=(world, port), nprocs=world, start_method="fork", join=True)
    print("Izoh: GPU klasterida backend='nccl', ishga tushirish torchrun bilan bo'ladi; kod mantig'i o'zgarmaydi.")
