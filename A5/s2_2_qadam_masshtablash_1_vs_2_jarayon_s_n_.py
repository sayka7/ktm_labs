"""A5 — 2-qadam. Masshtablash: 1 vs 2 jarayon, S(N) va E(N)

Bir xil MLP uchun to'rt konfiguratsiya o'lchanadi: 1 jarayon (1 yadro) — baza T(1); 2 jarayonli DDP kuchli rejimda (global paket o'zgarmaydi, har rankka yarmi); kuchsiz rejimda (har rankda to'liq paket, global 2 baravar); va taqqoslash uchun 1 jarayon 2 yadro bilan (intra-op parallellik). Tezlashish S(N) va samaradorlik E(N) jadvalda chiqadi. Talaba `HID`, `GB`, `STEPS` ni o'zgartirib, model kattalashganda samaradorlik qanday o'zgarishini ko'radi.

Kutilgan natija: Baza ~4-5 s; DDP kuchli rejimda S(2)≈1.6-1.8 (E≈80-90%), kuchsiz rejimda E≈90%; intra-op 2 yadro ham ~1.9x. Qadam 15-25 s davom etadi.
"""
# 2-qadam. Masshtablash: 1 jarayon vs 2 jarayon (kuchli va kuchsiz masshtablash), S(N) va E(N)
import os, time, random
import torch, torch.nn as nn, torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

N_FEAT, HID, N_CLS = 64, 1024, 8
STEPS = 60

def make_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(N_FEAT, HID), nn.ReLU(), nn.Linear(HID, HID), nn.ReLU(), nn.Linear(HID, N_CLS))

def train_steps(model, bs, steps):
    """bs hajmli paketda `steps` ta qadam; vaqtni qaytaradi (2 ta isitish qadami hisobga olinmaydi)."""
    opt = torch.optim.SGD(model.parameters(), lr=0.01); lossf = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(1)
    X = torch.randn(bs, N_FEAT, generator=g); y = torch.randint(0, N_CLS, (bs,), generator=g)
    for i in range(steps + 2):
        if i == 2: t0 = time.time()
        opt.zero_grad(set_to_none=True); lossf(model(X), y).backward(); opt.step()
    return time.time() - t0

def run_ddp(world, bs_per_rank):
    # TODO 1: world ta jarayonda DDP modelni bs_per_rank paket bilan train_steps orqali o'qiting
    #         va rank 0 vaqtini Queue orqali asosiy jarayonga qaytaring
    raise NotImplementedError("TODO 1: run_ddp")

if __name__ == "__main__":
    torch.set_num_threads(1)
    GB = 256                                               # global paket
    # TODO 2: to'rt o'lchov: T(1) (1 jarayon, 1 yadro), DDP kuchli (har rankka GB//2), DDP kuchsiz (har rankda GB),
    #         va 1 jarayon 2 yadro (torch.set_num_threads(2))
    raise NotImplementedError("TODO 2: to'rt o'lchov")
    rows = []
    print("=== Masshtablash jadvali (gloo, CPU) ===")
    # TODO 3: har konfiguratsiya uchun namuna/s, S(N) (o'tkazuvchanlik nisbati) va E(N) = S/N ni chop eting
