"""A5 — 2-qadam. Masshtablash: 1 vs 2 jarayon, S(N) va E(N)

Bir xil MLP uchun to'rt konfiguratsiya o'lchanadi: 1 jarayon (1 yadro) — baza T(1); 2 jarayonli DDP kuchli rejimda (global paket o'zgarmaydi, har rankka yarmi); kuchsiz rejimda (har rankda to'liq paket, global 2 baravar); va taqqoslash uchun 1 jarayon 2 yadro bilan (intra-op parallellik). Tezlashish S(N) va samaradorlik E(N) jadvalda chiqadi. Talaba `HID`, `GB`, `STEPS` ni o'zgartirib, model kattalashganda samaradorlik qanday o'zgarishini ko'radi.

Kutilgan natija: Baza ~4-5 s; DDP kuchli rejimda S(2)≈1.6-1.8 (E≈80-90%), kuchsiz rejimda E≈90%; intra-op 2 yadro ham ~1.9x. Qadam 15-25 s davom etadi.
"""
# 2-qadam. Masshtablash: 1 jarayon vs 2 jarayon (kuchli va kuchsiz masshtablash), S(N) va E(N)
# CPU sandbox: 2 yadro. Har jarayon 1 yadro oladi — bu "2 ta GPU" ning CPU muqobili.
import os, time, random
import torch, torch.nn as nn, torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

N_FEAT, HID, N_CLS = 64, 1024, 8
STEPS = 60                      # o'lchanadigan optimizatsiya qadamlari soni

def make_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(N_FEAT, HID), nn.ReLU(), nn.Linear(HID, HID), nn.ReLU(), nn.Linear(HID, N_CLS))

def train_steps(model, bs, steps):
    """bs hajmli paketda `steps` ta qadam; vaqtni qaytaradi (2 ta isitish qadami hisobga olinmaydi)."""
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    lossf = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(1)
    X = torch.randn(bs, N_FEAT, generator=g); y = torch.randint(0, N_CLS, (bs,), generator=g)
    for i in range(steps + 2):
        if i == 2: t0 = time.time()
        opt.zero_grad(set_to_none=True)
        lossf(model(X), y).backward(); opt.step()
    return time.time() - t0

def worker(rank, world, port, bs_per_rank, q):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), GLOO_SOCKET_IFNAME="lo")
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    model = DDP(make_model())
    dt = train_steps(model, bs_per_rank, STEPS)
    if rank == 0: q.put(dt)
    dist.destroy_process_group()

def run_ddp(world, bs_per_rank):
    q = mp.get_context("fork").Queue()
    port = random.randint(20000, 60000)
    mp.start_processes(worker, args=(world, port, bs_per_rank, q), nprocs=world, start_method="fork", join=True)
    return q.get()

if __name__ == "__main__":
    torch.set_num_threads(1)
    GB = 256                                               # global paket (kuchli masshtablash uchun)
    t1 = train_steps(make_model(), GB, STEPS)              # T(1): bitta jarayon, 1 yadro
    t2_strong = run_ddp(2, GB // 2)                        # kuchli: global paket o'zgarmaydi -> har rankka 128
    t2_weak = run_ddp(2, GB)                               # kuchsiz: har rankda 256 -> global 512
    torch.set_num_threads(2)
    t1_2thr = train_steps(make_model(), GB, STEPS)         # taqqoslash: 1 jarayon, lekin 2 yadro (intra-op)

    rows = [("1 jarayon (1 yadro) — baza", GB, t1, 1),
            ("2 jarayon DDP, kuchli (global 256)", GB, t2_strong, 2),
            ("2 jarayon DDP, kuchsiz (global 512)", 2 * GB, t2_weak, 2),
            ("1 jarayon, 2 yadro (intra-op)", GB, t1_2thr, 2)]
    print("=== Masshtablash jadvali (gloo, CPU) ===")
    print(f"{'Konfiguratsiya':38s} {'global paket':>12s} {'vaqt,s':>8s} {'namuna/s':>10s} {'S(N)':>6s} {'E(N)':>7s}")
    for name, gb, t, n in rows:
        thr = gb * STEPS / t
        # kuchsiz rejimda tezlashish o'tkazuvchanlik nisbati bilan hisoblanadi
        s = (thr / (GB * STEPS / t1))
        print(f"{name:38s} {gb:12d} {t:8.2f} {thr:10.0f} {s:6.2f} {100 * s / n:6.0f}%")
    print("\nIzoh: S(N)=T(1)/T(N) (kuchsiz rejimda namuna/s nisbati), E(N)=S(N)/N.")
    print("Kuchli masshtablashda har rankka kamroq ish tegadi — samaradorlik odatda pastroq.")
    print("Haqiqiy GPU klasterida bir tugunda E(2) ~ 90-95%, tugunlar orasida 70-90% kutiladi.")
