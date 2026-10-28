"""A5 — 5-qadam. Parallel inferens: sharding, ishchilar puli, natijalarni yig'ish

Digits ustida o'qitilgan MLP og'irliklari NumPy'ga eksport qilinadi, ~360 ming so'rov 2048 talik bo'laklarga bo'linib multiprocessing Pool'ga tarqatiladi (scatter), natijalar tartib raqami bo'yicha yig'iladi (gather). 1 va 2 ishchi uchun o'tkazuvchanlik, S(N), E(N) va bo'lak kechikishining p50/p99 qiymatlari jadvalda chiqadi.

Kutilgan natija: 1 ishchi ~0.5-0.6 s, 2 ishchi ~0.3 s; S(2)≈1.7-1.9, E≈85-95%; aniqlik ikkala holatda bir xil (0.993).
"""
# 5-qadam. Parallel inferens: ma'lumotni bo'lib yuborish (sharding), ishchilar puli, natijalarni yig'ish
# Model — digits ustida o'qitilgan kichik MLP; inferens NumPy'da (har ishchi o'z nusxasi bilan).
import os
os.environ["OMP_NUM_THREADS"] = "1"        # har ishchi 1 yadro — BLAS ichki parallelligi o'chiriladi
import time, numpy as np, torch, torch.nn as nn
import multiprocessing as mp
from sklearn.datasets import load_digits

def train_model():
    torch.manual_seed(0); torch.set_num_threads(1)
    d = load_digits(); X = torch.tensor(d.data / 16.0, dtype=torch.float32); y = torch.tensor(d.target)
    m = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(150):
        opt.zero_grad(); nn.functional.cross_entropy(m(X), y).backward(); opt.step()
    acc = (m(X).argmax(1) == y).float().mean().item()
    W = [(l.weight.detach().numpy().T.copy(), l.bias.detach().numpy().copy()) for l in m if isinstance(l, nn.Linear)]
    return W, d.data / 16.0, d.target, acc

W_GLOBAL = None
def init_worker(W):
    global W_GLOBAL; W_GLOBAL = W          # fork: model nusxasi har ishchiga ko'chiriladi

def predict_shard(args):
    idx, X = args                          # idx — bo'lakning tartib raqami (natijani to'g'ri tartibda yig'ish uchun)
    (W1, b1), (W2, b2) = W_GLOBAL
    t0 = time.perf_counter()
    h = np.maximum(X @ W1 + b1, 0); out = (h @ W2 + b2).argmax(1)
    return idx, out, time.perf_counter() - t0

if __name__ == "__main__":
    W, X, y, acc = train_model()
    # inferens yukini kattalashtiramiz: 1797 ta rasmni 200 marta takrorlab ~360k so'rov
    Xbig = np.tile(X, (200, 1)).astype(np.float32); ybig = np.tile(y, 200)
    SHARD = 2048
    shards = [(i, Xbig[s:s + SHARD]) for i, s in enumerate(range(0, len(Xbig), SHARD))]
    print(f"model aniqligi={acc:.3f}; so'rovlar={len(Xbig)}; bo'laklar={len(shards)} x {SHARD}")
    print("=== Parallel inferens o'tkazuvchanligi ===")
    print(f"{'ishchilar':>9s} {'vaqt,s':>8s} {'so\'rov/s':>10s} {'S(N)':>6s} {'E(N)':>6s} {'p50 bo\'lak,ms':>13s} {'p99 bo\'lak,ms':>13s} {'aniqlik':>8s}")
    base = None
    for nw in (1, 2):
        ctx = mp.get_context("fork")
        with ctx.Pool(nw, initializer=init_worker, initargs=(W,)) as pool:
            pool.map(predict_shard, shards[:2])          # isitish
            t0 = time.perf_counter()
            res = pool.map(predict_shard, shards)         # scatter -> hisob -> gather
            dt = time.perf_counter() - t0
        res.sort(key=lambda r: r[0])                      # all-gather: tartib bo'yicha yig'ish
        pred = np.concatenate([r[1] for r in res]); lat = np.array([r[2] for r in res]) * 1e3
        thr = len(Xbig) / dt
        base = base or thr
        s = thr / base
        print(f"{nw:9d} {dt:8.3f} {thr:10.0f} {s:6.2f} {100*s/nw:5.0f}% {np.percentile(lat,50):13.2f} {np.percentile(lat,99):13.2f} {(pred==ybig).mean():8.3f}")
    print("\nXulosa: inferensda qadam sayin sinxronizatsiya yo'q ('noqulay parallel'), shuning uchun")
    print("masshtablash o'qitishga qaraganda yaxshiroq; cheklov — bo'laklarni tarqatish/yig'ish (IPC) xarajati.")
    print("Real xizmatda esa p50/p99 kechikish va dinamik paketlash (dynamic batching) ham o'lchanadi.")
