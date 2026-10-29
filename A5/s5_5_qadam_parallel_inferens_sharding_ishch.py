"""A5 — 5-qadam. Parallel inferens: sharding, ishchilar puli, natijalarni yig'ish

Digits ustida o'qitilgan MLP og'irliklari NumPy'ga eksport qilinadi, ~360 ming so'rov 2048 talik bo'laklarga bo'linib multiprocessing Pool'ga tarqatiladi (scatter), natijalar tartib raqami bo'yicha yig'iladi (gather). 1 va 2 ishchi uchun o'tkazuvchanlik, S(N), E(N) va bo'lak kechikishining p50/p99 qiymatlari jadvalda chiqadi.

Kutilgan natija: 1 ishchi ~0.5-0.6 s, 2 ishchi ~0.3 s; S(2)≈1.7-1.9, E≈85-95%; aniqlik ikkala holatda bir xil (0.993).
"""
# 5-qadam. Parallel inferens: ma'lumotni bo'lib yuborish (sharding), ishchilar puli, natijalarni yig'ish
import os
os.environ["OMP_NUM_THREADS"] = "1"        # har ishchi 1 yadro
import time, numpy as np, torch, torch.nn as nn
import multiprocessing as mp
from sklearn.datasets import load_digits

def train_model():
    """Berilgan: digits MLP o'qitiladi, og'irliklar NumPy (W, b) juftliklari sifatida qaytadi (o'zgartirmang)."""
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
    global W_GLOBAL; W_GLOBAL = W

def predict_shard(args):
    # TODO 1: (idx, X) bo'lagi uchun NumPy'da ReLU-MLP inferens; (idx, bashorat, vaqt) qaytaring
    raise NotImplementedError("TODO 1: predict_shard")

if __name__ == "__main__":
    W, X, y, acc = train_model()
    Xbig = np.tile(X, (200, 1)).astype(np.float32); ybig = np.tile(y, 200)
    SHARD = 2048
    # TODO 2: Xbig ni SHARD hajmli (tartib_raqami, bo'lak) juftliklariga bo'ling
    shards = []
    print("=== Parallel inferens o'tkazuvchanligi ===")
    # TODO 3: nw = 1, 2 uchun fork Pool(initializer=init_worker) bilan isitib, keyin pool.map bilan vaqtni o'lchang
    # TODO 4: natijalarni tartib raqami bo'yicha yig'ing; so'rov/s, S(N), E(N), bo'lak p50/p99 (ms) va aniqlikni chiqaring
    raise NotImplementedError("TODO 3-4: parallel inferens")
