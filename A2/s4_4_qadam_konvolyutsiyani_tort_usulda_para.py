"""A2 — 4-qadam. Konvolyutsiyani to'rt usulda parallellashtirish

3×3 konvolyutsiya (16→32 kanal, 96×96) Numba to'g'ridan-to'g'ri sikl (1 ip), prange (1 va 2 ip), NumPy im2col+GEMM va PyTorch conv2d (oneDNN, 1 va 2 ip, batch 8) bilan bajariladi; barcha natijalar etalon bilan solishtiriladi. Kutilgan natija: prange 2 ipda ~2×, im2col ~16×, torch 2 ip ~20× tezlanish; xato < 1e-5. Bu cuDNN'ning implicit GEMM g'oyasini CPU'da ko'rsatadi.

Kutilgan natija: 81,4 MFLOP; jadval: Numba 1 ip ~2 GFLOP/s, prange 2 ip ~4, im2col ~33, torch 2 ip ~46 GFLOP/s; maks xato ~5e-6.
"""
# 4-qadam. Konvolyutsiya (3x3) ni to'rt usulda: Numba to'g'ridan-to'g'ri (1 ip), Numba prange (2 ip),
# NumPy im2col + GEMM, PyTorch conv2d (oneDNN, 2 ip). Hammasi bir xil natija berishi tekshiriladi.
import time
import numpy as np
import numba
from numba import njit, prange
import torch
import torch.nn.functional as F

torch.set_num_threads(2)
torch.backends.nnpack.set_flags(False)
numba.set_num_threads(2)
np.random.seed(0)

CIN, COUT, H, W, K = 16, 32, 96, 96, 3
HO, WO = H - K + 1, W - K + 1
FLOP = 2 * K * K * CIN * COUT * HO * WO


@njit
def conv_seq(x, w, y):
    cin, h, wd = x.shape
    cout, ho, wo = y.shape
    for co in range(cout):
        for i in range(ho):
            for j in range(wo):
                s = 0.0
                for ci in range(cin):
                    for di in range(3):
                        for dj in range(3):
                            s += x[ci, i + di, j + dj] * w[co, ci, di, dj]
                y[co, i, j] = s


@njit(parallel=True)
def conv_par(x, w, y):
    cin, h, wd = x.shape
    cout, ho, wo = y.shape
    for co in prange(cout):                # chiqish kanallari iplar orasida taqsimlanadi
        for i in range(ho):
            for j in range(wo):
                s = 0.0
                for ci in range(cin):
                    for di in range(3):
                        for dj in range(3):
                            s += x[ci, i + di, j + dj] * w[co, ci, di, dj]
                y[co, i, j] = s


def conv_im2col(x, w):
    """Konvolyutsiyani matritsa ko'paytirishga keltirish (cuDNN'ning implicit GEMM g'oyasi)."""
    cols = np.empty((CIN * K * K, HO * WO), dtype=np.float32)
    r = 0
    for ci in range(CIN):
        for di in range(K):
            for dj in range(K):
                cols[r] = x[ci, di:di + HO, dj:dj + WO].ravel()
                r += 1
    return (w.reshape(COUT, -1) @ cols).reshape(COUT, HO, WO)


def olchov(f, takror=5):
    f()
    v = []
    for _ in range(takror):
        t0 = time.perf_counter(); f(); v.append(time.perf_counter() - t0)
    return sorted(v)[len(v) // 2]


x = np.random.rand(CIN, H, W).astype(np.float32)
w = (np.random.rand(COUT, CIN, K, K).astype(np.float32) - 0.5)
y1 = np.zeros((COUT, HO, WO), np.float32)
y2 = np.zeros((COUT, HO, WO), np.float32)
xt, wt = torch.from_numpy(x)[None], torch.from_numpy(w)

print("=== Konvolyutsiya ===")
print(f"Kirish {CIN}x{H}x{W}, yadro {COUT}x{CIN}x{K}x{K}, chiqish {COUT}x{HO}x{WO}: {FLOP / 1e6:.1f} MFLOP")
t0 = time.perf_counter(); conv_seq(x, w, y1); conv_par(x, w, y2); print(f"Numba kompilyatsiya: {time.perf_counter() - t0:.2f} s")

etalon = F.conv2d(xt, wt)[0].numpy()
natija = []
natija.append(("Numba to'g'ridan-to'g'ri, 1 ip", olchov(lambda: conv_seq(x, w, y1)), y1))
numba.set_num_threads(1)
natija.append(("Numba prange, 1 ip", olchov(lambda: conv_par(x, w, y2)), y2))
numba.set_num_threads(2)
natija.append(("Numba prange, 2 ip", olchov(lambda: conv_par(x, w, y2)), y2))
natija.append(("NumPy im2col + GEMM", olchov(lambda: conv_im2col(x, w)), conv_im2col(x, w)))
with torch.no_grad():
    torch.set_num_threads(1)
    natija.append(("torch conv2d, 1 ip", olchov(lambda: F.conv2d(xt, wt)), etalon))
    torch.set_num_threads(2)
    natija.append(("torch conv2d, 2 ip", olchov(lambda: F.conv2d(xt, wt)), etalon))
    xb = xt.repeat(8, 1, 1, 1)
    t_b = olchov(lambda: F.conv2d(xb, wt))
    natija.append(("torch conv2d, 2 ip, batch 8", t_b / 8, etalon))

print(f"\n{'usul':30s} {'ms':>8s} {'GFLOP/s':>8s} {'tezlanish':>9s} {'maks xato':>10s}")
t_bazaviy = natija[0][1]
for nom, t, yy in natija:
    print(f"{nom:30s} {t * 1e3:8.2f} {FLOP / t / 1e9:8.2f} {t_bazaviy / t:9.1f} {np.abs(yy - etalon).max():10.1e}")
print("\nXulosa: to'g'ridan-to'g'ri sikl -> prange (2x) -> GEMM'ga keltirish (BLAS) -> oneDNN (vektorlash + bloklash).")
print("GPU bo'lsa: F.conv2d cuDNN'ga o'tadi (implicit GEMM/Winograd), FP16'da Tensor Core; o'lchash uchun torch.cuda.synchronize() kerak.")
