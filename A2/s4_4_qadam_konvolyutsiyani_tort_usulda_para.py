"""A2 — 4-qadam. Konvolyutsiyani to'rt usulda parallellashtirish

3×3 konvolyutsiya (16→32 kanal, 96×96) Numba to'g'ridan-to'g'ri sikl (1 ip), prange (1 va 2 ip), NumPy im2col+GEMM va PyTorch conv2d (oneDNN, 1 va 2 ip, batch 8) bilan bajariladi; barcha natijalar etalon bilan solishtiriladi. Kutilgan natija: prange 2 ipda ~2×, im2col ~16×, torch 2 ip ~20× tezlanish; xato < 1e-5. Bu cuDNN'ning implicit GEMM g'oyasini CPU'da ko'rsatadi.

Kutilgan natija: 81,4 MFLOP; jadval: Numba 1 ip ~2 GFLOP/s, prange 2 ip ~4, im2col ~33, torch 2 ip ~46 GFLOP/s; maks xato ~5e-6.
"""
# 4-qadam. Konvolyutsiya (3x3) ni to'rt usulda: Numba (1 ip), Numba prange (2 ip),
# NumPy im2col + GEMM, PyTorch conv2d (oneDNN). Hammasi bir xil natija berishi tekshiriladi.
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
# TODO 1: konvolyutsiyaning FLOP sonini hisoblang (~81,4 MFLOP chiqishi kerak)
FLOP = None


# TODO 2: conv_seq(x, w, y) — 6 ta ichma-ich sikl bilan to'g'ridan-to'g'ri konvolyutsiya (@njit)
# TODO 3: conv_par(x, w, y) — chiqish kanallari bo'ylab prange (@njit(parallel=True))
def conv_seq(x, w, y):
    raise NotImplementedError("TODO 2: conv_seq")


def conv_par(x, w, y):
    raise NotImplementedError("TODO 3: conv_par")


def conv_im2col(x, w):
    """Konvolyutsiyani matritsa ko'paytirishga keltirish (cuDNN'ning implicit GEMM g'oyasi)."""
    # TODO 4: (CIN*K*K, HO*WO) cols matritsasini to'ldiring va w.reshape(COUT, -1) @ cols ni qaytaring
    raise NotImplementedError("TODO 4: conv_im2col")


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
print(f"Kirish {CIN}x{H}x{W}, chiqish {COUT}x{HO}x{WO}: {FLOP / 1e6:.1f} MFLOP")
etalon = F.conv2d(xt, wt)[0].numpy()
# TODO 5: barcha usullarni (Numba 1/2 ip, im2col, torch conv2d 1/2 ip va batch 8) o'lchab jadval chiqaring:
#         ms, GFLOP/s, birinchi usulga nisbatan tezlanish, etalondan maks xato
