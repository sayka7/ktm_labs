"""A2 — 2-qadam. Numba @njit va prange — OpenMP'ning Python muqobili

@njit matmul (ikj) kompilyatsiya qilinadi, parallel=True + prange varianti 1 va 2 ipda (numba.set_num_threads) o'lchanadi, NumPy BLAS bilan solishtiriladi; alohida reduktsiya (sum x²) misoli 2 ipda xotira o'tkazuvchanligi cheklovini ko'rsatadi. Kutilgan natija: JIT kompilyatsiya ~0,3–1 s, prange 2 ipda ~1,6–1,9× tezlanish, BLAS yana 1,5× tezroq. Bu real labdagi `#pragma omp parallel for` va `reduction(+:s)` ning to'g'ridan-to'g'ri muqobili.

Kutilgan natija: Kompilyatsiya vaqtlari, threading layer nomi, 4 qatorli jadval (seq ~16 GFLOP/s, prange 2 ip ~27, BLAS ~39), reduktsiya 11 → 22 GB/s.
"""
# 2-qadam. Numba @njit va prange — OpenMP'ning Python muqobili
# @njit kodni mashina kodiga kompilyatsiya qiladi (JIT), parallel=True + prange siklni iplarga bo'ladi.
import time
import numpy as np
import numba
from numba import njit, prange

numba.set_num_threads(2)


# TODO 1: ketma-ket ikj matmul'ni @njit bilan yozing (C ga qo'shib boring)
def matmul_seq(A, B, C):
    raise NotImplementedError("TODO 1: matmul_seq")


# TODO 2: tashqi siklni prange bilan parallellashtirilgan varianti (@njit(parallel=True))
def matmul_par(A, B, C):
    raise NotImplementedError("TODO 2: matmul_par")


# TODO 3: sum(x*x) reduktsiyasini prange bilan yozing
def yigindi_par(x):
    raise NotImplementedError("TODO 3: yigindi_par")


def olchov(f, *args, takror=5):
    v = []
    for _ in range(takror):
        args[-1][:] = 0.0
        t0 = time.perf_counter()
        f(*args)
        v.append(time.perf_counter() - t0)
    return sorted(v)[len(v) // 2]


n = 256
A = np.random.rand(n, n).astype(np.float32)
B = np.random.rand(n, n).astype(np.float32)
C = np.zeros((n, n), dtype=np.float32)
flop = 2 * n ** 3

print("=== Numba njit va prange ===")
# TODO 4: birinchi chaqiruv vaqtini (JIT kompilyatsiya) seq va par uchun o'lchang, threading_layer() ni chiqaring
matmul_seq(A, B, C)
# TODO 5: A @ B bilan to'g'rilikni tekshiring, so'ng seq, prange (1 va 2 ip) va NumPy BLAS jadvalini chiqaring

# TODO 6: 4M elementli x uchun reduktsiyani 1 va 2 ipda o'lchab GB/s ni chiqaring
print("\n--- Reduktsiya: sum(x*x), 4M element ---")
