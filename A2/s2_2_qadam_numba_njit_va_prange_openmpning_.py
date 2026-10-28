"""A2 — 2-qadam. Numba @njit va prange — OpenMP'ning Python muqobili

@njit matmul (ikj) kompilyatsiya qilinadi, parallel=True + prange varianti 1 va 2 ipda (numba.set_num_threads) o'lchanadi, NumPy BLAS bilan solishtiriladi; alohida reduktsiya (sum x²) misoli 2 ipda xotira o'tkazuvchanligi cheklovini ko'rsatadi. Kutilgan natija: JIT kompilyatsiya ~0,3–1 s, prange 2 ipda ~1,6–1,9× tezlanish, BLAS yana 1,5× tezroq. Bu real labdagi `#pragma omp parallel for` va `reduction(+:s)` ning to'g'ridan-to'g'ri muqobili.

Kutilgan natija: Kompilyatsiya vaqtlari, threading layer nomi, 4 qatorli jadval (seq ~16 GFLOP/s, prange 2 ip ~27, BLAS ~39), reduktsiya 11 → 22 GB/s.
"""
# 2-qadam. Numba @njit va prange — OpenMP'ning Python muqobili
# @njit kodni mashina kodiga kompilyatsiya qiladi (JIT), parallel=True + prange esa siklni
# iplar orasida taqsimlaydi (OpenMP `#pragma omp parallel for` kabi). Sandbox'da 2 yadro bor.
import time
import numpy as np
import numba
from numba import njit, prange

numba.set_num_threads(2)


@njit(cache=False)
def matmul_seq(A, B, C):
    n, m, p = A.shape[0], A.shape[1], B.shape[1]
    for i in range(n):
        for k in range(m):                 # ikj: B qatori bo'ylab ketma-ket murojaat (kesh uchun qulay)
            a = A[i, k]
            for j in range(p):
                C[i, j] += a * B[k, j]


@njit(parallel=True, cache=False)
def matmul_par(A, B, C):
    n, m, p = A.shape[0], A.shape[1], B.shape[1]
    for i in prange(n):                    # tashqi sikl iplar orasida taqsimlanadi (schedule(static) kabi)
        for k in range(m):
            a = A[i, k]
            for j in range(p):
                C[i, j] += a * B[k, j]


@njit(parallel=True, cache=False)
def yigindi_par(x):
    s = 0.0
    for i in prange(x.shape[0]):           # reduction(+:s) ga ekvivalent — numba avtomatik aniqlaydi
        s += x[i] * x[i]
    return s


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
print(f"Numba {numba.__version__}, iplar: {numba.get_num_threads()}, threading layer tanlanadi...")

# Birinchi chaqiruv = kompilyatsiya vaqti (OpenMP'da bu kompilyator vaqti, bu yerda runtime'da)
t0 = time.perf_counter(); matmul_seq(A, B, C); t_komp_seq = time.perf_counter() - t0
C[:] = 0
t0 = time.perf_counter(); matmul_par(A, B, C); t_komp_par = time.perf_counter() - t0
print(f"JIT kompilyatsiya: seq {t_komp_seq:.2f} s, parallel {t_komp_par:.2f} s (bir marta, keyin keshdan)")
print(f"Threading layer: {numba.threading_layer()}")

# --- Python etalon (kichik n) bilan to'g'rilik ---
etalon = A @ B
C[:] = 0; matmul_par(A, B, C)
print(f"Maks xato (par vs NumPy): {np.abs(C - etalon).max():.2e}")

print(f"\n{'variant':22s} {'iplar':>5s} {'vaqt, ms':>9s} {'GFLOP/s':>8s} {'tezlanish':>9s}")
t_seq = olchov(matmul_seq, A, B, C)
print(f"{'njit seq (ikj)':22s} {1:5d} {t_seq * 1e3:9.2f} {flop / t_seq / 1e9:8.2f} {1.0:9.2f}")
for iplar in (1, 2):
    numba.set_num_threads(iplar)
    t = olchov(matmul_par, A, B, C)
    print(f"{'njit parallel prange':22s} {iplar:5d} {t * 1e3:9.2f} {flop / t / 1e9:8.2f} {t_seq / t:9.2f}")
numba.set_num_threads(2)
t_np = olchov(lambda a, b, c: np.matmul(a, b, out=c), A, B, C)
print(f"{'NumPy BLAS (2 ip)':22s} {2:5d} {t_np * 1e3:9.2f} {flop / t_np / 1e9:8.2f} {t_seq / t_np:9.2f}")

# --- Reduktsiya (yig'indi) 1 vs 2 ip ---
x = np.random.rand(4_000_000).astype(np.float64)
yigindi_par(x)                                   # kompilyatsiya
print("\n--- Reduktsiya: sum(x*x), 4M element ---")
for iplar in (1, 2):
    numba.set_num_threads(iplar)
    v = []
    for _ in range(7):
        t0 = time.perf_counter(); s = yigindi_par(x); v.append(time.perf_counter() - t0)
    t = sorted(v)[3]
    print(f"iplar={iplar}: {t * 1e3:7.2f} ms, {2 * x.nbytes / t / 1e9:5.1f} GB/s samarali (o'qish), natija {s:.1f}")
print("Reduktsiya xotira bilan cheklangan: 2 ip hisobni emas, DRAM o'tkazuvchanligini bo'lishadi.")
print("\nGPU bo'lsa: @njit o'rniga @cuda.jit, prange o'rniga cuda.grid(2) indekslari, massivlar cuda.to_device bilan ko'chiriladi,")
print("vaqt o'lchashdan oldin cuda.synchronize() shart.")
