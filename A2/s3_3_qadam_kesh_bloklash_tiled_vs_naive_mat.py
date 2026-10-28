"""A2 — 3-qadam. Kesh bloklash (tiled) vs naive matritsa ko'paytirish

Uch algoritm bir ipda solishtiriladi: naive ijk (B ustun bo'ylab — har qadam yangi kesh satri), ikj (qator bo'ylab, vektorlashadi) va bloklangan (tiled) variant; so'ng blok o'lchami 8…256 ta'siri o'lchanadi. Kutilgan natija: ijk → ikj ~10× tezlanish; kichik bloklar sikl xarajati tufayli sekin, 128–256 da ikj darajasiga yetadi. Kodda Numba'ning muhim nozik jihati ko'rsatilgan: range(jj, jj+bs) indekslari manfiy-tekshiruv tufayli vektorlashmaydi, slice bilan hal qilinadi.

Kutilgan natija: n=256/512/1024 jadvali: naive ~1,7 GFLOP/s, ikj ~15–17, tiled ~12; blok o'lchami jadvali 5 → 15 GFLOP/s.
"""
# 3-qadam. Kesh bloklash (cache tiling): naive (ijk) vs ikj vs bloklangan (tiled) matritsa ko'paytirish
# GPU'dagi umumiy xotirali (shared memory) yadroning CPU muqobili — blok L1/L2 keshga sig'adi va har
# yuklangan element ko'p marta qayta ishlatiladi. Numba bilan 1 ipda o'lchaymiz (algoritm farqini ko'rish uchun).
import time
import numpy as np
from numba import njit

BS = 64   # blok o'lchami (o'zgartirib ko'ring: 16, 32, 128)


@njit
def matmul_ijk(A, B, C):
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):             # B[k, j] — ustun bo'ylab, har qadam yangi kesh satri
                s += A[i, k] * B[k, j]
            C[i, j] = s


@njit
def matmul_ikj(A, B, C):
    n = A.shape[0]
    for i in range(n):
        for k in range(n):
            a = A[i, k]
            for j in range(n):             # B[k, j] — qator bo'ylab ketma-ket, vektorlashadi
                C[i, j] += a * B[k, j]


@njit
def matmul_tiled(A, B, C, bs):
    """Bloklangan ikj. Muhim nozik jihat: numba `range(jj, jj+bs)` indekslarini manfiy bo'lishi mumkin deb hisoblab
    har murojaatga tekshiruv qo'shadi va ichki sikl vektorlashmaydi (2 GFLOP/s). Qator bo'lagini (slice) olib,
    0 dan boshlanadigan indeks bilan yursak — vektorlashadi. n bs ga karrali deb olinadi."""
    n = A.shape[0]
    for ii in range(0, n, bs):
        for kk in range(0, n, bs):
            for jj in range(0, n, bs):
                for i in range(ii, ii + bs):
                    Ci = C[i, jj:jj + bs]
                    for k in range(kk, kk + bs):
                        a = A[i, k]
                        Bk = B[k, jj:jj + bs]
                        for j in range(bs):
                            Ci[j] += a * Bk[j]


def olchov(f, A, B, C, *extra, takror=3):
    v = []
    for _ in range(takror):
        C[:] = 0
        t0 = time.perf_counter()
        f(A, B, C, *extra)
        v.append(time.perf_counter() - t0)
    return sorted(v)[len(v) // 2]


print("=== Kesh bloklash ===")
# kompilyatsiya (kichik massivda)
a0 = np.ones((8, 8), np.float32); c0 = np.zeros((8, 8), np.float32)
matmul_ijk(a0, a0, c0); matmul_ikj(a0, a0, c0); matmul_tiled(a0, a0, c0, 4)

print(f"{'variant':16s} {'n':>5s} {'xotira, MB':>10s} {'vaqt, ms':>9s} {'GFLOP/s':>8s} {'xato':>8s}")
for n in (256, 512, 1024):
    A = np.random.rand(n, n).astype(np.float32)
    B = np.random.rand(n, n).astype(np.float32)
    C = np.zeros((n, n), np.float32)
    etalon = A @ B
    flop = 2 * n ** 3
    mb = 3 * n * n * 4 / 1e6
    for nom, f, extra in (("naive ijk", matmul_ijk, ()), ("ikj", matmul_ikj, ()), (f"tiled bs={BS}", matmul_tiled, (BS,))):
        if nom == "naive ijk" and n > 512:
            continue                       # sodda variant katta n da juda sekin (bir necha soniya)
        t = olchov(f, A, B, C, *extra, takror=1 if n >= 512 else 3)
        print(f"{nom:16s} {n:5d} {mb:10.1f} {t * 1e3:9.1f} {flop / t / 1e9:8.2f} {np.abs(C - etalon).max():8.1e}")

# Blok o'lchami ta'siri (n=1024)
n = 1024
A = np.random.rand(n, n).astype(np.float32); B = np.random.rand(n, n).astype(np.float32); C = np.zeros((n, n), np.float32)
print(f"\n--- Blok o'lchami ta'siri, n={n} ---")
print(f"{'bs':>4s} {'blok hajmi (3 blok), KB':>24s} {'vaqt, ms':>9s} {'GFLOP/s':>8s}")
for bs in (8, 16, 32, 64, 128, 256):
    t = olchov(matmul_tiled, A, B, C, bs, takror=1)
    print(f"{bs:4d} {3 * bs * bs * 4 / 1024:24.1f} {t * 1e3:9.1f} {2 * n ** 3 / t / 1e9:8.2f}")
print("\nArifmetik intensivlik: ijk ~ 0.25 FLOP/bayt (har 2 FLOP uchun 8 bayt), tiled ~ bs/6 FLOP/bayt.")
print("Eslatma: bu serverda L3 kesh katta, shuning uchun n=1024 da ikj ham keshdan ishlaydi va tiled undan oshmasligi mumkin;")
print("bloklash foydasi matritsa keshdan katta bo'lganda (n>=4096, 64+ MB) va GPU umumiy xotirasida yaqqol ko'rinadi.")
print("GPU bo'lsa: blok = cuda.shared.array((TPB,TPB)), yuklash -> cuda.syncthreads() -> hisob; TPB=16 da har element 16 marta qayta ishlatiladi.")
