"""A2 — 1-qadam. Sof Python vs NumPy vektorlash

Uch sikldagi sof Python matritsa ko'paytirish (n=64, 128) va NumPy BLAS chaqiruvi bir xil metodika bilan o'lchanadi, GFLOP/s va tezlanish hisoblanadi; so'ng 1M elementli GELU uchun list comprehension va vektorlangan NumPy solishtiriladi. Kutilgan natija: matmul'da 250–500×, GELU'da ~10× tezlanish; NumPy'ning oraliq massivlar tufayli xotira trafigi ko'rsatiladi. n va N ni o'zgartirib tezlanish qanday o'sishini kuzating.

Kutilgan natija: Sof Python ~0,04 GFLOP/s, NumPy FP64 10–20 GFLOP/s, FP32 n=1024 da ~45 GFLOP/s; GELU 236 ms vs 21 ms.
"""
# 1-qadam. Sof Python vs NumPy vektorlash: matritsa ko'paytirish va elementlararo amal
# Bu yerda GPU yo'q (CUDA mavjud emas) — avval bitta yadroda interpretator va vektorlangan
# kutubxona o'rtasidagi farqni o'lchaymiz. O'lchov metodikasi: isinish + bir necha takror, median.
import time
import random
import numpy as np

random.seed(0)
np.random.seed(0)


def median_vaqt(f, takror=3):
    """Funksiyani takror marta bajarib median vaqtni (s) qaytaradi (1 isinish)."""
    f()
    v = []
    for _ in range(takror):
        t0 = time.perf_counter()
        f()
        v.append(time.perf_counter() - t0)
    return sorted(v)[len(v) // 2]


# --- (a) Matritsa ko'paytirish ---
def matmul_python(A, B, n):
    C = [[0.0] * n for _ in range(n)]
    for i in range(n):
        Ai = A[i]
        Ci = C[i]
        for k in range(n):                 # ikj tartib: B qatori bo'ylab ketma-ket murojaat
            a = Ai[k]
            Bk = B[k]
            for j in range(n):
                Ci[j] += a * Bk[j]
    return C


print("=== Sof Python vs NumPy ===")
print(f"{'variant':24s} {'n':>5s} {'vaqt, ms':>10s} {'GFLOP/s':>9s}")
natijalar = {}
for n in (64, 128):
    A = [[random.random() for _ in range(n)] for _ in range(n)]
    B = [[random.random() for _ in range(n)] for _ in range(n)]
    flop = 2 * n ** 3
    t_py = median_vaqt(lambda: matmul_python(A, B, n), takror=1 if n >= 128 else 3)
    An, Bn = np.array(A, dtype=np.float64), np.array(B, dtype=np.float64)
    t_np = median_vaqt(lambda: An @ Bn, takror=5)
    # to'g'rilikni tekshirish
    xato = np.abs(np.array(matmul_python(A, B, n)) - An @ Bn).max()
    print(f"{'sof Python (3 sikl)':24s} {n:5d} {t_py * 1e3:10.2f} {flop / t_py / 1e9:9.4f}")
    print(f"{'NumPy @ (BLAS, FP64)':24s} {n:5d} {t_np * 1e3:10.3f} {flop / t_np / 1e9:9.2f}")
    natijalar[n] = t_py / t_np
    print(f"  tezlanish {t_py / t_np:8.0f}x, maks xato {xato:.1e}")

for n in (512, 1024):
    An = np.random.rand(n, n).astype(np.float32)
    Bn = np.random.rand(n, n).astype(np.float32)
    t_np = median_vaqt(lambda: An @ Bn, takror=5)
    print(f"{'NumPy @ (BLAS, FP32)':24s} {n:5d} {t_np * 1e3:10.2f} {2 * n ** 3 / t_np / 1e9:9.2f}")

# --- (b) Elementlararo amal: GELU faollashtirish 1M element ---
N = 1_000_000
x_list = [random.uniform(-3, 3) for _ in range(N)]
x_np = np.array(x_list, dtype=np.float32)
import math

def gelu_python(xs):
    return [0.5 * x * (1 + math.tanh(0.79788456 * (x + 0.044715 * x ** 3))) for x in xs]

def gelu_numpy(x):
    return 0.5 * x * (1 + np.tanh(0.79788456 * (x + 0.044715 * x ** 3)))

t_py = median_vaqt(lambda: gelu_python(x_list), takror=1)
t_np = median_vaqt(lambda: gelu_numpy(x_np), takror=5)
print(f"\n--- GELU, {N / 1e6:.0f}M element ---")
print(f"sof Python (list comprehension): {t_py * 1e3:8.1f} ms  ({N / t_py / 1e6:6.1f} M element/s)")
print(f"NumPy (vektorlangan):            {t_np * 1e3:8.1f} ms  ({N / t_np / 1e6:6.1f} M element/s)  tezlanish {t_py / t_np:.0f}x")
# NumPy'da har amal alohida o'tish: ~8 ta oraliq massiv (har biri 4 MB) yoziladi va o'qiladi
trafik = 8 * 2 * N * 4
print(f"NumPy samarali xotira trafigi: ~{trafik / 1e6:.0f} MB -> {trafik / t_np / 1e9:.1f} GB/s (yadro birlashtirilmagan)")
print("\nXulosa: interpretator 1 element uchun ~100 ns sarflaydi; BLAS/vektor buyruqlar 100-1000x tez.")
print("GPU bo'lsa: shu NumPy kodi CuPy'da o'zgarishsiz ishlaydi, lekin 1M element uchun yadro ishga tushirish xarajati (~10 mks) sezilarli.")
