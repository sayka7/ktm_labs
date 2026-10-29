"""A2 — 1-qadam. Sof Python vs NumPy vektorlash

Uch sikldagi sof Python matritsa ko'paytirish (n=64, 128) va NumPy BLAS chaqiruvi bir xil metodika bilan o'lchanadi, GFLOP/s va tezlanish hisoblanadi; so'ng 1M elementli GELU uchun list comprehension va vektorlangan NumPy solishtiriladi. Kutilgan natija: matmul'da 250–500×, GELU'da ~10× tezlanish; NumPy'ning oraliq massivlar tufayli xotira trafigi ko'rsatiladi. n va N ni o'zgartirib tezlanish qanday o'sishini kuzating.

Kutilgan natija: Sof Python ~0,04 GFLOP/s, NumPy FP64 10–20 GFLOP/s, FP32 n=1024 da ~45 GFLOP/s; GELU 236 ms vs 21 ms.
"""
# 1-qadam. Sof Python vs NumPy vektorlash: matritsa ko'paytirish va elementlararo amal
# GPU yo'q — bitta yadroda interpretator va vektorlangan kutubxona farqini o'lchaymiz.
# Metodika: isinish + bir necha takror, median.
import time
import math
import random
import numpy as np

random.seed(0)
np.random.seed(0)


def median_vaqt(f, takror=3):
    """Funksiyani takror marta bajarib median vaqtni (s) qaytaradi (1 isinish)."""
    # TODO 1: avval bir marta isitib oling, so'ng takror marta perf_counter bilan o'lchab median qaytaring
    raise NotImplementedError("TODO 1: median_vaqt")


def matmul_python(A, B, n):
    # TODO 2: ro'yxatlar ustida uch sikl (ikj tartib) bilan C = A*B ni hisoblang
    raise NotImplementedError("TODO 2: matmul_python")


print("=== Sof Python vs NumPy ===")
print(f"{'variant':24s} {'n':>5s} {'vaqt, ms':>10s} {'GFLOP/s':>9s}")
for n in (64, 128):
    A = [[random.random() for _ in range(n)] for _ in range(n)]
    B = [[random.random() for _ in range(n)] for _ in range(n)]
    # TODO 3: FLOP sonini, sof Python va NumPy (An @ Bn, FP64) vaqtini, GFLOP/s, tezlanish va maks xatoni chiqaring
    t_py = median_vaqt(lambda: matmul_python(A, B, n), takror=1 if n >= 128 else 3)

# TODO 4: n=512 va 1024 uchun FP32 NumPy matmul GFLOP/s ni chiqaring

# --- (b) GELU 1M element ---
N = 1_000_000
x_list = [random.uniform(-3, 3) for _ in range(N)]
x_np = np.array(x_list, dtype=np.float32)

def gelu_python(xs):
    # TODO 5: tanh-yaqinlashuvli GELU ni list comprehension bilan yozing
    raise NotImplementedError("TODO 5: gelu_python")

def gelu_numpy(x):
    # TODO 5: xuddi shu formulani np.tanh bilan butun massivga qo'llang
    raise NotImplementedError("TODO 5: gelu_numpy")

# TODO 6: ikkala GELU vaqtini, M element/s, tezlanish va NumPy oraliq massivlar xotira trafigini chiqaring
print(f"\n--- GELU, {N / 1e6:.0f}M element ---")
