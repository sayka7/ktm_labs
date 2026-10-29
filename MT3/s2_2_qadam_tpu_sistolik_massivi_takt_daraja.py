"""MT3 — 2-qadam. TPU sistolik massivi: takt darajasidagi simulyator va MXU foydalanish modeli

NumPy'da output-stationary sistolik massiv takt-ma-takt simulyatsiya qilinadi: A satrlari chapdan kechikish (skew) bilan, B ustunlari yuqoridan oqib kiradi, registrlar har taktda siljiydi. Natija A@B bilan aynan mos kelishi, taktlar soni K+R+C-2 formulasiga tengligi va PE foydalanish ulushi tekshiriladi; t=6 dagi diagonal to'lqin fronti ko'rsatiladi. So'ngra TPU v4 ning 128x128 MXU (8 dona, 1,05 GHz = 275 TFLOP/s) uchun weight-stationary analitik model bilan real qatlamlarda (BERT, ResNet, LLM dekod, toq o'lcham) foydalanish va samarali TFLOP/s hisoblanadi.

Kutilgan natija: Simulyatorda xato 0; K=256 da foydalanish ~90 %; MXU: BERT 4096 token ~94 %, GEMV M=1 ~0,4 %, toq N=130 ~45 %; grafik.
"""
# 2-qadam. TPU yuragi — sistolik massiv: takt darajasidagi simulyator va MXU 128x128 foydalanish modeli
# Chap tomondan A satrlari (kechikish bilan, "skew"), yuqoridan B ustunlari oqib kiradi; har PE har taktda
# bitta MAC bajaradi va ma'lumotni qo'shnisiga uzatadi. Natija PE ichida to'planadi (output-stationary).
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)


def sistolik(A, B, R, C, kadr_t=None):
    """R x C sistolik massiv, output-stationary. A: (R, K), B: (K, C). Qaytaradi: natija, taktlar, foydali MAC, kadr."""
    K = A.shape[1]
    a_reg = np.zeros((R, C)); b_reg = np.zeros((R, C)); acc = np.zeros((R, C))
    faol_mac = 0; kadr = None
    # TODO 1: umumiy taktlar soni T ni toping (oxirgi ko'paytma qaysi PE ga qachon keladi?)
    T = None
    for t in range(T):
        # TODO 2: registrlarni siljiting (a — o'ngga, b — pastga)
        # TODO 3: chap ustunga A satrlarini, yuqori satrga B ustunlarini kechikish (skew) bilan kiriting
        # TODO 4: faol PE'larni sanang va acc ga a_reg*b_reg ni qo'shing; kadr_t da faol niqobni saqlang
        raise NotImplementedError("TODO 2-4: bir taktlik sistolik qadam")
    return acc, T, faol_mac, kadr


print("=== Sistolik massiv simulyatori: to'g'rilik tekshiruvi ===")
for R, C, K in [(4, 4, 4), (8, 8, 8), (8, 8, 64), (16, 16, 16), (16, 16, 256)]:
    A = rng.integers(1, 9, (R, K)).astype(float); B = rng.integers(1, 9, (K, C)).astype(float)
    D, T, mac, _ = sistolik(A, B, R, C)
    # TODO 5: xato = |D - A@B| maks., PE-foydalanish = mac / (R*C*T) ni chiqaring

# TODO 6: 8x8 massivda t=6 kadrini '#'/'.' ko'rinishida chiqaring

# --- TPU v4 MXU modeli: 128x128, weight-stationary; 4 MXU x 2 yadro, 1,05 GHz ---
RM = CM = 128
F = 1.05e9; MXU = 8


def mxu_foydalanish(M, K, N, ikki_bufer=True):
    """Og'irlik plitkasi (128x128) yuklanadi, M ta faollashtirish satri oqib o'tadi.
    Har plitka: yuklash 128 takt (ikki buferda yashiriladi) + M + 2*128 - 2 takt. Bo'sh kataklar (padding) — isrof."""
    # TODO 7: plitkalar soni, jami taktlar va foydali MAC / (RM*CM*jami_takt) ni qaytaring
    raise NotImplementedError("TODO 7: MXU foydalanish modeli")


QATLAM = [  # nom, M (token/piksel x batch), K, N
    ("BERT FFN, 512 token",            512, 768, 3072),
    ("BERT FFN, 4096 token (batch 8)", 4096, 768, 3072),
    ("BERT FFN, 32 token",             32, 768, 3072),
    ("ResNet conv1 (K=7*7*3=147)",     112 * 112, 147, 64),
    ("ResNet conv3x3 (K=576, N=64)",   56 * 56, 576, 64),
    ("ResNet conv3x3 (K=4608, N=512)", 7 * 7 * 32, 4608, 512),
    ("LLM dekod GEMV (M=1)",           1, 4096, 4096),
    ("LLM dekod, batch 64",            64, 4096, 4096),
    ("Toq o'lcham (N=130)",            2048, 1024, 130),
]
print("\n=== TPU v4 MXU foydalanish modeli (analitik, taxminiy) ===")
# TODO 8: har qatlam uchun foydalanish_% va samarali_TFLOP/s (275 * u) jadvalini chiqaring, grafik chizing

print("\n=== Simulyatsiya tugadi ===")
