"""MT3 — 2-qadam. TPU sistolik massivi: takt darajasidagi simulyator va MXU foydalanish modeli

NumPy'da output-stationary sistolik massiv takt-ma-takt simulyatsiya qilinadi: A satrlari chapdan kechikish (skew) bilan, B ustunlari yuqoridan oqib kiradi, registrlar har taktda siljiydi. Natija A@B bilan aynan mos kelishi, taktlar soni K+R+C-2 formulasiga tengligi va PE foydalanish ulushi tekshiriladi; t=6 dagi diagonal to'lqin fronti ko'rsatiladi. So'ngra TPU v4 ning 128x128 MXU (8 dona, 1,05 GHz = 275 TFLOP/s) uchun weight-stationary analitik model bilan real qatlamlarda (BERT, ResNet, LLM dekod, toq o'lcham) foydalanish va samarali TFLOP/s hisoblanadi.

Kutilgan natija: Simulyatorda xato 0; K=256 da foydalanish ~90 %; MXU: BERT 4096 token ~94 %, GEMV M=1 ~0,4 %, toq N=130 ~45 %; grafik.
"""
# 2-qadam. TPU yuragi — sistolik massiv: takt darajasidagi simulyator va MXU 128x128 foydalanish modeli
# Chap tomondan A satrlari (kechikish bilan, "skew"), yuqoridan B ustunlari oqib kiradi; har PE (protsessor elementi)
# har taktda bitta MAC bajaradi va ma'lumotni qo'shnisiga uzatadi. Natija PE ichida to'planadi (output-stationary).
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
    T = K + R + C - 2                      # oxirgi ko'paytma PE(R-1, C-1) ga t = (K-1)+(R-1)+(C-1) da keladi
    for t in range(T):
        # registrlar siljiydi: a o'ngga, b pastga (bu — "sistolik" yurak urishi)
        a_reg[:, 1:] = a_reg[:, :-1].copy(); b_reg[1:, :] = b_reg[:-1, :].copy()
        for i in range(R):                 # i-satr t-i taktda boshlanadi (skew)
            k = t - i
            a_reg[i, 0] = A[i, k] if 0 <= k < K else 0.0
        for j in range(C):
            k = t - j
            b_reg[0, j] = B[k, j] if 0 <= k < K else 0.0
        faol = (a_reg != 0) & (b_reg != 0)
        faol_mac += int(faol.sum())
        acc += a_reg * b_reg
        if kadr_t == t:
            kadr = faol.copy()
    return acc, T, faol_mac, kadr


print("=== Sistolik massiv simulyatori: to'g'rilik tekshiruvi ===")
print(f"{'massiv':>7s} {'K':>4s} {'taktlar':>8s} {'formula K+R+C-2':>16s} {'MAC':>7s} {'PE-foydalanish':>15s} {'xato':>9s}")
for R, C, K in [(4, 4, 4), (8, 8, 8), (8, 8, 64), (16, 16, 16), (16, 16, 256)]:
    A = rng.integers(1, 9, (R, K)).astype(float); B = rng.integers(1, 9, (K, C)).astype(float)   # nol yo'q — faol PE sanash aniq
    D, T, mac, _ = sistolik(A, B, R, C)
    xato = np.abs(D - A @ B).max()
    print(f"{R:>3d}x{C:<3d} {K:4d} {T:8d} {K + R + C - 2:16d} {mac:7d} {100 * mac / (R * C * T):14.1f}% {xato:9.1e}")
print("K katta bo'lganda to'ldirish/bo'shatish (fill/drain) ulushi kamayadi va foydalanish 100 % ga yaqinlashadi.")

# --- To'lqin fronti: 8x8 massivda 6-taktda qaysi PE'lar ishlayapti ---
A = rng.integers(1, 9, (8, 8)).astype(float); B = rng.integers(1, 9, (8, 8)).astype(float)
_, _, _, kadr = sistolik(A, B, 8, 8, kadr_t=6)
print("\n=== 8x8 massiv, t=6: faol PE'lar (#) — diagonal to'lqin ===")
for i in range(8):
    print("   " + " ".join("#" if kadr[i, j] else "." for j in range(8)))

# --- TPU v4 MXU modeli: 128x128, weight-stationary; 4 MXU x 2 yadro, 1,05 GHz ---
RM = CM = 128
F = 1.05e9; MXU = 8
print(f"\nNazariy cho'qqi: {MXU} MXU x {RM}x{CM} x 2 op x {F / 1e9:.2f} GHz = {MXU * RM * CM * 2 * F / 1e12:.0f} TFLOP/s (BF16)")


def mxu_foydalanish(M, K, N, ikki_bufer=True):
    """Og'irlik plitkasi (128x128) yuklanadi, M ta faollashtirish satri oqib o'tadi.
    Har plitka: yuklash 128 takt (ikki buferda yashiriladi) + M + 2*128 - 2 takt. Bo'sh kataklar (padding) — isrof."""
    plitkalar = math.ceil(K / RM) * math.ceil(N / CM)
    takt_plitka = M + RM + CM - 2 + (0 if ikki_bufer else RM)
    jami_takt = plitkalar * takt_plitka
    return (M * K * N) / (RM * CM * jami_takt)


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
satr = []
for nom, M, K, N in QATLAM:
    u = mxu_foydalanish(M, K, N)
    satr.append({"qatlam": nom, "M": M, "K": K, "N": N, "foydalanish_%": 100 * u,
                 "samarali_TFLOP/s": 275 * u, "ikki_bufersiz_%": 100 * mxu_foydalanish(M, K, N, False)})
df = pd.DataFrame(satr)
print("\n=== TPU v4 MXU foydalanish modeli (analitik, taxminiy) ===")
print(df.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
print("Xulosa: M kichik (dekod, kichik batch) yoki K/N 128 ga karrali bo'lmasa, MXU bo'sh turadi;")
print("XLA kompilyatori shuning uchun batch va o'lchamlarni 128 ga moslashni (padding) talab qiladi.")

# --- Grafik: foydalanish vs M (batch x token) ---
Ms = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
fig, ax = plt.subplots(figsize=(8, 4.5))
for K, N in [(768, 3072), (4096, 4096), (147, 64)]:
    ax.semilogx(Ms, [100 * mxu_foydalanish(m, K, N) for m in Ms], marker="o", label=f"K={K}, N={N}")
ax.set_xlabel("M (batch x token / piksel)"); ax.set_ylabel("MXU foydalanish, %")
ax.set_title("Sistolik massiv 128x128: foydalanish M ga bog'liq"); ax.grid(alpha=0.3); ax.legend()
plt.tight_layout()
print("\n=== Simulyatsiya tugadi ===")
