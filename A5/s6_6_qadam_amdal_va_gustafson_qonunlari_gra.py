"""A5 — 6-qadam. Amdal va Gustafson qonunlari grafigi

Parallel ulush p=0.90/0.95/0.99 uchun Amdal (kuchli masshtablash) va Gustafson (kuchsiz) tezlashish egri chiziqlari N=1..256 oralig'ida hisoblanadi va log-log grafikda chiziladi; har N uchun E(N) jadvali va N→∞ chegarasi chiqadi. Talaba o'z o'lchovlaridan (2-qadam) p ni baholab, grafikka qo'shishi mumkin.

Kutilgan natija: Jadval: p=0.90 da S(256)≈9.7 (chegara 10x), p=0.99 da 72; Gustafson deyarli chiziqli. Ikki panelli grafik.
"""
# 6-qadam. Amdal va Gustafson qonunlari: nazariy tezlashish chegaralari (grafik + jadval)
import numpy as np, matplotlib.pyplot as plt

N = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
fracs = [0.90, 0.95, 0.99]           # parallel qilinadigan ulush p

def amdahl(p, n):
    # TODO 1: kuchli masshtablash (ish hajmi o'zgarmas) tezlashishi S(N)
    raise NotImplementedError("TODO 1: Amdal")

def gustafson(p, n):
    # TODO 2: kuchsiz masshtablash (ish hajmi N bilan o'sadi) tezlashishi S(N)
    raise NotImplementedError("TODO 2: Gustafson")

print("=== Amdal qonuni: S(N) = ? ===")
# TODO 3: har N uchun p bo'yicha S va E% jadvali; N -> cheksiz chegarasi
print(amdahl(fracs[0], N))
print("\n=== Gustafson qonuni: S(N) = ? ===")
# TODO 4: N = 8, 64, 256 uchun Gustafson S; ikki panelli log-log grafik (ideal chiziq bilan)
