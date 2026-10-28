"""A5 — 6-qadam. Amdal va Gustafson qonunlari grafigi

Parallel ulush p=0.90/0.95/0.99 uchun Amdal (kuchli masshtablash) va Gustafson (kuchsiz) tezlashish egri chiziqlari N=1..256 oralig'ida hisoblanadi va log-log grafikda chiziladi; har N uchun E(N) jadvali va N→∞ chegarasi chiqadi. Talaba o'z o'lchovlaridan (2-qadam) p ni baholab, grafikka qo'shishi mumkin.

Kutilgan natija: Jadval: p=0.90 da S(256)≈9.7 (chegara 10x), p=0.99 da 72; Gustafson deyarli chiziqli. Ikki panelli grafik.
"""
# 6-qadam. Amdal va Gustafson qonunlari: nazariy tezlashish chegaralari (grafik + jadval)
import numpy as np, matplotlib.pyplot as plt

N = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
fracs = [0.90, 0.95, 0.99]           # parallel qilinadigan ulush p

def amdahl(p, n):    return 1.0 / ((1 - p) + p / n)            # kuchli masshtablash (ish hajmi o'zgarmas)
def gustafson(p, n): return (1 - p) + p * n                     # kuchsiz masshtablash (ish hajmi N bilan o'sadi)

print("=== Amdal qonuni: S(N) = 1 / ((1-p) + p/N) ===")
print(f"{'N':>4s} " + " ".join(f"p={p:.2f}  S / E%" for p in fracs))
for n in N:
    print(f"{n:4d} " + " ".join(f"{amdahl(p, n):8.2f} /{100*amdahl(p, n)/n:4.0f}%" for p in fracs))
print("\n=== Gustafson qonuni: S(N) = (1-p) + p*N ===")
for n in (8, 64, 256):
    print(f"N={n:3d}: " + ", ".join(f"p={p:.2f} -> S={gustafson(p, n):6.1f}" for p in fracs))

print("\nAmdal chegarasi (N -> cheksiz): " + ", ".join(f"p={p:.2f} -> {1/(1-p):.0f}x" for p in fracs))
# Masalan: ResNet-18 DDP da aloqa ulushi ~12% (4 GPU, PCIe) -> ketma-ket ulush ~0.12 -> maksimal ~8x
p_ddp = 1 - 0.12
print(f"Misol: aloqa ulushi 12% (ketma-ket deb olsak) -> p={p_ddp:.2f}, S(4)={amdahl(p_ddp,4):.2f}, E(4)={100*amdahl(p_ddp,4)/4:.0f}%, chegara {1/(1-p_ddp):.1f}x")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for p in fracs:
    ax[0].plot(N, amdahl(p, N), marker="o", label=f"p={p}")
    ax[1].plot(N, gustafson(p, N), marker="s", label=f"p={p}")
ax[0].plot(N, N, "k--", lw=1, label="ideal"); ax[1].plot(N, N, "k--", lw=1, label="ideal")
ax[0].set(xscale="log", yscale="log", title="Amdal (kuchli masshtablash)", xlabel="N (ishchilar)", ylabel="S(N)")
ax[1].set(xscale="log", yscale="log", title="Gustafson (kuchsiz masshtablash)", xlabel="N (ishchilar)")
for a in ax: a.grid(alpha=.3); a.legend()
plt.tight_layout()
