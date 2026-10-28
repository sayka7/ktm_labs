"""A2 — 7-qadam. Amdahl va Gustafson qonunlari: nazariya va 2 yadroda o'lchov

S(n)=1/((1−p)+p/n) jadvali p=0,5…0,99 uchun chiqariladi; so'ng «dastur» = Python ketma-ket qism + torch matmul 1 va 2 ipda o'lchanib, S(2) dan parallel ulush p baholanadi va shu p bilan 8 hamda 1024 yadrodagi (GPU) tezlanish bashorat qilinadi. Grafikda Amdahl egri chiziqlari, o'lchangan nuqta va Gustafson chiziqlari ko'rsatiladi. Kutilgan natija: faqat matmul p≈1, ketma-ket qism qo'shilganda p≈0,97 → 1024 yadroda tezlanish atigi ~30×.

Kutilgan natija: Amdahl jadvali (p=0,95: n=inf da 20), o'lchov jadvali S(2) ~1,9–2,0, p baholari, 2 panelli grafik.
"""
# 7-qadam. Amdahl va Gustafson qonunlari: nazariy egri chiziqlar + 2 yadroda o'lchangan parallel ulushni baholash
# S(n) = 1 / ((1-p) + p/n). O'lchangan S(2) dan p = (1 - 1/S) / (1 - 1/n) topiladi va
# shu p bilan 8, 64, 1000 yadrodagi (GPU!) tezlanish chegarasi bashorat qilinadi.
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

torch.manual_seed(0)


def amdahl(p, n):
    return 1.0 / ((1 - p) + p / n)


def gustafson(p, n):
    return (1 - p) + p * n


print("=== Amdahl qonuni ===")
print(f"{'p':>5s} " + " ".join(f"{'n=' + str(n):>8s}" for n in (2, 4, 8, 64, 1024, 'inf')))
for p in (0.5, 0.8, 0.9, 0.95, 0.99):
    satr = [amdahl(p, n) for n in (2, 4, 8, 64, 1024)] + [1 / (1 - p)]
    print(f"{p:5.2f} " + " ".join(f"{s:8.2f}" for s in satr))

# --- O'lchov: "dastur" = ketma-ket qism (Python'da oldindan ishlov) + parallel qism (torch matmul) ---
def olchov(f, takror=5):
    f(); f()
    v = []
    for _ in range(takror):
        t0 = time.perf_counter(); f(); v.append(time.perf_counter() - t0)
    return sorted(v)[len(v) // 2]

n = 768
A = torch.randn(n, n); B = torch.randn(n, n)

def ketma_ket_qism():
    # parallellashtirilmagan ish: Python siklida kichik amallar (masalan, NMS, log yozish)
    s = 0.0
    for i in range(60000):
        s += (i % 7) * 0.5
    return s

def dastur(ketma_ket=True):
    if ketma_ket:
        ketma_ket_qism()
    for _ in range(4):
        C = A @ B
    return C

print("\n--- O'lchov: 1 ip vs 2 ip (torch.set_num_threads) ---")
print(f"{'variant':34s} {'1 ip, ms':>9s} {'2 ip, ms':>9s} {'S(2)':>6s} {'p (baho)':>9s} {'S(8)':>6s} {'S(1024)':>8s}")
natijalar = []
for nom, kk in (("faqat matmul (p ~ 1)", False), ("ketma-ket qism + matmul", True)):
    torch.set_num_threads(1); t1 = olchov(lambda: dastur(kk))
    torch.set_num_threads(2); t2 = olchov(lambda: dastur(kk))
    S2 = t1 / t2
    p = min(1.0, max(0.0, (1 - 1 / S2) / (1 - 1 / 2)))
    natijalar.append((nom, p, S2))
    print(f"{nom:34s} {t1 * 1e3:9.1f} {t2 * 1e3:9.1f} {S2:6.2f} {p:9.3f} {amdahl(p, 8):6.2f} {amdahl(p, 1024):8.2f}")
torch.set_num_threads(1); t_seq_qism = olchov(ketma_ket_qism)
print(f"Ketma-ket qismning o'zi: {t_seq_qism * 1e3:.1f} ms (o'lchangan) — Amdahl bahosi bilan solishtiring.")

# --- Grafik ---
nn_ = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
for p in (0.5, 0.9, 0.95, 0.99):
    ax1.plot(nn_, amdahl(p, nn_), label=f"Amdahl p={p}")
for nom, p, S2 in natijalar:
    ax1.plot(nn_, amdahl(p, nn_), "--", label=f"o'lchangan p={p:.2f} ({nom[:14]})")
    ax1.plot([2], [S2], "ko")
ax1.set_xscale("log", base=2); ax1.set_yscale("log", base=2)
ax1.set_xlabel("Protsessorlar soni n"); ax1.set_ylabel("Tezlanish S(n)")
ax1.set_title("Amdahl qonuni: tezlanish chegarasi"); ax1.grid(alpha=0.3, which="both"); ax1.legend(fontsize=7)
for p in (0.5, 0.9, 0.99):
    ax2.plot(nn_, gustafson(p, nn_), label=f"Gustafson p={p}")
    ax2.plot(nn_, amdahl(p, nn_), ":", color="gray")
ax2.set_xscale("log", base=2); ax2.set_yscale("log", base=2)
ax2.set_xlabel("Protsessorlar soni n"); ax2.set_ylabel("Masshtablangan tezlanish")
ax2.set_title("Gustafson (masala hajmi o'sadi) vs Amdahl (nuqtali)"); ax2.grid(alpha=0.3, which="both"); ax2.legend(fontsize=7)
fig.tight_layout()
print("\nXulosa: GPU (minglab yadro) faqat p -> 1 bo'lganda samarali; shuning uchun paket (batch) kattalashtiriladi (Gustafson)")
print("va ketma-ket qismlar (NMS, ma'lumot yuklash, CPU-GPU ko'chirish) alohida optimallashtiriladi.")
