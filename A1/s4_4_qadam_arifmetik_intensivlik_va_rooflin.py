"""A1 — 4-qadam. Arifmetik intensivlik va Roofline modeli

Har platforma uchun tizza nuqtasi (P_peak / B_mem) hisoblanadi; YOLOv8s (FP16/INT8, batch 1/8), GEMM va GELU yuklamalari uchun AI = FLOP/bayt topilib, erishiladigan unumdorlik P = min(P_peak, AI·B_mem) va rejim (hisoblash yoki xotira bilan cheklangan) aniqlanadi. Kutilgan natija: YOLOv8s FP16 batch 1 Orin NX'da xotira bilan (X), A100'da hisoblash bilan (H) cheklangan; log-log Roofline grafigi chiziladi.

Kutilgan natija: Tizza nuqtalari (A100 INT8 ~322, Orin NX ~490 FLOP/bayt), yuklama×platforma jadvali H/X belgilari bilan, Roofline grafigi.
"""
# 4-qadam. Arifmetik intensivlik va Roofline modeli
# P = min(P_peak, AI * B_mem). Har platforma uchun tizza nuqtasi (ridge point) hisoblanadi,
# YOLOv8s yuklamasi turli batch'larda va boshqa yuklamalar (GEMM, elementlararo amal) joylashtiriladi.
import numpy as np
import matplotlib.pyplot as plt

# Platforma: nom, cho'qqi unumdorlik (TFLOP/s yoki TOPS, dense), xotira o'tkazuvchanligi (GB/s)
PLATFORMALAR = [
    ("A100 PCIe (INT8 TC)",   624, 1935),
    ("A100 PCIe (FP16 TC)",   312, 1935),
    ("Orin NX (INT8 dense)",  50,  102),
    ("Coral Edge TPU",        4,   32),     # host DRAM orqali (taxminiy)
    ("x86 CPU 8 yadro (FP32)", 1.0, 70),
]

# Yuklama: FLOP va xotira trafigi (bayt) — soddalashtirilgan model
GFLOP = 28.6e9                 # yolov8s @ 640, bir inferens
PARAMS = 11.17e6
AKTIV_BAYT_FP16 = 60e6         # faollashuvlar o'qish+yozish (taxminiy, FP16)


def yuklama_ai(batch, bayt_param):
    """Arifmetik intensivlik: vaznlar paket uchun bir marta o'qiladi, faollashuvlar har kadr uchun."""
    flop = GFLOP * batch
    bayt = PARAMS * bayt_param + AKTIV_BAYT_FP16 * batch * (bayt_param / 2)
    return flop / bayt


print("=== Roofline tahlili ===")
print(f"{'Platforma':26s} {'P_peak':>8s} {'B_mem GB/s':>10s} {'tizza AI':>9s}")
for nom, peak, bw in PLATFORMALAR:
    print(f"{nom:26s} {peak:8.1f} {bw:10.0f} {peak * 1e12 / (bw * 1e9):9.0f}")

YUKLAMALAR = [
    ("YOLOv8s FP16, batch 1",  yuklama_ai(1, 2)),
    ("YOLOv8s FP16, batch 8",  yuklama_ai(8, 2)),
    ("YOLOv8s INT8, batch 1",  yuklama_ai(1, 1)),
    ("YOLOv8s INT8, batch 8",  yuklama_ai(8, 1)),
    ("GEMM 1024x1024 FP32",    2 * 1024 ** 3 / (3 * 1024 ** 2 * 4)),   # n/6 FLOP/bayt
    ("GELU elementlararo FP32", 8 / 8),                                   # ~8 FLOP / 8 bayt
]

print(f"\n{'Yuklama':26s} {'AI':>7s}  " + "  ".join(f"{p[0][:14]:>14s}" for p in PLATFORMALAR))
for nom, ai in YUKLAMALAR:
    hujayra = []
    for pnom, peak, bw in PLATFORMALAR:
        erishiladigan = min(peak, ai * bw / 1e3)        # TFLOP/s
        rejim = "H" if ai * bw / 1e3 >= peak else "X"   # H = hisoblash bilan, X = xotira bilan cheklangan
        hujayra.append(f"{erishiladigan:9.1f} {rejim}")
    print(f"{nom:26s} {ai:7.1f}  " + "  ".join(f"{h:>14s}" for h in hujayra))
print("H = hisoblash bilan cheklangan, X = xotira bilan cheklangan (TFLOP/s erishiladigan)")

# Talab bilan solishtirish: 160 inferens/s uchun 4.58 TFLOP/s kerak
TALAB = 4.58
print(f"\nTalab: {TALAB} TFLOP/s. Orin NX INT8 batch 1 da erishiladigan: {min(50, yuklama_ai(1, 1) * 102 / 1e3):.1f} TFLOP/s")

# --- Roofline grafigi ---
ai_x = np.logspace(-1, 4, 200)
fig, ax = plt.subplots(figsize=(8, 5))
for nom, peak, bw in PLATFORMALAR:
    ax.plot(ai_x, np.minimum(peak, ai_x * bw / 1e3), label=nom)
for nom, ai in YUKLAMALAR:
    ax.axvline(ai, color="gray", ls=":", lw=0.8)
    ax.text(ai, 0.15, nom, rotation=90, fontsize=7, va="bottom", ha="right")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Arifmetik intensivlik, FLOP/bayt")
ax.set_ylabel("Erishiladigan unumdorlik, TFLOP/s")
ax.set_title("Roofline: platformalar va yuklamalar")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
