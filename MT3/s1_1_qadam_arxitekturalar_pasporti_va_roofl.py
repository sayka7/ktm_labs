"""MT3 — 1-qadam. Arxitekturalar pasporti va Roofline modeli

CPU (AMD EPYC 9654), GPU (NVIDIA H100 SXM5), TPU (Google TPU v4), NPU (Hailo-8), chekka SoC (Jetson AGX Orin) va FPGA (AMD Alveo U250) uchun ochiq spetsifikatsiyalardagi cho'qqi qiymatlar (INT8/FP16 TOP/s, xotira o'tkazuvchanligi, quvvat) jadvali tuziladi. Har biri uchun ridge nuqtasi (cho'qqi / o'tkazuvchanlik) va TOP/s/Vt hisoblanadi. Keyin 6 ta tipik AI yadrosi (katta GEMM, ResNet conv3x3, BERT FFN, LLM dekod GEMV, depthwise conv, ReLU) uchun arifmetik intensivlik formuladan topiladi va Roofline bo'yicha erishiladigan unumdorlik hamda chegara turi (hisoblash yoki xotira) ko'rsatiladi. Log-log Roofline grafigi chiziladi. Barcha cho'qqi qiymatlari taxminiy — ular avlod, chastota va rejimga qarab o'zgaradi.

Kutilgan natija: Pasport jadvali, AI jadvali (GEMM ~2700, GEMV ~2 op/bayt), Roofline jadvali: katta GEMM hamma joyda H, GEMV/ReLU hamma joyda X; LLM dekod token/s ~ BW/7 GB; grafik.
"""
# 1-qadam. Arxitekturalar pasporti va Roofline modeli: CPU, GPU, TPU, NPU, FPGA
# Roofline: erishiladigan unumdorlik P = min(P_cho'qqi, AI * BW), bu yerda
#   AI (arifmetik intensivlik) = amallar soni / xotiradan o'qilgan-yozilgan baytlar (op/bayt).
# Cho'qqi qiymatlar ishlab chiqaruvchilarning ochiq spetsifikatsiyalaridan olingan, TAXMINIY (zich, sparsity'siz).
import math
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.width", 160)

# --- Qurilmalar pasporti: cho'qqi (Top/s), xotira o'tkazuvchanligi (GB/s), quvvat (Vt) ---
# INT8 ustuni — inferens uchun, FP16/BF16 — o'qitish uchun. Raqamlar taxminiy, avlod/takt chastotasiga qarab o'zgaradi.
NAN = float("nan")   # ochiq spetsifikatsiyada berilmagan yoki asosiy rejim emas
QURILMA = [
    # nom,                         tur,    FP16/BF16 TFLOP/s, INT8 TOP/s, BW GB/s, quvvat Vt, izoh
    ("AMD EPYC 9654 (96 yadro)",   "CPU",    7.4,    29.5,   460.8, 360, "FP32 ~7,4 TFLOP/s (AVX-512, 2,4 GHz); BF16 alohida tezlanmaydi deb olindi"),
    ("NVIDIA H100 SXM5",           "GPU",  989.0,  1979.0,  3350.0, 700, "Tensor Core, zich; HBM3"),
    ("Google TPU v4 (1 chip)",     "TPU",  275.0,   275.0,  1200.0, 200, "2 yadro x 4 MXU 128x128, 1,05 GHz; quvvat ~170-200 Vt"),
    ("Hailo-8 (M.2)",              "NPU",    NAN,    26.0,     4.0, 2.5, "tashqi DRAM yo'q, xost bilan PCIe Gen3 x4 ~4 GB/s"),
    ("Jetson AGX Orin 64GB",       "SoC",    NAN,   138.0,   204.8,  60, "GPU+DLA; INT8 275 TOP/s sparsity bilan, zich ~138"),
    ("AMD Alveo U250",             "FPGA",   NAN,    33.3,    77.0, 225, "INT8 33,3 TOP/s, DDR4 4x"),
]
df = pd.DataFrame(QURILMA, columns=["nom", "tur", "fp16_tflops", "int8_tops", "bw_gbs", "vt", "izoh"])
df["ridge_int8"] = df.int8_tops * 1e12 / (df.bw_gbs * 1e9)       # op/bayt: shundan yuqorida — hisoblash chegarasi
df["tops_per_vt"] = df.int8_tops / df.vt
print("=== Arxitekturalar pasporti (taxminiy, ochiq spetsifikatsiyalar) ===")
print(df[["nom", "tur", "fp16_tflops", "int8_tops", "bw_gbs", "vt", "ridge_int8", "tops_per_vt"]]
      .to_string(index=False, float_format=lambda v: f"{v:.1f}", na_rep="—"))
print("— : ochiq spetsifikatsiyada berilmagan yoki qurilmaning asosiy rejimi emas")

# --- AI ish yuklamalari: amallar va baytlar formulalardan hisoblanadi (INT8, 1 bayt/element) ---
def gemm(M, K, N, b=1):
    ops = 2 * M * K * N
    bayt = b * (M * K + K * N + M * N)          # ideal: har matritsa xotiradan bir marta o'tadi
    return ops, bayt

def conv(H, W, Ci, Co, k, b=1):
    ops = 2 * H * W * Ci * Co * k * k
    bayt = b * (H * W * Ci + k * k * Ci * Co + H * W * Co)
    return ops, bayt

def dwconv(H, W, C, k, b=1):
    ops = 2 * H * W * C * k * k
    bayt = b * (2 * H * W * C + k * k * C)
    return ops, bayt

# kalit: qisqa nom (jadval ustuni) ; qiymat: (to'liq nom, (amallar, baytlar))
YUKLAMA = {
    "GEMM":    ("GEMM 4096^3 (o'qitish bloki)",     gemm(4096, 4096, 4096)),
    "Conv3x3": ("ResNet conv3x3 56x56x64->64",      conv(56, 56, 64, 64, 3)),
    "BERT":    ("BERT FFN (128 token, 768->3072)",  gemm(128, 768, 3072)),
    "GEMV":    ("LLM dekod GEMV (1x4096x4096)",     gemm(1, 4096, 4096)),
    "DWconv":  ("Depthwise 3x3 112x112x32",         dwconv(112, 112, 32, 3)),
    "ReLU":    ("ReLU/qo'shish (element bo'yicha)", (112 * 112 * 64, 2 * 112 * 112 * 64)),
}
print("\n=== Yuklamalar: arifmetik intensivlik ===")
for k, (nom, (o, b)) in YUKLAMA.items():
    print(f"{nom:34s} amal={o / 1e9:9.3f} G  bayt={b / 1e6:8.2f} MB  AI={o / b:8.1f} op/bayt")

print("\n=== Roofline: erishiladigan INT8 unumdorlik, TOP/s (chegara turi) ===")
satrlar = []
for _, q in df.iterrows():
    satr = {"qurilma": q.nom.split(" (")[0]}
    for k, (nom, (o, b)) in YUKLAMA.items():
        ai = o / b
        p = min(q.int8_tops, ai * q.bw_gbs / 1e3)
        chegara = "H" if q.int8_tops <= ai * q.bw_gbs / 1e3 else "X"    # H - hisoblash, X - xotira
        satr[k] = f"{p:8.2f} {chegara}"
    satrlar.append(satr)
print(pd.DataFrame(satrlar).to_string(index=False))
print("H = hisoblash chegarasi (compute-bound), X = xotira chegarasi (memory-bound)")
print("Eslatma: Hailo-8 uchun BW = xost PCIe — eng yomon holat; og'irliklar chip ichidagi SRAM'ga sig'sa, real natija yuqoriroq.")

# --- LLM dekod misoli: tokenlar/s ~ BW / model hajmi (7B parametr, INT8 = 7 GB) ---
print("\n=== LLM dekod (7B, INT8, batch=1): token/s ~ BW / 7 GB ===")
for _, q in df.iterrows():
    tps = q.bw_gbs / 7.0
    print(f"{q.nom:28s} ~{tps:8.1f} token/s  (cho'qqi hisoblash ulushi: {100 * tps * 14e9 / (q.int8_tops * 1e12):6.2f} %)")
print("Xulosa: dekod bosqichi deyarli hamma joyda xotira chegarasida — HBM o'tkazuvchanligi hal qiladi.")

# --- Grafik: log-log Roofline ---
fig, ax = plt.subplots(figsize=(9, 5.5))
ai_q = [10 ** (e / 20) for e in range(-20, 101)]
for _, q in df.iterrows():
    ax.loglog(ai_q, [min(q.int8_tops, a * q.bw_gbs / 1e3) for a in ai_q], label=f"{q.tur}: {q.nom.split(' (')[0]}")
for k, (nom, (o, b)) in YUKLAMA.items():
    ax.axvline(o / b, color="gray", ls=":", lw=0.8)
    ax.text(o / b, 0.02, k, rotation=90, fontsize=7, va="bottom")
ax.set_xlabel("Arifmetik intensivlik, op/bayt"); ax.set_ylabel("Erishiladigan INT8 TOP/s")
ax.set_title("Roofline: CPU / GPU / TPU / NPU / FPGA (taxminiy cho'qqilar)")
ax.set_ylim(0.01, 5000); ax.legend(fontsize=7, loc="upper left"); ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
print("\nRidge nuqtasi qancha katta bo'lsa, qurilmani to'liq yuklash uchun shuncha 'qalin' (qayta ishlatishli) yuklama kerak.")
