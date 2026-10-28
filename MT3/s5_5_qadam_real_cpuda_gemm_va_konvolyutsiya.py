"""MT3 — 5-qadam. Real CPU'da GEMM va konvolyutsiya o'lchovlari, boshqa arxitekturalarga ekstrapolyatsiya

Sandbox CPU (2 yadro) uchun haqiqiy Roofline quriladi: xotira o'tkazuvchanligi STREAM triad bilan, amaliy cho'qqi esa eng yaxshi FP32 GEMM bilan o'lchanadi. GEMM 64...1024, GEMV, conv3x3, conv1x1, depthwise va ReLU uchun GFLOP/s, arifmetik intensivlik va Roofline'ga nisbatan samaradorlik jadvali chiqadi. Keyin har yadro samaradorligi saqlanadi deb faraz qilinib, 1-qadam pasportidagi arxitekturalarda kutilgan vaqt (ishga tushirish xarajati bilan) ekstrapolyatsiya qilinadi. Bu — birinchi yaqinlashish, aniq bashorat emas.

Kutilgan natija: Triad ~20-40 GB/s, GEMM cho'qqisi ~40-60 GFLOP/s (mashinaga bog'liq); depthwise samaradorligi < 10 %; ekstrapolyatsiyada kichik yadrolar tezlatkichlarda 5-10 us 'pol'ga uriladi.
"""
# 5-qadam. Real CPU'da GEMM va konvolyutsiya o'lchovlari, o'lchangan Roofline va boshqa arxitekturalarga ekstrapolyatsiya
# Sandbox CPU (2 yadro) — bizning "laboratoriya qurilmamiz". Avval uning haqiqiy cho'qqisi (eng yaxshi GEMM) va
# xotira o'tkazuvchanligi (STREAM triad) o'lchanadi, keyin har yadroning samaradorligi (o'lchangan / Roofline chegarasi)
# topiladi va shu samaradorlik bilan GPU/TPU/NPU/FPGA'dagi vaqt TAXMINIY bashorat qilinadi.
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn
import matplotlib.pyplot as plt

torch.manual_seed(0); torch.set_num_threads(2); torch.backends.nnpack.set_flags(False)


def olch(f, takror=7):
    f(); f()
    t = []
    for _ in range(takror):
        t0 = time.perf_counter(); f(); t.append(time.perf_counter() - t0)
    return float(np.median(t))


# --- 1) Xotira o'tkazuvchanligi: STREAM triad a = b + s*c (float32, 3 massiv x 16 MB) ---
n = 4 * 1024 * 1024
b = torch.rand(n); c = torch.rand(n); a = torch.empty(n)
t = olch(lambda: torch.add(b, c, alpha=1.5, out=a), 9)
BW = 3 * n * 4 / t / 1e9
print("=== Sandbox CPU o'lchovlari ===")
print(f"STREAM triad o'tkazuvchanligi: {BW:6.2f} GB/s")

# --- 2) GEMM FP32: o'lcham bo'yicha GFLOP/s ---
natija = []
for N in (64, 128, 256, 512, 1024):
    A = torch.randn(N, N); B = torch.randn(N, N)
    t = olch(lambda: A @ B, 5 if N < 1024 else 3)
    ops, bayt = 2 * N ** 3, 4 * 3 * N * N
    natija.append(("GEMM %d" % N, ops, bayt, t))
# GEMV (LLM dekod kabi) — xotira chegarasida
W = torch.randn(4096, 2048); v = torch.randn(2048)
t = olch(lambda: W @ v, 9); natija.append(("GEMV 4096x2048", 2 * 4096 * 2048, 4 * (4096 * 2048 + 2048 + 4096), t))

# --- 3) Konvolyutsiyalar (batch 8) ---
for nom, Ci, Co, H, k, g in (("conv3x3 32->64 28x28", 32, 64, 28, 3, 1), ("conv1x1 64->128 28x28", 64, 128, 28, 1, 1),
                             ("depthwise 3x3 64ch 56x56", 64, 64, 56, 3, 64)):
    x = torch.randn(8, Ci, H, H); w = torch.randn(Co, Ci // g, k, k)
    t = olch(lambda: Fn.conv2d(x, w, padding=k // 2, groups=g), 5)
    ops = 2 * 8 * H * H * Co * (Ci // g) * k * k
    bayt = 4 * (x.numel() + w.numel() + 8 * Co * H * H)
    natija.append((nom, ops, bayt, t))
# --- element bo'yicha: ReLU ---
z = torch.randn(8, 64, 56, 56)
t = olch(lambda: torch.relu(z), 9); natija.append(("ReLU 8x64x56x56", z.numel(), 8 * z.numel(), t))

df = pd.DataFrame(natija, columns=["yadro", "ops", "bayt", "t_s"])
df["GFLOP/s"] = df.ops / df.t_s / 1e9
df["AI"] = df.ops / df.bayt
PEAK = df["GFLOP/s"].max()                                 # o'lchangan "amaliy cho'qqi"
df["roofline"] = np.minimum(PEAK, df.AI * BW)
df["samaradorlik_%"] = 100 * df["GFLOP/s"] / df["roofline"]   # >100 % — ma'lumot keshga sig'gan (triad BW dan tez)
print(f"Amaliy cho'qqi (eng yaxshi GEMM): {PEAK:6.1f} GFLOP/s, ridge nuqtasi = {PEAK / BW:5.1f} FLOP/bayt\n")
print(df[["yadro", "AI", "GFLOP/s", "roofline", "samaradorlik_%"]].to_string(index=False, float_format=lambda v: f"{v:.2f}"))

# --- 4) Ekstrapolyatsiya: bir xil yadro boshqa arxitekturada (taxminiy cho'qqilar, FP16/INT8 rejimida) ---
ARX = {  # nom: (cho'qqi TOP/s, BW GB/s, ishga tushirish xarajati us — TAXMINIY faraz)  — 1-qadam pasportidan
    "EPYC 9654 (INT8)": (29.5, 460.8, 0), "H100 (FP16)": (989.0, 3350.0, 5), "TPU v4 (BF16)": (275.0, 1200.0, 5),
    "Jetson AGX Orin (INT8)": (138.0, 204.8, 8), "Alveo U250 (INT8)": (33.3, 77.0, 10), "Hailo-8 (INT8)": (26.0, 4.0, 50)}
print("100 % dan katta samaradorlik: matritsa L2/L3 keshga qisman sig'gan — RAM o'tkazuvchanligidan tezroq o'qilgan.")
print("\n=== Ekstrapolyatsiya: kutilgan vaqt, mikrosoniya (samaradorlik sandbox'dagidek + ishga tushirish xarajati) ===")
satr = []
for _, r in df.iterrows():
    s = {"yadro": r.yadro, "sandbox_us": 1e6 * r.t_s}
    e = min(max(r["samaradorlik_%"], 5), 100) / 100            # samaradorlik 5..100 % oralig'ida
    for nom, (pk, bw, ust) in ARX.items():
        bayt_k = r.bayt / 4 * (2 if "FP16" in nom or "BF16" in nom else 1)   # FP32 -> 2 yoki 1 bayt/element
        s[nom.split(" (")[0]] = ust + 1e6 * r.ops / (e * min(pk * 1e12, r.ops / bayt_k * bw * 1e9))
    satr.append(s)
print(pd.DataFrame(satr).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
print("Ogohlantirish: bu birinchi yaqinlashish. Kichik yadrolarda tezlatkich vaqtini ishga tushirish xarajati ('pol') belgilaydi —")
print("shuning uchun tezlatkichlarda yadrolarni birlashtirish (fusion) va katta batch muhim. Xostdan nusxa (PCIe) hisobga olinmagan.")

# --- Grafik: o'lchangan Roofline ---
fig, ax = plt.subplots(figsize=(8, 4.8))
ai = np.logspace(-1.5, 3, 100)
ax.loglog(ai, np.minimum(PEAK, ai * BW), "k-", label=f"sandbox Roofline ({PEAK:.0f} GFLOP/s, {BW:.1f} GB/s)")
for _, r in df.iterrows():
    ax.loglog(r.AI, r["GFLOP/s"], "o"); ax.annotate(r.yadro, (r.AI, r["GFLOP/s"]), fontsize=7)
ax.set_xlabel("AI, FLOP/bayt"); ax.set_ylabel("GFLOP/s"); ax.set_title("Sandbox CPU: o'lchangan Roofline")
ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8); plt.tight_layout()
print("\n=== O'lchovlar tugadi ===")
