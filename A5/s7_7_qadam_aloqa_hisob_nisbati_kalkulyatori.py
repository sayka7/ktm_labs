"""A5 — 7-qadam. Aloqa/hisob nisbati kalkulyatori (GPU klasteri uchun baho)

Analitik model: T_step = T_comp + (1-overlap)·T_comm, bunda T_comp FLOPs/(cho'qqi·MFU), T_comm — halqa all-reduce hajmi/tarmoq o'tkazuvchanligi. ResNet-18/50, ViT-B, GPT-2 1.5B uchun 4 xil tarmoqda r = T_comm/T_comp va E(N) hisoblanadi, gradient to'plashning ta'siri ko'rsatiladi, ResNet-50 uchun E(N) grafigi chiziladi. Talaba o'z klasteri parametrlarini (PEAK, MFU, BATCH, BYTES_PER_GRAD) kiritadi.

Kutilgan natija: NVLink'da E≈99%, 10 GbE da ResNet-50 uchun E≈26% (accum=8 bilan 74%); katta model (ViT-B, GPT-2) yaxshiroq masshtablanadi.
"""
# 7-qadam. Aloqa/hisob nisbati kalkulyatori — GPU klasteri uchun taxminiy baho (CPU'da faqat arifmetika)
# Model: T_step = T_comp + (1 - overlap) * T_comm,  T_comm = 2(N-1)/N * S / BW,  T_comp = FLOPs_step / (peak * MFU)
import numpy as np, matplotlib.pyplot as plt

MODELS = {   # nom: (parametrlar, FLOPs bir namuna uchun forward, GFLOP)
    "ResNet-18": (11.7e6, 1.8), "ResNet-50": (25.6e6, 4.1), "ViT-B/16": (86e6, 17.6), "GPT-2 1.5B (seq 1024)": (1.5e9, 3.0e3)}
NETS = {"NVLink 600GB/s": 600e9, "IB 400Gbit/s": 50e9, "IB 100Gbit/s": 12.5e9, "Eth 10Gbit/s": 1.25e9}
PEAK, MFU = 312e12, 0.40          # A100 BF16 cho'qqi va amaldagi foydalanish (Model FLOPs Utilization)
BATCH, BYTES_PER_GRAD, OVERLAP, ACCUM = 128, 2, 0.5, 1   # paket/GPU, BF16 gradient, overlap ulushi, to'plash qadamlari

def estimate(params, gflop, N, bw, batch=BATCH, bpg=BYTES_PER_GRAD, accum=ACCUM, overlap=OVERLAP):
    t_comp = 3 * gflop * 1e9 * batch * accum / (PEAK * MFU)         # forward + backward ~ 3x forward
    t_comm = (2 * (N - 1) / N) * params * bpg / bw if N > 1 else 0.0  # bitta all-reduce
    t_step = t_comp + (1 - overlap) * t_comm
    return t_comp, t_comm, t_comp / (t_comp + (1 - overlap) * t_comm)   # samaradorlik bahosi E(N)

print("=== Aloqa/hisob nisbati (N=8 GPU, paket 128/GPU, BF16 gradient, overlap 50%) ===")
print(f"{'Model':24s} {'T_comp,ms':>10s} " + " ".join(f"{k[:14]:>21s}" for k in NETS))
for name, (p, g) in MODELS.items():
    cells = []
    for bw in NETS.values():
        tc, tm, e = estimate(p, g, 8, bw)
        cells.append(f"{tm*1e3:7.1f}ms r={tm/tc:5.2f} E={e*100:3.0f}%")
    print(f"{name:24s} {tc*1e3:10.2f} " + " ".join(f"{c:>21s}" for c in cells))
print("r = T_comm/T_comp (aloqa/hisob nisbati); r << 1 bo'lsa masshtablash yaxshi.")

print("\n=== Gradient to'plash ta'siri: ResNet-50, 8 GPU, 10 GbE ===")
for acc in (1, 2, 4, 8):
    tc, tm, e = estimate(*MODELS["ResNet-50"], 8, NETS["Eth 10Gbit/s"], accum=acc)
    print(f"accum={acc}: T_comp={tc*1e3:6.1f}ms  T_comm={tm*1e3:6.1f}ms  r={tm/tc:.2f}  E={e*100:.0f}%")

# Grafik: samaradorlik N ga bog'liq, ResNet-50, turli tarmoqlar
Ns = [1, 2, 4, 8, 16, 32, 64]
plt.figure(figsize=(7, 4))
for net, bw in NETS.items():
    plt.plot(Ns, [100 * estimate(*MODELS["ResNet-50"], n, bw)[2] for n in Ns], marker="o", label=net)
plt.xscale("log", base=2); plt.ylim(0, 105); plt.xlabel("N GPU"); plt.ylabel("E(N), %")
plt.title("ResNet-50 DDP samaradorligi bahosi (taxminiy model)"); plt.grid(alpha=.3); plt.legend(fontsize=8)
plt.tight_layout()
print("\nIzoh: bu analitik model; haqiqiy qiymat profiler (torch.profiler, Nsight) bilan o'lchanadi.")
print("Kattaroq model (ViT-B) kichigidan (ResNet-18) yaxshiroq masshtablanadi — hisob tezroq o'sadi, aloqa emas.")
