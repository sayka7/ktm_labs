"""A5 — 7-qadam. Aloqa/hisob nisbati kalkulyatori (GPU klasteri uchun baho)

Analitik model: T_step = T_comp + (1-overlap)·T_comm, bunda T_comp FLOPs/(cho'qqi·MFU), T_comm — halqa all-reduce hajmi/tarmoq o'tkazuvchanligi. ResNet-18/50, ViT-B, GPT-2 1.5B uchun 4 xil tarmoqda r = T_comm/T_comp va E(N) hisoblanadi, gradient to'plashning ta'siri ko'rsatiladi, ResNet-50 uchun E(N) grafigi chiziladi. Talaba o'z klasteri parametrlarini (PEAK, MFU, BATCH, BYTES_PER_GRAD) kiritadi.

Kutilgan natija: NVLink'da E≈99%, 10 GbE da ResNet-50 uchun E≈26% (accum=8 bilan 74%); katta model (ViT-B, GPT-2) yaxshiroq masshtablanadi.
"""
# 7-qadam. Aloqa/hisob nisbati kalkulyatori — GPU klasteri uchun taxminiy baho (CPU'da faqat arifmetika)
import numpy as np, matplotlib.pyplot as plt

MODELS = {   # nom: (parametrlar, FLOPs bir namuna uchun forward, GFLOP)
    "ResNet-18": (11.7e6, 1.8), "ResNet-50": (25.6e6, 4.1), "ViT-B/16": (86e6, 17.6), "GPT-2 1.5B (seq 1024)": (1.5e9, 3.0e3)}
NETS = {"NVLink 600GB/s": 600e9, "IB 400Gbit/s": 50e9, "IB 100Gbit/s": 12.5e9, "Eth 10Gbit/s": 1.25e9}
PEAK, MFU = 312e12, 0.40          # A100 BF16 cho'qqi va amaldagi foydalanish
BATCH, BYTES_PER_GRAD, OVERLAP, ACCUM = 128, 2, 0.5, 1

def estimate(params, gflop, N, bw, batch=BATCH, bpg=BYTES_PER_GRAD, accum=ACCUM, overlap=OVERLAP):
    # TODO 1: t_comp — bir qadam hisob vaqti (forward + backward, accum mikro-paket)
    # TODO 2: t_comm — bitta halqa all-reduce vaqti (N=1 da 0)
    # TODO 3: (t_comp, t_comm, E) qaytaring, E — overlap hisobga olingan samaradorlik
    raise NotImplementedError("TODO 1-3: estimate")

print("=== Aloqa/hisob nisbati (N=8 GPU, paket 128/GPU, BF16 gradient, overlap 50%) ===")
# TODO 4: har model va tarmoq uchun T_comm, r = T_comm/T_comp va E ni jadvalda chiqaring
print(estimate(*MODELS["ResNet-50"], 8, NETS["Eth 10Gbit/s"]))
print("\n=== Gradient to'plash ta'siri: ResNet-50, 8 GPU, 10 GbE ===")
# TODO 5: accum = 1, 2, 4, 8 uchun T_comp, T_comm, r, E; ResNet-50 uchun E(N) grafigi (N=1..64)
