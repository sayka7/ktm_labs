"""MT3 — 5-qadam. Real CPU'da GEMM va konvolyutsiya o'lchovlari, boshqa arxitekturalarga ekstrapolyatsiya

Sandbox CPU (2 yadro) uchun haqiqiy Roofline quriladi: xotira o'tkazuvchanligi STREAM triad bilan, amaliy cho'qqi esa eng yaxshi FP32 GEMM bilan o'lchanadi. GEMM 64...1024, GEMV, conv3x3, conv1x1, depthwise va ReLU uchun GFLOP/s, arifmetik intensivlik va Roofline'ga nisbatan samaradorlik jadvali chiqadi. Keyin har yadro samaradorligi saqlanadi deb faraz qilinib, 1-qadam pasportidagi arxitekturalarda kutilgan vaqt (ishga tushirish xarajati bilan) ekstrapolyatsiya qilinadi. Bu — birinchi yaqinlashish, aniq bashorat emas.

Kutilgan natija: Triad ~20-40 GB/s, GEMM cho'qqisi ~40-60 GFLOP/s (mashinaga bog'liq); depthwise samaradorligi < 10 %; ekstrapolyatsiyada kichik yadrolar tezlatkichlarda 5-10 us 'pol'ga uriladi.
"""
# 5-qadam. Real CPU'da GEMM va konvolyutsiya o'lchovlari, o'lchangan Roofline va boshqa arxitekturalarga ekstrapolyatsiya
# Avval sandbox CPU ning haqiqiy cho'qqisi (eng yaxshi GEMM) va xotira o'tkazuvchanligi (STREAM triad) o'lchanadi,
# keyin har yadro samaradorligi topiladi va shu samaradorlik bilan GPU/TPU/NPU/FPGA'dagi vaqt TAXMINIY bashorat qilinadi.
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn
import matplotlib.pyplot as plt

torch.manual_seed(0); torch.set_num_threads(2); torch.backends.nnpack.set_flags(False)


def olch(f, takror=7):
    # TODO 1: isitish (warm-up) chaqiruvlaridan keyin takror marta o'lchab, median vaqtni (s) qaytaring
    raise NotImplementedError("TODO 1: vaqt o'lchash")


print("=== Sandbox CPU o'lchovlari ===")
n = 4 * 1024 * 1024
b = torch.rand(n); c = torch.rand(n); a = torch.empty(n)
# TODO 2: STREAM triad a = b + 1.5*c ni o'lchab, BW (GB/s) ni hisoblang (nechta massiv x necha bayt?)
BW = None

natija = []   # (yadro nomi, amallar, baytlar, vaqt_s)
# TODO 3: GEMM N = 64..1024 va GEMV 4096x2048 uchun ops, bayt (FP32 = 4 bayt) va vaqtni natija ga qo'shing
# TODO 4: conv3x3 32->64 28x28, conv1x1 64->128 28x28, depthwise 3x3 64ch 56x56 (batch 8) va ReLU 8x64x56x56 ni o'lchang

df = pd.DataFrame(natija, columns=["yadro", "ops", "bayt", "t_s"])
# TODO 5: GFLOP/s, AI, PEAK (amaliy cho'qqi), roofline = min(PEAK, AI*BW) va samaradorlik_% ustunlarini qo'shing
PEAK = None
print(f"Amaliy cho'qqi (eng yaxshi GEMM): {PEAK:6.1f} GFLOP/s, ridge nuqtasi = {PEAK / BW:5.1f} FLOP/bayt\n")

ARX = {  # nom: (cho'qqi TOP/s, BW GB/s, ishga tushirish xarajati us — TAXMINIY faraz)  — 1-qadam pasportidan
    "EPYC 9654 (INT8)": (29.5, 460.8, 0), "H100 (FP16)": (989.0, 3350.0, 5), "TPU v4 (BF16)": (275.0, 1200.0, 5),
    "Jetson AGX Orin (INT8)": (138.0, 204.8, 8), "Alveo U250 (INT8)": (33.3, 77.0, 10), "Hailo-8 (INT8)": (26.0, 4.0, 50)}
print("\n=== Ekstrapolyatsiya: kutilgan vaqt, mikrosoniya (samaradorlik sandbox'dagidek + ishga tushirish xarajati) ===")
# TODO 6: har yadro x arxitektura uchun vaqt = ust + ops / (e * min(cho'qqi, AI_yangi * BW)); e 5..100 % ga cheklanadi

# TODO 7: o'lchangan Roofline grafigini chizing (chiziq + nuqtalar)
print("\n=== O'lchovlar tugadi ===")
