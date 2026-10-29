"""MT3 — 1-qadam. Arxitekturalar pasporti va Roofline modeli

CPU (AMD EPYC 9654), GPU (NVIDIA H100 SXM5), TPU (Google TPU v4), NPU (Hailo-8), chekka SoC (Jetson AGX Orin) va FPGA (AMD Alveo U250) uchun ochiq spetsifikatsiyalardagi cho'qqi qiymatlar (INT8/FP16 TOP/s, xotira o'tkazuvchanligi, quvvat) jadvali tuziladi. Har biri uchun ridge nuqtasi (cho'qqi / o'tkazuvchanlik) va TOP/s/Vt hisoblanadi. Keyin 6 ta tipik AI yadrosi (katta GEMM, ResNet conv3x3, BERT FFN, LLM dekod GEMV, depthwise conv, ReLU) uchun arifmetik intensivlik formuladan topiladi va Roofline bo'yicha erishiladigan unumdorlik hamda chegara turi (hisoblash yoki xotira) ko'rsatiladi. Log-log Roofline grafigi chiziladi. Barcha cho'qqi qiymatlari taxminiy — ular avlod, chastota va rejimga qarab o'zgaradi.

Kutilgan natija: Pasport jadvali, AI jadvali (GEMM ~2700, GEMV ~2 op/bayt), Roofline jadvali: katta GEMM hamma joyda H, GEMV/ReLU hamma joyda X; LLM dekod token/s ~ BW/7 GB; grafik.
"""
# 1-qadam. Arxitekturalar pasporti va Roofline modeli: CPU, GPU, TPU, NPU, FPGA
# Cho'qqi qiymatlar ishlab chiqaruvchilarning ochiq spetsifikatsiyalaridan olingan, TAXMINIY (zich, sparsity'siz).
import math
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.width", 160)

# --- Berilgan ma'lumotlar (o'zgartirmang) ---
NAN = float("nan")
QURILMA = [
    # nom,                         tur,    FP16/BF16 TFLOP/s, INT8 TOP/s, BW GB/s, quvvat Vt, izoh
    ("AMD EPYC 9654 (96 yadro)",   "CPU",    7.4,    29.5,   460.8, 360, "AVX-512, 2,4 GHz"),
    ("NVIDIA H100 SXM5",           "GPU",  989.0,  1979.0,  3350.0, 700, "Tensor Core, zich; HBM3"),
    ("Google TPU v4 (1 chip)",     "TPU",  275.0,   275.0,  1200.0, 200, "2 yadro x 4 MXU 128x128"),
    ("Hailo-8 (M.2)",              "NPU",    NAN,    26.0,     4.0, 2.5, "xost bilan PCIe ~4 GB/s"),
    ("Jetson AGX Orin 64GB",       "SoC",    NAN,   138.0,   204.8,  60, "GPU+DLA; zich ~138"),
    ("AMD Alveo U250",             "FPGA",   NAN,    33.3,    77.0, 225, "DDR4 4x"),
]
df = pd.DataFrame(QURILMA, columns=["nom", "tur", "fp16_tflops", "int8_tops", "bw_gbs", "vt", "izoh"])

# TODO 1: ridge nuqtasi (op/bayt) va energiya samaradorligi (TOP/s/Vt) ustunlarini qo'shing
df["ridge_int8"] = None
df["tops_per_vt"] = None
print("=== Arxitekturalar pasporti (taxminiy, ochiq spetsifikatsiyalar) ===")
print(df[["nom", "tur", "int8_tops", "bw_gbs", "vt", "ridge_int8", "tops_per_vt"]].to_string(index=False))

# TODO 2: amallar soni va baytlar (INT8, b=1 bayt/element) formulalarini yozing
def gemm(M, K, N, b=1):
    raise NotImplementedError("TODO 2: GEMM amallari va baytlari")

def conv(H, W, Ci, Co, k, b=1):
    raise NotImplementedError("TODO 2: conv amallari va baytlari")

def dwconv(H, W, C, k, b=1):
    raise NotImplementedError("TODO 2: depthwise conv amallari va baytlari")

YUKLAMA = {
    "GEMM":    ("GEMM 4096^3 (o'qitish bloki)",     gemm(4096, 4096, 4096)),
    "Conv3x3": ("ResNet conv3x3 56x56x64->64",      conv(56, 56, 64, 64, 3)),
    "BERT":    ("BERT FFN (128 token, 768->3072)",  gemm(128, 768, 3072)),
    "GEMV":    ("LLM dekod GEMV (1x4096x4096)",     gemm(1, 4096, 4096)),
    "DWconv":  ("Depthwise 3x3 112x112x32",         dwconv(112, 112, 32, 3)),
    "ReLU":    ("ReLU/qo'shish (element bo'yicha)", (112 * 112 * 64, 2 * 112 * 112 * 64)),
}
print("\n=== Yuklamalar: arifmetik intensivlik ===")
# TODO 3: har yuklama uchun AI = amal / bayt ni chiqaring

print("\n=== Roofline: erishiladigan INT8 unumdorlik, TOP/s (chegara turi) ===")
# TODO 4: har qurilma x yuklama uchun P = min(cho'qqi, AI*BW) va chegara turini (H/X) jadvalga yig'ing

print("\n=== LLM dekod (7B, INT8, batch=1): token/s ~ BW / 7 GB ===")
# TODO 5: har qurilma uchun token/s ni baholang

# TODO 6: log-log Roofline grafigini chizing (ax.loglog)
