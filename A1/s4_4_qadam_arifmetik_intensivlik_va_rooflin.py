"""A1 — 4-qadam. Arifmetik intensivlik va Roofline modeli

Har platforma uchun tizza nuqtasi (P_peak / B_mem) hisoblanadi; YOLOv8s (FP16/INT8, batch 1/8), GEMM va GELU yuklamalari uchun AI = FLOP/bayt topilib, erishiladigan unumdorlik P = min(P_peak, AI·B_mem) va rejim (hisoblash yoki xotira bilan cheklangan) aniqlanadi. Kutilgan natija: YOLOv8s FP16 batch 1 Orin NX'da xotira bilan (X), A100'da hisoblash bilan (H) cheklangan; log-log Roofline grafigi chiziladi.

Kutilgan natija: Tizza nuqtalari (A100 INT8 ~322, Orin NX ~490 FLOP/bayt), yuklama×platforma jadvali H/X belgilari bilan, Roofline grafigi.
"""
# 4-qadam. Arifmetik intensivlik va Roofline modeli
import numpy as np
import matplotlib.pyplot as plt

# Berilgan ma'lumotlar: nom, cho'qqi unumdorlik (TFLOP/s yoki TOPS), xotira o'tkazuvchanligi (GB/s)
PLATFORMALAR = [
    ("A100 PCIe (INT8 TC)",   624, 1935),
    ("A100 PCIe (FP16 TC)",   312, 1935),
    ("Orin NX (INT8 dense)",  50,  102),
    ("Coral Edge TPU",        4,   32),     # host DRAM orqali (taxminiy)
    ("x86 CPU 8 yadro (FP32)", 1.0, 70),
]
GFLOP = 28.6e9                 # yolov8s @ 640, bir inferens
PARAMS = 11.17e6
AKTIV_BAYT_FP16 = 60e6         # faollashuvlar o'qish+yozish (taxminiy, FP16)


def yuklama_ai(batch, bayt_param):
    """Arifmetik intensivlik (FLOP/bayt): vaznlar paketga bir marta, faollashuvlar har kadrga."""
    # TODO 2: paket uchun FLOP va xotira trafigini (bayt) hisoblab, nisbatini qaytaring
    raise NotImplementedError("TODO 2: yuklama_ai")


print("=== Roofline tahlili ===")
print(f"{'Platforma':26s} {'P_peak':>8s} {'B_mem GB/s':>10s} {'tizza AI':>9s}")
# TODO 1: har platforma uchun tizza nuqtasi AI = P_peak / B_mem (FLOP/bayt) ni chiqaring

# TODO 3: YOLOv8s FP16/INT8 x batch 1/8, GEMM 1024 FP32 va GELU yuklamalari ro'yxati (nom, AI)
YUKLAMALAR = [
    ("YOLOv8s FP16, batch 1",  yuklama_ai(1, 2)),
    # ... qolganlarini qo'shing
]

# TODO 4: har yuklama x platforma uchun erishiladigan P = min(P_peak, AI*B_mem) va rejim H/X
print("H = hisoblash bilan cheklangan, X = xotira bilan cheklangan (TFLOP/s erishiladigan)")

# TODO 5: log-log Roofline grafigi: har platforma chizig'i va yuklamalar vertikal chiziqlari
