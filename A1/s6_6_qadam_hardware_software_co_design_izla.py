"""A1 — 6-qadam. Hardware–Software Co-design: izlanish fazosini grid search bilan ko'rib chiqish

Dasturiy qarorlar (3 model × 3 imgsz × 2 format × 3 batch) va apparat qarorlari (6 platforma) birgalikda — 270 ta konfiguratsiya — zaxira, kechikish, quvvat, narx, harorat va format cheklovlari bo'yicha baholanadi. Yaroqli konfiguratsiyalar orasidan mAP bo'yicha eng yaxshilari, har platforma uchun eng yaxshi variant va narx–sifat Pareto chegarasi topiladi; grafikda yaroqli (rangli) va yaroqsiz (kulrang) nuqtalar ko'rsatiladi. Cheklov qiymatlarini o'zgartirib, izlanish fazosi qanday torayishini kuzating.

Kutilgan natija: 270 konfiguratsiya, ~118 yaroqli; top-10 jadval (yolov8m AGX Orin / Orin NX INT8), Pareto ro'yxati (Hailo-8 → Orin NX → AGX Orin), scatter grafik.
"""
# 6-qadam. Hardware-Software Co-design: izlanish fazosini to'liq ko'rib chiqish (grid search)
import itertools
import pandas as pd
import matplotlib.pyplot as plt

KAMERALAR, FPS = 16, 10
INFERENS_S = KAMERALAR * FPS
INFERENS_BYUDJET_MS = 60       # kechikish byudjetidan inferensga ajratilgan qism
MIN_ZAXIRA = 1.5               # o'tkazuvchanlik zaxirasi (pik yuklama uchun)
MAKS_VT, MAKS_NARX = 60, 2000

# Berilgan ma'lumotlar (o'zgartirmang)
MODELLAR = {"yolov8n": (8.7, 37.3), "yolov8s": (28.6, 44.9), "yolov8m": (78.9, 50.2)}
IMGSZ = (416, 512, 640)
FORMATLAR = {"FP16": 2, "INT8": 1}
BATCHLAR = (1, 4, 8)
# platforma: INT8 dense TOPS, samaradorlik, Vt, narx, NVDEC 1080p30 oqim, FP16 bor, INT8 bor, sanoat harorati
PLATFORMALAR = {
    "Orin NX 16GB":  (50,  0.45, 25,  650,   7, True,  True,  True),
    "AGX Orin 64GB": (138, 0.45, 60,  2000, 14, True,  True,  True),
    "A100 PCIe":     (624, 0.50, 300, 10000, 10, True,  True,  False),
    "Coral TPU":     (4,   0.50, 2,   80,    0, False, True,  False),
    "Hailo-8":       (26,  0.50, 3,   200,   0, False, True,  True),
    "x86 CPU 8c":    (2.5, 0.60, 65,  400,   0, True,  True,  True),
}


def baholash(model, imgsz, fmt, batch, plat):
    """Bitta konfiguratsiya uchun ko'rsatkichlar lug'ati yoki None (format qo'llanmasa)."""
    # TODO 1: GFLOP (imgsz^2 ga proporsional) va mAP proxysi (kichik kirish -12*(1-imgsz/640), INT8 -1)
    # TODO 2: format qo'llab-quvvatlanmasa None; FP16 da tops_real yarmi; zaxira = tops_real / talab
    # TODO 3: kechikish = paket hisoblash (batch=1 da x1.3) + paket to'lguncha kutish
    # TODO 4: dekoder yo'q bo'lsa +20 Vt; 5 cheklov (zaxira, kechikish, quvvat, narx, harorat)
    raise NotImplementedError("TODO 1-4: baholash")


# TODO 5: itertools.product bilan barcha kombinatsiyalarni baholab DataFrame yarating
df = None

print("=== Co-design izlanish fazosi ===")
print(f"Jami konfiguratsiyalar: {len(df)}, yaroqli: {df.yaroqli.sum()}")

# TODO 6: yaroqlilarni mAP bo'yicha saralab top-10 va har platforma uchun eng yaxshisini chiqaring
# TODO 7: narx bo'yicha saralab "Pareto chegarasi (narx vs mAP)" ni toping va scatter grafik chizing
