"""A1 — 2-qadam. Platformalar jadvali va talablar matritsasi bo'yicha tanlash skripti

Olti platforma (Orin NX, AGX Orin, A100, Coral Edge TPU, x86 CPU, Hailo-8) uchun haqiqiy unumdorlik, zaxira koeffitsiyenti, dekodlash yetarliligi va bir inferensga energiya hisoblanadi; so'ng 7 mezonli og'irlikli talablar matritsasi bo'yicha ball beriladi. Kutilgan natija: Orin NX ~96 %, Coral va CPU zaxira < 1 bo'lgani uchun yaroqsiz. Og'irliklarni va baholash funksiyalarini o'zgartirib, tanlov qanchalik barqarorligini tekshiring.

Kutilgan natija: Ikki jadval: texnik ko'rsatkichlar (zaxira_koef: Orin NX ~4,9, A100 ~68, Coral 0,44) va og'irlikli ballar; TANLOV: Jetson Orin NX.
"""
# 2-qadam. Platformalar jadvali va talablar matritsasi bo'yicha tanlash skripti
import pandas as pd

# Talab (1-qadam natijasi): yolov8s @ 640, 160 inferens/s
TALAB_TFLOPS = 28.6 * 160 / 1e3       # 4.58 TFLOP/s haqiqiy
KAMERA_OQIM = 16                      # 1080p10 oqimlar soni
GFLOP = 28.6

# Berilgan ma'lumotlar (o'zgartirmang)
platformalar = pd.DataFrame([
    # nom, cho'qqi INT8 dense TOPS, samaradorlik, quvvat Vt, narx $, NVDEC 1080p30 oqim, INT8, FP16, sanoat harorati, ekotizim (0-5)
    ("Jetson Orin NX 16GB",  50,  0.45, 25,  650,   7, True,  True,  True,  5),
    ("Jetson AGX Orin 64GB", 138, 0.45, 60,  2000, 14, True,  True,  True,  5),
    ("A100 PCIe 80GB",       624, 0.50, 300, 10000, 10, True,  True,  False, 5),
    ("Coral Edge TPU M.2",   4,   0.50, 2,   80,    0, True,  False, False, 2),
    ("x86 CPU 8 yadro",      2.5, 0.60, 65,  400,   0, True,  True,  True,  4),
    ("Hailo-8 M.2",          26,  0.50, 3,   200,   0, True,  False, True,  3),
], columns=["nom", "tops_peak", "eff", "vt", "narx", "nvdec_1080p30", "int8", "fp16", "sanoat_t", "ekotizim"])

# TODO 1: tops_real (cho'qqi x samaradorlik) va zaxira_koef (tops_real / talab) ustunlarini qo'shing
# TODO 2: dekod_yetarli — NVDEC 16 ta 1080p10 oqimni dekodlay oladimi?
# TODO 3: inferens_J — bitta inferensga sarflangan energiya (J)

print("=== Platformalarni baholash ===")
print(platformalar[["nom", "tops_peak", "tops_real", "zaxira_koef", "dekod_yetarli", "vt", "narx", "inferens_J"]].to_markdown(index=False))


def ball_zaxira(z):
    # TODO 4: z < 1 -> 0; 1..3 oralig'ida chiziqli; 3 dan yuqori -> 1
    raise NotImplementedError("TODO 4")

# TODO 5: (mezon, og'irlik, funksiya r -> 0..1) ro'yxati: o'tkazuvchanlik 5, kechikish 5, harorat 4,
#         quvvat 3, dekodlash 4, narx 3, ekotizim 3
MEZONLAR = []

# TODO 6: har platforma uchun og'irlikli ballar, jami_% va yaroqli (zaxira >= 1) ni hisoblab, saralang
jadval = None
print("\n--- Og'irlikli ballar (talablar matritsasi, jami 100 %) ---")
# TODO 7: eng yuqori ballli YAROQLI platformani "TANLOV: ..." deb chiqaring
