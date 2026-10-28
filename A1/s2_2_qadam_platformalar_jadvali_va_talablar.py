"""A1 — 2-qadam. Platformalar jadvali va talablar matritsasi bo'yicha tanlash skripti

Olti platforma (Orin NX, AGX Orin, A100, Coral Edge TPU, x86 CPU, Hailo-8) uchun haqiqiy unumdorlik, zaxira koeffitsiyenti, dekodlash yetarliligi va bir inferensga energiya hisoblanadi; so'ng 7 mezonli og'irlikli talablar matritsasi bo'yicha ball beriladi. Kutilgan natija: Orin NX ~96 %, Coral va CPU zaxira < 1 bo'lgani uchun yaroqsiz. Og'irliklarni va baholash funksiyalarini o'zgartirib, tanlov qanchalik barqarorligini tekshiring.

Kutilgan natija: Ikki jadval: texnik ko'rsatkichlar (zaxira_koef: Orin NX ~4,9, A100 ~68, Coral 0,44) va og'irlikli ballar; TANLOV: Jetson Orin NX.
"""
# 2-qadam. Platformalar jadvali va talablar matritsasi bo'yicha tanlash skripti
# Har platforma uchun haqiqiy unumdorlik (cho'qqi x samaradorlik), zaxira koeffitsiyenti,
# dekodlash yetarliligi va og'irlikli ball hisoblanadi. Qiymatlar rasmiy tavsiflardan (taxminiy).
import pandas as pd

# Talab (1-qadam natijasi): yolov8s @ 640, 160 inferens/s
TALAB_TFLOPS = 28.6 * 160 / 1e3       # 4.58 TFLOP/s haqiqiy
KAMERA_OQIM = 16                      # 1080p10 oqimlar soni
GFLOP = 28.6

platformalar = pd.DataFrame([
    # nom, cho'qqi INT8 dense TOPS, samaradorlik, quvvat Vt, narx $, NVDEC 1080p30 oqim, INT8, FP16, sanoat harorati, ekotizim (0-5)
    ("Jetson Orin NX 16GB",  50,  0.45, 25,  650,   7, True,  True,  True,  5),
    ("Jetson AGX Orin 64GB", 138, 0.45, 60,  2000, 14, True,  True,  True,  5),
    ("A100 PCIe 80GB",       624, 0.50, 300, 10000, 10, True,  True,  False, 5),
    ("Coral Edge TPU M.2",   4,   0.50, 2,   80,    0, True,  False, False, 2),
    ("x86 CPU 8 yadro",      2.5, 0.60, 65,  400,   0, True,  True,  True,  4),
    ("Hailo-8 M.2",          26,  0.50, 3,   200,   0, True,  False, True,  3),
], columns=["nom", "tops_peak", "eff", "vt", "narx", "nvdec_1080p30", "int8", "fp16", "sanoat_t", "ekotizim"])

platformalar["tops_real"] = platformalar.tops_peak * platformalar.eff
platformalar["zaxira_koef"] = (platformalar.tops_real / TALAB_TFLOPS).round(2)
# 1080p10 oqim = 1/3 ta 1080p30 oqim
platformalar["dekod_yetarli"] = platformalar.nvdec_1080p30 * 3 >= KAMERA_OQIM
# bitta inferens uchun energiya (J) = quvvat / (sekundiga inferenslar soni)
platformalar["inferens_J"] = (platformalar.vt / (platformalar.tops_real * 1e3 / GFLOP)).round(4)

print("=== Platformalarni baholash ===")
print(platformalar[["nom", "tops_peak", "tops_real", "zaxira_koef", "dekod_yetarli", "vt", "narx", "inferens_J"]].to_markdown(index=False))

# --- Talablar matritsasi: (mezon, og'irlik, baholash funksiyasi -> 0..1) ---
# zaxira: 1 dan kichik = 0; 1-3 oralig'i chiziqli; 3 dan yuqori = 1
def ball_zaxira(z):
    return 0.0 if z < 1 else min(1.0, (z - 1) / 2)

MEZONLAR = [
    ("O'tkazuvchanlik (zaxira>=3)", 5, lambda r: ball_zaxira(r.zaxira_koef)),
    ("Kechikish (<=200 ms)",        5, lambda r: 1.0 if r.zaxira_koef >= 1 else 0.0),
    ("Ish harorati (+40 C)",        4, lambda r: 1.0 if r.sanoat_t else 0.0),
    ("Quvvat (<=60 Vt)",            3, lambda r: 1.0 if r.vt <= 60 else 0.0),
    ("Video dekodlash (16 oqim)",   4, lambda r: 1.0 if r.dekod_yetarli else 0.3),
    ("Narx (<=2000 $)",             3, lambda r: max(0.0, 1 - r.narx / 2000) if r.narx <= 2000 else 0.0),
    ("Dasturiy ekotizim",           3, lambda r: r.ekotizim / 5),
]
JAMI_OGIRLIK = sum(w for _, w, _ in MEZONLAR)

natija = []
for _, r in platformalar.iterrows():
    ballar = {m: round(f(r) * w, 2) for m, w, f in MEZONLAR}
    jami = round(sum(ballar.values()) / JAMI_OGIRLIK * 100, 1)
    # funksional talab (zaxira >= 1) bajarilmasa — platforma umuman yaroqsiz
    yaroqli = r.zaxira_koef >= 1
    natija.append({"nom": r.nom, **ballar, "jami_%": jami, "yaroqli": yaroqli})

jadval = pd.DataFrame(natija).sort_values(["yaroqli", "jami_%"], ascending=False)
print("\n--- Og'irlikli ballar (talablar matritsasi, jami 100 %) ---")
print(jadval.to_markdown(index=False))
tanlov = jadval[jadval.yaroqli].iloc[0]
print(f"\nTANLOV: {tanlov.nom} ({tanlov['jami_%']} %)")
print("Yaroqsiz (zaxira < 1):", ", ".join(jadval[~jadval.yaroqli].nom))
