"""MT3 — 7-qadam. Yakuniy tanlov matritsasi: WSM, TOPSIS va vaznlar sezgirligi

Oldingi qadamlar natijalari 6 mezonli jadvalga yig'iladi (unumdorlik, kechikish, energiya, TCO, ekotizim, moslashuvchanlik) va qat'iy cheklovlar (quvvat, internetsiz ishlash) qo'llanadi. Uch ssenariy (data-markaz, chekka kamera, moliya) uchun vaznli yig'indi (WSM) va TOPSIS reytinglari hisoblanadi; Dirichlet taqsimoti bilan vaznlar 2000 marta tebratilib, 1-o'rin barqarorligi baholanadi. Natija issiqlik xaritasi ko'rinishidagi tanlov matritsasi.

Kutilgan natija: Data-markaz va moliyada GPU 1-o'rin (barqarorlik 80-98 %), chekkada Jetson/NPU — usullar kelishmasligi ko'rsatiladi; matritsa va grafik.
"""
# 7-qadam. Yakuniy tanlov matritsasi: ko'p mezonli qaror (WSM + TOPSIS) va vaznlar sezgirligi (Monte-Karlo)
# Oldingi qadamlar natijalari (taxminiy) mezonlar jadvaliga yig'iladi; 3 ssenariy uchun turli vaznlar bilan reyting.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
pd.set_option("display.width", 160)

# --- Berilgan ma'lumotlar (o'zgartirmang) ---
MEZON = {"unumdorlik": +1, "kechikish": -1, "energiya": -1, "TCO": -1, "ekotizim": +1, "moslashuvchanlik": +1}
JADVAL = pd.DataFrame({
    "unumdorlik":       [1080, 84000, 16800, 1590, 5050, 2440],
    "kechikish":        [6.0, 1.0, 1.5, 2.5, 2.0, 1.2],
    "energiya":         [473, 10.1, 11.9, 3.5, 11.9, 133],
    "TCO":              [18.7, 8.3, 17.9, 0.5, 0.5, 8.4],
    "ekotizim":         [5, 5, 4, 2, 4, 2],
    "moslashuvchanlik": [5, 4, 3, 2, 4, 5],
    "quvvat_vt":        [510, 850, 200, 5.5, 60, 325],
    "oflayn":           [1, 1, 0, 1, 1, 1],
}, index=["CPU", "GPU", "TPU", "NPU", "Jetson", "FPGA"])
print("=== Mezonlar jadvali (oldingi qadamlardan, taxminiy) ===")
print(JADVAL.to_string())

SSENARIY = {  # vaznlar yig'indisi 1; cheklov — lambda
    "Data-markaz: katta hajmli inferens": ({"unumdorlik": .30, "kechikish": .10, "energiya": .15, "TCO": .25, "ekotizim": .15, "moslashuvchanlik": .05}, lambda r: True),
    "Chekka kamera: <= 60 Vt, internetsiz":  ({"unumdorlik": .10, "kechikish": .20, "energiya": .35, "TCO": .15, "ekotizim": .10, "moslashuvchanlik": .10}, lambda r: r.quvvat_vt <= 60 and r.oflayn == 1),
    "Moliya: past kechikish, tez o'zgaruvchi model": ({"unumdorlik": .10, "kechikish": .35, "energiya": .05, "TCO": .10, "ekotizim": .15, "moslashuvchanlik": .25}, lambda r: r.oflayn == 1),
}


def normal(df):
    """Min-max normallash: 1 — eng yaxshi, 0 — eng yomon (yo'nalish hisobga olinadi)."""
    # TODO 1: har mezon uchun (unumdorlik/energiya/TCO — log10 shkalada) [0, 1] ga keltiring, -1 yo'nalishni teskari qiling
    raise NotImplementedError("TODO 1: normallash")


def topsis(df, w):
    """TOPSIS: ideal va anti-ideal yechimgacha masofa nisbati."""
    # TODO 2: vektor normallash, vaznlash, ideal/anti-ideal nuqta, masofalar va dm / (dp + dm)
    raise NotImplementedError("TODO 2: TOPSIS")


issiq = {}
for nom, (w, cheklov) in SSENARIY.items():
    # TODO 3: cheklov bo'yicha nomzodlarni filtrlang, WSM va TOPSIS ballarini hisoblang
    # TODO 4: Dirichlet (W0*50) bilan 2000 marta vaznlarni tebratib, har nomzodning 1-o'rin ulushini (%) toping
    # TODO 5: natijani WSM bo'yicha tartiblab chiqaring, WSM va TOPSIS tavsiyalari kelishganini tekshiring
    print(f"\n=== Ssenariy: {nom} ===")
    raise NotImplementedError("TODO 3-5: ssenariy reytingi")

print("\n=== Yakuniy tanlov matritsasi (WSM, '—' = cheklov bo'yicha mos emas) ===")
# TODO 6: issiq dan DataFrame tuzib chiqaring va imshow bilan issiqlik xaritasini chizing
print("\n=== Tanlov matritsasi tayyor ===")
