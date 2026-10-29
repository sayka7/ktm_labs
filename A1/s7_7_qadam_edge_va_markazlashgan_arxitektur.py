"""A1 — 7-qadam. Edge va markazlashgan arxitekturaning 3 yillik TCO taqqoslovi

Uch variant (har sexda Orin NX; markaziy A100 server; markaziy L4 server) uchun apparat, elektr energiyasi (PUE bilan), tarmoq va texnik xizmat xarajatlari 3 yilga hisoblanadi; sexlar soni 1–24 oralig'ida tenglik nuqtasi qidiriladi va grafik chiziladi. Kutilgan natija: 16 kamera uchun Edge ~1,5 ming $, A100 ~26 ming $; 24 sexgacha markaz Edge'dan arzonlashmaydi. Elektr narxi, tarmoq ijara va server sig'imini o'zgartirib qaytadan hisoblang.

Kutilgan natija: 1/2/4/8 sex uchun TCO jadvallari ($/kamera/oy bilan), tenglik nuqtasi xabari, 3 chiziqli TCO grafigi.
"""
# 7-qadam. Chekka (Edge) va markazlashgan arxitekturalarning 3 yillik TCO taqqoslovi
# TCO = apparat + elektr energiyasi + sovutish + tarmoq + texnik xizmat.
import numpy as np
import matplotlib.pyplot as plt

YIL = 3
SOAT_YIL = 24 * 365
NARX_KVT_SOAT = 0.08          # $ / kVt*soat (o'zgartirib ko'ring)
SEXLAR = (1, 2, 4, 8)          # sexlar soni, har birida 16 kamera

# Berilgan ma'lumotlar: apparat $, quvvat Vt, PUE, tarmoq $/yil, xizmat $/yil
VARIANTLAR = {
    "Edge: Orin NX har sexda": dict(apparat=650 + 350, vt=25 + 15, pue=1.1, tarmoq=0, xizmat=150, birlik_har_sex=1),
    "Markaz: A100 server":     dict(apparat=10000 + 6000, vt=300 + 250, pue=1.6, tarmoq=1200, xizmat=1500, birlik_har_sex=0),
    "Markaz: L4 server":       dict(apparat=2500 + 4000, vt=72 + 200, pue=1.6, tarmoq=1200, xizmat=1000, birlik_har_sex=0),
}
SIGIM_SEX = {"Edge: Orin NX har sexda": 1, "Markaz: A100 server": 12, "Markaz: L4 server": 3}


def tco(nom, sexlar):
    """Lug'at: birlik, apparat, energiya, tarmoq, xizmat, jami ($, YIL yil uchun)."""
    v = VARIANTLAR[nom]
    # TODO 1: kerakli birliklar soni (Edge: har sexga 1; markaz: sig'im bo'yicha yuqoriga yaxlitlash)
    # TODO 2: energiya $ = birlik x Vt x PUE -> kVt x soat x narx x yil
    # TODO 3: tarmoq (faqat markaz, har sex uchun ijara) va xizmat xarajatlari, jami
    raise NotImplementedError("TODO 1-3: tco")


print("=== TCO 3 yil ===")
for s in SEXLAR:
    print(f"\n--- {s} sex ({s * 16} kamera) ---")
    # TODO 4: har variant uchun qatorni $/kamera/oy bilan chiqaring (Edge, Markaziy A100, L4)
    for nom in VARIANTLAR:
        r = tco(nom, s)

# TODO 5: 1..24 sex uchun tenglik nuqtasini (A100 < Edge) toping va 3 chiziqli grafik chizing
