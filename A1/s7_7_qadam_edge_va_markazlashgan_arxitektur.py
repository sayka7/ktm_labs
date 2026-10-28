"""A1 — 7-qadam. Edge va markazlashgan arxitekturaning 3 yillik TCO taqqoslovi

Uch variant (har sexda Orin NX; markaziy A100 server; markaziy L4 server) uchun apparat, elektr energiyasi (PUE bilan), tarmoq va texnik xizmat xarajatlari 3 yilga hisoblanadi; sexlar soni 1–24 oralig'ida tenglik nuqtasi qidiriladi va grafik chiziladi. Kutilgan natija: 16 kamera uchun Edge ~1,5 ming $, A100 ~26 ming $; 24 sexgacha markaz Edge'dan arzonlashmaydi. Elektr narxi, tarmoq ijara va server sig'imini o'zgartirib qaytadan hisoblang.

Kutilgan natija: 1/2/4/8 sex uchun TCO jadvallari ($/kamera/oy bilan), tenglik nuqtasi xabari, 3 chiziqli TCO grafigi.
"""
# 7-qadam. Chekka (Edge) va markazlashgan arxitekturalarning 3 yillik TCO taqqoslovi
# TCO = apparat + elektr energiyasi + sovutish + tarmoq + texnik xizmat. Ikki ssenariy:
# (A) har sexda Jetson Orin NX (16 kamera uchun 1 modul), (B) markaziy serverda bitta A100 (yoki L4).
import numpy as np
import matplotlib.pyplot as plt

YIL = 3
SOAT_YIL = 24 * 365
NARX_KVT_SOAT = 0.08          # $ / kVt*soat (o'zgartirib ko'ring)
SEXLAR = (1, 2, 4, 8)          # sexlar soni, har birida 16 kamera

# Variant: nom, apparat $ (bitta birlik), quvvat Vt (bitta birlik), sovutish koef, tarmoq $/yil, xizmat $/yil
VARIANTLAR = {
    "Edge: Orin NX har sexda": dict(apparat=650 + 350, vt=25 + 15, pue=1.1, tarmoq=0, xizmat=150, birlik_har_sex=1),
    "Markaz: A100 server":     dict(apparat=10000 + 6000, vt=300 + 250, pue=1.6, tarmoq=1200, xizmat=1500, birlik_har_sex=0),
    "Markaz: L4 server":       dict(apparat=2500 + 4000, vt=72 + 200, pue=1.6, tarmoq=1200, xizmat=1000, birlik_har_sex=0),
}
# Markaz serveri sig'imi: A100 ~ 68x zaxira (1 sex), L4 (121 TOPS INT8 dense) ~ 13x zaxira
SIGIM_SEX = {"Edge: Orin NX har sexda": 1, "Markaz: A100 server": 12, "Markaz: L4 server": 3}


def tco(nom, sexlar):
    v = VARIANTLAR[nom]
    birlik = sexlar if v["birlik_har_sex"] else int(np.ceil(sexlar / SIGIM_SEX[nom]))
    apparat = birlik * v["apparat"]
    energiya = birlik * v["vt"] * v["pue"] / 1000 * SOAT_YIL * NARX_KVT_SOAT * YIL
    tarmoq = v["tarmoq"] * YIL * (sexlar if not v["birlik_har_sex"] else 0)
    xizmat = birlik * v["xizmat"] * YIL
    return dict(birlik=birlik, apparat=apparat, energiya=round(energiya), tarmoq=tarmoq, xizmat=xizmat,
                jami=round(apparat + energiya + tarmoq + xizmat))


print("=== TCO 3 yil ===")
print(f"Elektr narxi {NARX_KVT_SOAT} $/kVt*soat, {YIL} yil, har sexda 16 kamera")
natija = {n: [] for n in VARIANTLAR}
for s in SEXLAR:
    print(f"\n--- {s} sex ({s * 16} kamera) ---")
    print(f"{'Variant':26s} {'birlik':>6s} {'apparat':>8s} {'energiya':>8s} {'tarmoq':>7s} {'xizmat':>7s} {'JAMI $':>8s} {'$/kamera/oy':>11s}")
    for nom in VARIANTLAR:
        r = tco(nom, s)
        natija[nom].append(r["jami"])
        print(f"{nom:26s} {r['birlik']:6d} {r['apparat']:8d} {r['energiya']:8d} {r['tarmoq']:7d} {r['xizmat']:7d} {r['jami']:8d} {r['jami'] / (s * 16) / (YIL * 12):11.2f}")

# Tenglik nuqtasi: markaz A100 qachon Edge'dan arzon bo'ladi?
sex_x = np.arange(1, 25)
edge = np.array([tco("Edge: Orin NX har sexda", s)["jami"] for s in sex_x])
a100 = np.array([tco("Markaz: A100 server", s)["jami"] for s in sex_x])
l4 = np.array([tco("Markaz: L4 server", s)["jami"] for s in sex_x])
teng = sex_x[np.argmax(a100 < edge)] if (a100 < edge).any() else None
teng_matn = str(teng) if teng else "24 sexgacha yo'q"
print(f"\nMarkaziy A100 Edge'dan arzon bo'ladigan sexlar soni: {teng_matn}")
print("Eslatma: TCO'da kechikish va yagona nosozlik nuqtasi (markaz) hisobga olinmagan — ular talablar matritsasida.")

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(sex_x, edge / 1e3, "o-", label="Edge: Orin NX har sexda")
ax.plot(sex_x, a100 / 1e3, "s-", label="Markaz: A100 server")
ax.plot(sex_x, l4 / 1e3, "^-", label="Markaz: L4 server")
ax.set_xlabel("Sexlar soni (har birida 16 kamera)")
ax.set_ylabel("3 yillik TCO, ming $")
ax.set_title("Edge va markazlashgan arxitektura TCO")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
