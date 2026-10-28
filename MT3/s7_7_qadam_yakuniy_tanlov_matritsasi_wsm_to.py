"""MT3 — 7-qadam. Yakuniy tanlov matritsasi: WSM, TOPSIS va vaznlar sezgirligi

Oldingi qadamlar natijalari 6 mezonli jadvalga yig'iladi (unumdorlik, kechikish, energiya, TCO, ekotizim, moslashuvchanlik) va qat'iy cheklovlar (quvvat, internetsiz ishlash) qo'llanadi. Uch ssenariy (data-markaz, chekka kamera, moliya) uchun vaznli yig'indi (WSM) va TOPSIS reytinglari hisoblanadi; Dirichlet taqsimoti bilan vaznlar 2000 marta tebratilib, 1-o'rin barqarorligi baholanadi. Natija issiqlik xaritasi ko'rinishidagi tanlov matritsasi.

Kutilgan natija: Data-markaz va moliyada GPU 1-o'rin (barqarorlik 80-98 %), chekkada Jetson/NPU — usullar kelishmasligi ko'rsatiladi; matritsa va grafik.
"""
# 7-qadam. Yakuniy tanlov matritsasi: ko'p mezonli qaror (WSM + TOPSIS) va vaznlar sezgirligi (Monte-Karlo)
# Oldingi qadamlar natijalari (taxminiy) mezonlar jadvaliga yig'iladi. 3 ta ssenariy uchun turli vaznlar bilan
# arxitekturalar reytingi tuziladi, qat'iy cheklovlar (quvvat, bulutsiz ishlash) filtr sifatida qo'llanadi.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
pd.set_option("display.width", 160)

# Mezonlar: (qiymat yo'nalishi: +1 — ko'pi yaxshi, -1 — kami yaxshi)
MEZON = {"unumdorlik": +1, "kechikish": -1, "energiya": -1, "TCO": -1, "ekotizim": +1, "moslashuvchanlik": +1}
# Qiymatlar: unumdorlik — tasvir/s/qurilma (6-qadam), kechikish — batch=1 ms (taxminiy), energiya — mJ/tasvir (6-qadam),
# TCO — sent / 1 mln tasvir (6-qadam), ekotizim va moslashuvchanlik — ekspert bahosi 1..5 (dasturlash qulayligi,
# kutubxonalar, yangi operatorlarni qo'llab-quvvatlash). O'z bahoingizni qo'ying!
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
    """Min-max normallash: 1 — eng yaxshi, 0 — eng yomon (yo'nalish hisobga olinadi). Log-shkala — katta diapazonlar uchun."""
    n = pd.DataFrame(index=df.index)
    for m, yo in MEZON.items():
        x = np.log10(df[m].astype(float)) if m in ("unumdorlik", "energiya", "TCO") else df[m].astype(float)
        z = (x - x.min()) / (x.max() - x.min() + 1e-12)
        n[m] = z if yo > 0 else 1 - z
    return n


def topsis(df, w):
    """TOPSIS: ideal va anti-ideal yechimgacha masofa nisbati."""
    X = df[list(MEZON)].astype(float).values
    X = X / np.sqrt((X ** 2).sum(0)); V = X * np.array([w[m] for m in MEZON])
    yo = np.array(list(MEZON.values()))
    ideal = np.where(yo > 0, V.max(0), V.min(0)); anti = np.where(yo > 0, V.min(0), V.max(0))
    dp = np.sqrt(((V - ideal) ** 2).sum(1)); dm = np.sqrt(((V - anti) ** 2).sum(1))
    return pd.Series(dm / (dp + dm), index=df.index)


issiq = {}
for nom, (w, cheklov) in SSENARIY.items():
    nomzod = JADVAL[JADVAL.apply(cheklov, axis=1)]
    N = normal(nomzod)
    wsm = (N * pd.Series(w)).sum(axis=1)
    tp = topsis(nomzod, w)
    # Monte-Karlo: vaznlarni Dirichlet bilan +-tebratamiz (konsentratsiya 50) — 1-o'rin barqarorligi
    birinchi = pd.Series(0, index=nomzod.index)
    W0 = np.array([w[m] for m in MEZON])
    for _ in range(2000):
        wv = rng.dirichlet(W0 * 50)
        birinchi.iloc[int((N.values * wv).sum(1).argmax())] += 1
    res = pd.DataFrame({"WSM": wsm, "TOPSIS": tp, "1-o'rin_%": 100 * birinchi / 2000}).sort_values("WSM", ascending=False)
    res["o'rin"] = range(1, len(res) + 1)
    print(f"\n=== Ssenariy: {nom} ===")
    print(f"Chiqarib tashlangan (cheklov): {', '.join(sorted(set(JADVAL.index) - set(nomzod.index))) or "yo'q"}")
    print(res.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"Tavsiya: {res.index[0]} (WSM), TOPSIS bo'yicha: {res.TOPSIS.idxmax()}")
    if res.index[0] != res.TOPSIS.idxmax():
        print("Usullar kelishmadi: nomzod kam bo'lsa min-max normallash har mezonni 0/1 ga aylantiradi — qaror chegarada.")
    issiq[nom.split(":")[0]] = wsm.reindex(JADVAL.index)

print("\n=== Yakuniy tanlov matritsasi (WSM, '—' = cheklov bo'yicha mos emas) ===")
M = pd.DataFrame(issiq)
print(M.to_string(float_format=lambda v: f"{v:.2f}", na_rep="—"))

fig, ax = plt.subplots(figsize=(7, 4))
im = ax.imshow(M.fillna(0).values, cmap="viridis", aspect="auto")
ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(M.columns, fontsize=8)
ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(M.index)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M.values[i, j]; ax.text(j, i, "—" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", color="w", fontsize=8)
ax.set_title("Tanlov matritsasi (WSM ball)"); plt.colorbar(im); plt.tight_layout()
print("\n=== Tanlov matritsasi tayyor ===")
