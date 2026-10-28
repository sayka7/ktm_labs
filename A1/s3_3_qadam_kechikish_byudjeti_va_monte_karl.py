"""A1 — 3-qadam. Kechikish byudjeti va Monte-Karlo simulyatsiyasi

200 ms umumiy byudjet bloklar (tarmoq+dekodlash, oldindan ishlov, inferens, keyingi ishlov) orasida taqsimlanadi; har blok kechikishi lognormal taqsimot bilan 20 000 marta simulyatsiya qilinib p50/p95/p99 topiladi. Paket (batch) hajmi inferensni tejasa-da, kutish vaqtini oshirishi ko'rsatiladi. Kutilgan natija: batch=1 da p95 ~105 ms, batch=32 da p95 200 ms dan oshadi. Grafikda byudjet vs simulyatsiya ustunlari va batch egri chizig'i chiqadi.

Kutilgan natija: Bloklar jadvali (p50/p95/p99), jami p95 ~105 ms, batch jadvali: 16 da ~150 ms, 32 da ~240 ms; 2 panelli grafik.
"""
# 3-qadam. Uchdan-uchga kechikish byudjeti va uning Monte-Karlo simulyatsiyasi
# 200 ms umumiy byudjet bloklar orasida taqsimlanadi; har blok kechikishi tasodifiy (lognormal)
# deb olinadi va p50/p95/p99 hisoblanadi. Paket (batch) hajmi kutish vaqtini qanday oshirishini ko'ramiz.
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
BYUDJET_MS = 200
INFERENS_S = 160                       # 16 kamera x 10 kadr/s
N = 20000                              # simulyatsiya qilinadigan kadrlar soni

# Blok: (nomi, byudjet ms, odatiy (median) ms, tarqalish sigma (lognormal))
BLOKLAR = [
    ("Tarmoq + dekodlash", 40, 25, 0.35),
    ("Oldindan ishlov",    10,  6, 0.25),
    ("Inferens",           60, 35, 0.20),
    ("Keyingi ishlov+mantiq", 20, 12, 0.40),
]
ZAXIRA = BYUDJET_MS - sum(b[1] for b in BLOKLAR)


def simulyatsiya(batch):
    """Har blok uchun N ta kechikish namunasini qaytaradi (ms)."""
    ustunlar = {}
    for nom, _, med, sigma in BLOKLAR:
        ustunlar[nom] = rng.lognormal(np.log(med), sigma, N)
    # Paketlash: inferens batch ta kadr to'planguncha kutadi; o'rtacha kutish (batch-1)/2 / INFERENS_S
    if batch > 1:
        ustunlar["Paket kutish"] = rng.uniform(0, (batch - 1) / INFERENS_S * 1000, N)
        # batch ko'paysa, bir kadrga to'g'ri keladigan inferens vaqti kamayadi (GPU samaradorligi oshadi)
        ustunlar["Inferens"] *= (1 + 0.25 * (batch - 1)) / batch
    return ustunlar


print("=== Kechikish byudjeti ===")
print(f"{'Blok':26s} {'byudjet':>8s} {'p50':>7s} {'p95':>7s} {'p99':>7s}")
jami = np.zeros(N)
p95_bloklar = []
for nom, byud, med, sigma in BLOKLAR:
    x = rng.lognormal(np.log(med), sigma, N)
    jami += x
    p95_bloklar.append(np.percentile(x, 95))
    print(f"{nom:26s} {byud:8d} {np.percentile(x, 50):7.1f} {np.percentile(x, 95):7.1f} {np.percentile(x, 99):7.1f}")
print(f"{'Zaxira':26s} {ZAXIRA:8d}")
print(f"{'JAMI (batch=1)':26s} {BYUDJET_MS:8d} {np.percentile(jami, 50):7.1f} {np.percentile(jami, 95):7.1f} {np.percentile(jami, 99):7.1f}")
print(f"Byudjetdan oshgan kadrlar ulushi: {np.mean(jami > BYUDJET_MS) * 100:.2f} %")

print("\n--- Paket hajmining ta'siri (uchdan-uchga) ---")
print(f"{'batch':>5s} {'p50':>7s} {'p95':>7s} {'p99':>7s} {'>200ms %':>9s}  inferens/kadr(ms)")
batchlar = (1, 4, 8, 16, 32)
p95_batch = []
for b in batchlar:
    u = simulyatsiya(b)
    e2e = sum(u.values())
    p95_batch.append(np.percentile(e2e, 95))
    print(f"{b:5d} {np.percentile(e2e, 50):7.1f} {np.percentile(e2e, 95):7.1f} {np.percentile(e2e, 99):7.1f} {np.mean(e2e > BYUDJET_MS) * 100:9.2f}  {np.median(u['Inferens']):.1f}")

# --- Grafik: byudjet vs simulyatsiya p95 (stacked bar) va batch ta'siri ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
nomlar = [b[0] for b in BLOKLAR]
byudjetlar = [b[1] for b in BLOKLAR]
pastki = 0
for nom, byud, p95 in zip(nomlar, byudjetlar, p95_bloklar):
    ax1.bar("Byudjet", byud, bottom=pastki, label=nom)
    pastki += byud
ax1.bar("Byudjet", ZAXIRA, bottom=pastki, color="lightgray", label="Zaxira")
pastki = 0
for p95 in p95_bloklar:
    ax1.bar("Simulyatsiya p95", p95, bottom=pastki)
    pastki += p95
ax1.axhline(BYUDJET_MS, color="red", ls="--", label="200 ms chegara")
ax1.set_ylabel("ms")
ax1.set_title("Kechikish byudjeti taqsimoti")
ax1.legend(fontsize=7)
ax2.plot(batchlar, p95_batch, "o-")
ax2.axhline(BYUDJET_MS, color="red", ls="--")
ax2.set_xlabel("Paket hajmi (batch)")
ax2.set_ylabel("Uchdan-uchga p95, ms")
ax2.set_title("Paketlash kechikishni qanday oshiradi")
ax2.grid(alpha=0.3)
fig.tight_layout()
print("\nXulosa: batch 4-8 inferensni tejaydi, lekin 16+ da p95 200 ms chegaraga yaqinlashadi.")
