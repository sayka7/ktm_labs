"""A1 — 3-qadam. Kechikish byudjeti va Monte-Karlo simulyatsiyasi

200 ms umumiy byudjet bloklar (tarmoq+dekodlash, oldindan ishlov, inferens, keyingi ishlov) orasida taqsimlanadi; har blok kechikishi lognormal taqsimot bilan 20 000 marta simulyatsiya qilinib p50/p95/p99 topiladi. Paket (batch) hajmi inferensni tejasa-da, kutish vaqtini oshirishi ko'rsatiladi. Kutilgan natija: batch=1 da p95 ~105 ms, batch=32 da p95 200 ms dan oshadi. Grafikda byudjet vs simulyatsiya ustunlari va batch egri chizig'i chiqadi.

Kutilgan natija: Bloklar jadvali (p50/p95/p99), jami p95 ~105 ms, batch jadvali: 16 da ~150 ms, 32 da ~240 ms; 2 panelli grafik.
"""
# 3-qadam. Uchdan-uchga kechikish byudjeti va uning Monte-Karlo simulyatsiyasi
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
BYUDJET_MS = 200
INFERENS_S = 160                       # 16 kamera x 10 kadr/s
N = 20000                              # simulyatsiya qilinadigan kadrlar soni

# Berilgan ma'lumotlar: (nomi, byudjet ms, odatiy (median) ms, tarqalish sigma (lognormal))
BLOKLAR = [
    ("Tarmoq + dekodlash", 40, 25, 0.35),
    ("Oldindan ishlov",    10,  6, 0.25),
    ("Inferens",           60, 35, 0.20),
    ("Keyingi ishlov+mantiq", 20, 12, 0.40),
]
ZAXIRA = BYUDJET_MS - sum(b[1] for b in BLOKLAR)


def simulyatsiya(batch):
    """Har blok uchun N ta kechikish namunasini (ms) lug'at ko'rinishida qaytaradi."""
    # TODO 1: har blok uchun median va sigma bo'yicha lognormal namunalar yarating
    # TODO 3: batch > 1 da "Paket kutish" blokini qo'shing va inferens/kadr vaqtini kamaytiring
    raise NotImplementedError("TODO 1, 3: simulyatsiya")


print("=== Kechikish byudjeti ===")
print(f"{'Blok':26s} {'byudjet':>8s} {'p50':>7s} {'p95':>7s} {'p99':>7s}")
# TODO 2: har blok uchun p50/p95/p99 ni, so'ng JAMI (batch=1) va 200 ms dan oshgan ulushni chiqaring
p95_bloklar = []

print("\n--- Paket hajmining ta'siri (uchdan-uchga) ---")
batchlar = (1, 4, 8, 16, 32)
# TODO 4: har batch uchun uchdan-uchga p50/p95/p99 va >200 ms ulushini hisoblang
p95_batch = []
for b in batchlar:
    u = simulyatsiya(b)

# TODO 5: 2 panelli grafik: (1) byudjet vs simulyatsiya p95 ustunlari, (2) batch -> p95 egri chizig'i
