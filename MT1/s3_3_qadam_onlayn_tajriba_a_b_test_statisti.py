"""MT1 — 3-qadam. Onlayn tajriba: A/B test, statistik quvvat va peeking xatosi

Offline metrika yakuniy hukm emas — qarorni onlayn A/B test beradi. Simulyatorda har foydalanuvchining «haqiqiy» didi ma'lum, shuning uchun ikki siyosatning (A — mashhurlik, B — xatoli shaxsiy model) haqiqiy CTR farqi oldindan bilinadi. Tajriba 6000 va 40 000 seans bilan o'tkaziladi: ikki proporsiya z-testi, 95% ishonch oralig'i, so'ngra MDE bo'yicha kerakli tanlanma hajmi hisoblanadi. A/A testda 20 marta oraliq natijaga qarash (peeking) I tur xatoni 5% dan ~25% gacha oshirishi va yangilik effekti ko'rsatiladi.

Kutilgan natija: Haqiqiy o'sish ~+11%; 6000 seansda farq ahamiyatsiz, 40 000 da p < 0,001; MDE 10% uchun har guruhga ~25 ming seans; peeking soxta g'alabani ~0,05 dan ~0,25 ga oshiradi (taxminiy).
"""
# 3-qadam. Onlayn tajriba: A/B test simulyatsiyasi (mashhurlik vs shaxsiy tavsiya), statistik quvvat va «peeking» xatosi
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
# --- «Haqiqiy dunyo» (berilgan, o'zgartirmang)
n_u, n_i, k = 600, 400, 6
P, Q = rng.normal(0, 1, (n_u, k)), rng.normal(0, 1, (n_i, k))
pop = 1.0 / np.arange(1, n_i + 1) ** 0.9; pop = pop[rng.permutation(n_i)]
lp = (np.log(pop) - np.log(pop).mean()) / np.log(pop).std()
bi = 0.35 * rng.normal(0, 1, n_i) + 0.35 * lp
haqiqiy = 3.5 + rng.normal(0, 0.35, n_u)[:, None] + bi[None, :] + P @ Q.T / np.sqrt(k) * 0.9
def p_click(baho):
    return 0.10 / (1 + np.exp(-1.5 * (baho - 4.0)))
topA = np.argsort(-pop)[:10]
taxmin = haqiqiy + rng.normal(0, 1.4, haqiqiy.shape)
topB = np.argsort(-taxmin, axis=1)[:, :10]

# TODO 1: har foydalanuvchi uchun A va B siyosatlarining kutilgan CTR ini (ctrA, ctrB) va haqiqiy nisbiy o'sishni hisoblang
ctrA = ctrB = None

print("\n=== A/B test natijasi ===")
for n in (6000, 40000):
    # TODO 2: n seansni simulyatsiya qiling, ikki proporsiya z-testi, p-qiymat va farq uchun 95% CI ni chiqaring
    raise NotImplementedError("TODO 2: z-test")

# TODO 3: MDE (nisbiy) uchun har guruhga kerakli seanslar soni (alfa 0,05, quvvat 0,8)
def kerakli_n(p0, mde_nisbiy, alfa=0.05, quvvat=0.8):
    raise NotImplementedError("TODO 3")
print("\n=== Statistik quvvat (alfa 0,05, quvvat 0,8) ===")

# TODO 4: A/A testda (p0=0,05, 2000 tajriba) har 500 seansda 20 marta qarash — soxta g'alaba ulushini toping
print("\n=== Peeking xatosi (A/A test, 2000 tajriba) ===")
# TODO 5: yangilik effekti — o'sish dastlabki kunlarda bo'rttirilgan; 1- va 14-kun o'sishini taqqoslang
