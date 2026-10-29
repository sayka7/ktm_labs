"""MT1 — 6-qadam. Model monitoringi: drift (PSI, KS), haftalik sifat va qayta o'qitish

14 haftalik reyting oqimi simulyatsiya qilinadi: 8-haftadan katalogga yangi filmlar qo'shiladi va auditoriyaning yarmi qattiqroq baholay boshlaydi. Monitoring panelida har hafta RMSE, baholar va film taqsimoti uchun PSI, bashoratlar uchun KS testi va sovuq start ulushi hisoblanadi. Ogohlantirish qoidasi (PSI > 0,2 yoki RMSE +10%) ishga tushganda model qayta o'qitiladi (yangi hafta + eski ma'lumotning 30%) va monitoring etaloni yangilanadi.

Kutilgan natija: 8-haftada PSI_film ~5 va RMSE ~0,70 dan ~0,84 ga oshadi — DRIFT; qayta o'qitilgan v2 da RMSE ~0,74 ga tushadi (taxminiy). RMSE va PSI grafiklari.
"""
# 6-qadam. Model monitoringi: ma'lumot drifti (PSI, KS), sifat drifti (haftalik RMSE), ogohlantirish va qayta o'qitish
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

rng = np.random.default_rng(5)
n_u, n_i, W = 500, 400, 14
bu0, bi0 = rng.normal(0, 0.35, n_u), rng.normal(0, 0.5, n_i)
pop = 1.0 / np.arange(1, n_i + 1) ** 0.9; pop = pop[rng.permutation(n_i)]
yangi = np.arange(320, 400)

def hafta_mal(w, n=1500):
    """w-hafta reytinglari; 8-haftadan drift (berilgan, o'zgartirmang)."""
    p = pop.copy()
    if w < 8: p[yangi] = 0
    else: p[yangi] *= 6
    p /= p.sum()
    u, i = rng.integers(0, n_u, n), rng.choice(n_i, n, p=p)
    siljish = -0.45 * (w >= 8) * (u % 2 == 0)
    bi = bi0.copy(); bi[yangi] += 0.4
    r = np.clip(np.round(3.5 + bu0[u] + siljish + bi[i] + rng.normal(0, 0.6, n)), 1, 5)
    return pd.DataFrame(dict(w=w, user=u, item=i, rating=r))

data = pd.concat([hafta_mal(w) for w in range(1, W + 1)], ignore_index=True)

def bias_oqit(d, lam=5.0):
    mu = d.rating.mean(); bu, bi = np.zeros(n_u), np.zeros(n_i)
    for _ in range(8):
        g = d.assign(e=d.rating - mu - bu[d.user]).groupby("item").e; bi[g.sum().index] = g.sum() / (g.count() + lam)
        g = d.assign(e=d.rating - mu - bi[d.item]).groupby("user").e; bu[g.sum().index] = g.sum() / (g.count() + lam)
    return lambda d2: np.clip(mu + bu[d2.user] + bi[d2.item], 1, 5)

# TODO 1: PSI(ref, cur, bins) — ikki gistogramma ulushlari bo'yicha
def psi(ref, cur, bins):
    raise NotImplementedError("TODO 1: PSI")

ref = data[data.w <= 4]
model = bias_oqit(ref); model_ver, qayta_hafta = "v1", None
rows = []
for w in range(5, W + 1):
    cur = data[data.w == w]
    # TODO 2: RMSE, PSI_baho, PSI_film, KS p-qiymati (bashoratlar), sovuq start ulushi
    # TODO 3: signal = "DRIFT" agar PSI_film > 0,2 yoki RMSE etalondan 10% yomon
    # TODO 4: birinchi DRIFT da qayta o'qiting (yangi hafta + eski ma'lumotning 30%) va etalonni yangilang
    raise NotImplementedError("TODO 2-4: monitoring tsikli")

print("=== Monitoring paneli (haftalik) ===")
print("\n=== Qayta o'qitish ===")
