"""MT1 — 5-qadam. Amazon uslubidagi ombor: ABC tahlili, slotting va zaxira siyosati

300 SKU uchun 120 kunlik sintetik talab bo'yicha ABC tahlili bajariladi. So'ngra 300 joyli omborda tovarlarni joylashtirish (slotting) uch usulda solishtiriladi: tasodifiy, talab bo'yicha saralash va Venger usuli (`linear_sum_assignment`); 3000 buyurtma bo'yicha o'rtacha terish yo'li o'lchanadi. A sinf tovarlari uchun xavfsizlik zaxirasi, qayta buyurtma nuqtasi (ROP) va EOQ hisoblanadi va (s, Q) siyosati barqaror hamda tasodifiy yetkazish muddati bilan simulyatsiya qilinadi.

Kutilgan natija: A sinf: ~17% SKU qiymatning 80% i; talab bo'yicha slotting terish yo'lini ~55% qisqartiradi; tasodifiy yetkazish muddatida fill rate ~1,0 dan ~0,93 ga tushadi (taxminiy). Slotting xaritasi.
"""
# 5-qadam. Amazon uslubidagi ombor: ABC tahlili, talab bo'yicha joylashtirish (slotting) va zaxira siyosati
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

rng = np.random.default_rng(11)
# --- 300 SKU, 120 kunlik sintetik talab (berilgan, o'zgartirmang)
n_sku, D = 300, 120
mashhur = 1.0 / np.arange(1, n_sku + 1) ** 1.1; mashhur = mashhur[rng.permutation(n_sku)]
kunlik_o = 4000 * mashhur / mashhur.sum()
hafta = 1 + 0.25 * np.sin(2 * np.pi * np.arange(D) / 7)
talab = rng.poisson(np.outer(hafta, kunlik_o))
narx = rng.lognormal(3.0, 0.8, n_sku)

# TODO 1: ABC tahlili — qiymat bo'yicha saralab kumulyativ ulush: A <= 80%, B <= 95%, C — qolgani; sinflar jadvali
sinf = None
print("=== ABC tahlili ===")

# --- Ombor geometriyasi (berilgan): 20 qator x 15 ustun, stantsiya (0,0) da, Manhattan masofa
xy = np.array([(r * 3.0, c * 2.0) for r in range(20) for c in range(15)])
masofa = np.abs(xy).sum(1)
tezlik = talab[:90].mean(0)
def yigish_masofasi(joy):
    p = talab[90:].sum(0) / talab[90:].sum(); jami = 0.0; r = np.random.default_rng(0)
    for _ in range(3000):
        sku = r.choice(n_sku, r.integers(1, 5), p=p)
        jami += 2 * masofa[joy[sku]].max()
    return jami / 3000

# TODO 2: uch joylashtirish: tasodifiy, talab bo'yicha saralash va Venger usuli; terish yo'li va tejashni chiqaring
print("\n=== Slotting: o'rtacha terish yo'li (m/buyurtma) ===")

# TODO 3: A sinfdan 8 SKU uchun xavfsizlik zaxirasi, ROP va EOQ (L=5, servis 95%, S=50, h=25%)
L, xizmat, S_buy, h = 5, 0.95, 50.0, 0.25
print("\n=== Zaxira siyosati (A sinfdan 8 SKU) ===")
raise NotImplementedError("TODO 3: ss, ROP, EOQ")

# TODO 4: (s, Q) siyosatini oxirgi 30 kunda barqaror (5 kun) va tasodifiy (3–9 kun) yetkazish bilan simulyatsiya qiling; fill rate
def simulyatsiya(lt_tasodifiy, seed=0):
    ...
