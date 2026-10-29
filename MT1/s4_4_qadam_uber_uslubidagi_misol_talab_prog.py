"""MT1 — 4-qadam. Uber uslubidagi misol: talab prognozi, dinamik narx va haydovchilar tahlili

10 haftalik soatlik sintetik safar talabi (kunlik cho'qqilar, juma-shanba kechasi, yomg'ir, tadbirlar, trend) yaratiladi. Uch model — mavsumiy naiv (o'tgan hafta), Ridge (one-hot soat/kun) va gradient boosting — oxirgi haftada MAE/MAPE bilan solishtiriladi. Prognoz asosida dinamik narx (surge) qoidasi qo'llanadi va narx elastikligi bilan bajarilgan/qondirilmagan safarlar va tushum hisoblanadi. Oxirida 400 sintetik haydovchi KMeans bilan segmentlanadi (ish soati, qabul qilish, bekor qilish, reyting).

Kutilgan natija: MAPE: naiv ~14%, Ridge ~10%, boosting ~8%; surge qondirilmagan talabni ~4700 dan ~2300 gacha kamaytiradi; uchinchi segment — bekor qilish ~20% bo'lgan muammoli haydovchilar (taxminiy).
"""
# 4-qadam. Uber uslubidagi misol: soatlik safar talabi prognozi, dinamik narx (surge) va haydovchilar tahlili
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

rng = np.random.default_rng(3)
# --- Sintetik talab (berilgan, o'zgartirmang)
H = 24 * 7 * 10
t = np.arange(H); soat, kun = t % 24, (t // 24) % 7
kunlik = 1 + 0.9 * np.exp(-((soat - 8) ** 2) / 4) + 1.1 * np.exp(-((soat - 18) ** 2) / 6) - 0.6 * ((soat >= 1) & (soat <= 5))
dam = 1 + 0.5 * ((kun >= 4) & ((soat >= 21) | (soat <= 2)))
yomgir = (rng.random(H // 6) < 0.15).repeat(6)[:H].astype(float)
tadbir = np.zeros(H); tadbir[rng.choice(H, 12, replace=False)] = 1; tadbir = np.convolve(tadbir, np.ones(3), "same").clip(0, 1)
trend = 1 + 0.002 * t / 24
talab = rng.poisson(120 * kunlik * dam * trend * (1 + 0.35 * yomgir) * (1 + 0.8 * tadbir))
df = pd.DataFrame(dict(talab=talab, soat=soat, kun=kun, yomgir=yomgir, tadbir=tadbir))

# TODO 1: lag24, lag168, sin/cos soat belgilarini qo'shing; oxirgi 168 soat — test
X = ["soat", "kun", "yomgir", "tadbir", "lag24", "lag168", "sin_s", "cos_s"]
tr = te = None

# TODO 2: mavsumiy naiv, Ridge (one-hot soat/kun) va gradient boosting prognozlari; MAE va MAPE jadvali
def mape(y, p): ...
pr = {}
print("=== Talab prognozi (oxirgi hafta, 168 soat) ===")
raise NotImplementedError("TODO 2: uch modelni o'qitib solishtiring")

# TODO 3: surge koeffitsienti (talab/taklif, 1,0–2,5) va bozor modeli (elastiklik -0,7): bajarilgan, qondirilmagan, tushum
taklif = np.full(168, 190.0)
print("\n=== Dinamik narx (surge) ssenariylari ===")

# TODO 4: 400 sintetik haydovchini standartlashtirib KMeans (k=3) bilan segmentlang va segment profillarini chiqaring
print("\n=== Haydovchilar segmentlari (KMeans, k=3) ===")
