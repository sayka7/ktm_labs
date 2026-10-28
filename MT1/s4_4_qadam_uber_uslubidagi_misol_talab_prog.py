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
# --- Sintetik talab: 10 hafta x 24 soat; kunlik cho'qqilar (8:00, 18:00), juma/shanba kechasi, yomg'ir, tadbirlar
H = 24 * 7 * 10
t = np.arange(H); soat, kun = t % 24, (t // 24) % 7
kunlik = 1 + 0.9 * np.exp(-((soat - 8) ** 2) / 4) + 1.1 * np.exp(-((soat - 18) ** 2) / 6) - 0.6 * ((soat >= 1) & (soat <= 5))
dam = 1 + 0.5 * ((kun >= 4) & ((soat >= 21) | (soat <= 2)))              # juma-shanba kechasi
yomgir = (rng.random(H // 6) < 0.15).repeat(6)[:H].astype(float)          # 6 soatlik yomg'ir bloklari
tadbir = np.zeros(H); tadbir[rng.choice(H, 12, replace=False)] = 1; tadbir = np.convolve(tadbir, np.ones(3), "same").clip(0, 1)
trend = 1 + 0.002 * t / 24                                                  # sekin o'sish (xizmat ommalashmoqda)
lam = 120 * kunlik * dam * trend * (1 + 0.35 * yomgir) * (1 + 0.8 * tadbir)
talab = rng.poisson(lam)
df = pd.DataFrame(dict(talab=talab, soat=soat, kun=kun, yomgir=yomgir, tadbir=tadbir))
df["lag24"], df["lag168"] = df.talab.shift(24), df.talab.shift(168)
df["sin_s"], df["cos_s"] = np.sin(2 * np.pi * soat / 24), np.cos(2 * np.pi * soat / 24)
df = df.dropna().reset_index(drop=True)
tr, te = df.iloc[:-168], df.iloc[-168:]                                     # oxirgi hafta — test
X = ["soat", "kun", "yomgir", "tadbir", "lag24", "lag168", "sin_s", "cos_s"]

def mape(y, p): return float(np.mean(np.abs(y - p) / np.maximum(y, 1)) * 100)
def mae(y, p): return float(np.mean(np.abs(y - p)))
pr = {"Mavsumiy naiv (o'tgan hafta)": te.lag168.values}
oh = lambda d: pd.get_dummies(d[X].astype({"soat": "category", "kun": "category"}), columns=["soat", "kun"])
pr["Ridge (one-hot soat/kun)"] = Ridge(1.0).fit(oh(tr), tr.talab).predict(oh(te).reindex(columns=oh(tr).columns, fill_value=0))
gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(tr[X], tr.talab)
pr["Gradient boosting"] = gb.predict(te[X])
print("=== Talab prognozi (oxirgi hafta, 168 soat) ===")
print(pd.DataFrame([(k, mae(te.talab.values, v), mape(te.talab.values, v)) for k, v in pr.items()],
                   columns=["Model", "MAE (safar/soat)", "MAPE, %"]).round(2).to_string(index=False))

# --- Dinamik narx: haydovchilar taklifi cheklangan; surge = talab/taklif nisbatiga ko'ra (1,0–2,5)
taklif = np.full(168, 190.0)                                               # bir vaqtda faol haydovchilar quvvati (safar/soat)
def surge(prognoz): return np.clip(1 + 0.6 * (prognoz / taklif - 1), 1.0, 2.5)
def bozor(koef):
    talab_narx = te.talab.values * koef ** -0.7                           # narx elastikligi -0,7 (taxminiy)
    bajarildi = np.minimum(talab_narx, taklif * (1 + 0.25 * (koef - 1)))  # yuqori narx ko'proq haydovchini chiqaradi
    return bajarildi.sum(), (talab_narx - bajarildi).clip(0).sum(), (bajarildi * 30000 * koef).sum() / 1e6
print("\n=== Dinamik narx (surge) ssenariylari ===")
rows = []
for nom, k in [("Surge yo'q", np.ones(168)), ("Surge: naiv prognoz", surge(pr["Mavsumiy naiv (o'tgan hafta)"])),
               ("Surge: boosting prognozi", surge(pr["Gradient boosting"])), ("Surge: ideal (haqiqiy talab)", surge(te.talab.values))]:
    b, q, d = bozor(k); rows.append((nom, b, q, d, k.max()))
print(pd.DataFrame(rows, columns=["Ssenariy", "Bajarilgan safar", "Qondirilmagan", "Tushum, mln so'm", "Maks. koef"]).round(1).to_string(index=False))
print("Izoh: prognoz qanchalik aniq bo'lsa, surge shunchalik o'z vaqtida yoqiladi; xato prognoz yo'lovchini ortiqcha to'latadi yoki talabni qondirmaydi.")

# --- Haydovchilar tahlili: 400 sintetik haydovchi, KMeans bilan segmentlash
n = 400
hd = pd.DataFrame(dict(soat_hafta=rng.gamma(4, 9, n), qabul=np.clip(rng.beta(8, 2, n), 0, 1),
                       bekor=np.clip(rng.beta(1.5, 20, n), 0, 1), reyting=np.clip(rng.normal(4.75, 0.15, n), 3.5, 5)))
hd.loc[:60, ["qabul", "bekor"]] = np.c_[rng.uniform(0.3, 0.6, 61), rng.uniform(0.12, 0.3, 61)]   # muammoli guruh
km = KMeans(3, n_init=10, random_state=0).fit(StandardScaler().fit_transform(hd))
hd["segment"] = km.labels_
print("\n=== Haydovchilar segmentlari (KMeans, k=3) ===")
print(hd.groupby("segment").agg(soni=("reyting", "size"), soat_hafta=("soat_hafta", "mean"), qabul=("qabul", "mean"),
                                bekor=("bekor", "mean"), reyting=("reyting", "mean")).round(3).to_string())
print("Amaliy xulosa: bekor qilish ulushi yuqori segmentga o'qitish/rag'bat; ko'p soat ishlaydiganlarga charchoq nazorati.")

plt.figure(figsize=(9, 3.2)); plt.plot(te.talab.values, label="haqiqiy", c="k")
plt.plot(pr["Gradient boosting"], label="boosting", c="tab:orange"); plt.plot(pr["Mavsumiy naiv (o'tgan hafta)"], label="naiv", c="tab:blue", alpha=0.6)
plt.axhline(190, ls="--", c="tab:red", label="haydovchi quvvati"); plt.xlabel("Soat (test haftasi)"); plt.ylabel("Safar/soat"); plt.legend(ncol=4); plt.tight_layout()
