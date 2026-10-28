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
n_u, n_i, W = 500, 400, 14                                         # foydalanuvchi, film, hafta
bu0, bi0 = rng.normal(0, 0.35, n_u), rng.normal(0, 0.5, n_i)
pop = 1.0 / np.arange(1, n_i + 1) ** 0.9; pop = pop[rng.permutation(n_i)]
yangi = np.arange(320, 400)                                         # 8-haftadan katalogga qo'shiladigan 80 ta yangi film

def hafta_mal(w, n=1500):
    """w-hafta reytinglari. 8-haftadan: yangi filmlar chiqadi, yosh auditoriya kelib baholar qattiqroq bo'ladi (drift)."""
    p = pop.copy()
    if w < 8: p[yangi] = 0
    else: p[yangi] *= 6                                             # yangi filmlarga ommaviy qiziqish
    p /= p.sum()
    u, i = rng.integers(0, n_u, n), rng.choice(n_i, n, p=p)
    siljish = -0.45 * (w >= 8) * (u % 2 == 0)                        # foydalanuvchilarning yarmi qattiqroq baholay boshlaydi
    bi = bi0.copy(); bi[yangi] += 0.4                                 # yangi filmlar o'rtacha yaxshiroq
    r = np.clip(np.round(3.5 + bu0[u] + siljish + bi[i] + rng.normal(0, 0.6, n)), 1, 5)
    return pd.DataFrame(dict(w=w, user=u, item=i, rating=r))

data = pd.concat([hafta_mal(w) for w in range(1, W + 1)], ignore_index=True)

def bias_oqit(d, lam=5.0):
    mu = d.rating.mean(); bu, bi = np.zeros(n_u), np.zeros(n_i)
    for _ in range(8):
        g = d.assign(e=d.rating - mu - bu[d.user]).groupby("item").e; bi[g.sum().index] = g.sum() / (g.count() + lam)
        g = d.assign(e=d.rating - mu - bi[d.item]).groupby("user").e; bu[g.sum().index] = g.sum() / (g.count() + lam)
    return lambda d2: np.clip(mu + bu[d2.user] + bi[d2.item], 1, 5)

def psi(ref, cur, bins):
    """Population Stability Index: sum (c - r) * ln(c / r); < 0,1 barqaror, 0,1–0,2 kuzatish, > 0,2 drift."""
    r = np.histogram(ref, bins)[0] / len(ref) + 1e-4; c = np.histogram(cur, bins)[0] / len(cur) + 1e-4
    return float(np.sum((c - r) * np.log(c / r)))

ref = data[data.w <= 4]
model = bias_oqit(ref); model_ver, qayta_hafta = "v1", None
ref_pred = model(ref)
rows = []
for w in range(5, W + 1):
    cur = data[data.w == w]; pr = model(cur)
    rmse = float(np.sqrt(np.mean((cur.rating - pr) ** 2)))
    p_rat = psi(ref.rating, cur.rating, np.arange(0.5, 6))           # baholar taqsimoti drifti
    p_itm = psi(ref.item, cur.item, np.arange(0, n_i + 1, 20))        # film (kontent) taqsimoti drifti
    ks = stats.ks_2samp(ref_pred, pr).pvalue                          # bashoratlar taqsimoti drifti
    sovuq = float(np.isin(cur.item, ref.item.unique(), invert=True).mean())
    etalon = rows[0][2] if rows else rmse                             # 5-hafta RMSE — etalon
    signal = "DRIFT" if (p_itm > 0.2 or rmse > 1.10 * etalon) else "ok"
    rows.append((w, model_ver, rmse, p_rat, p_itm, ks, sovuq, signal))
    if signal == "DRIFT" and qayta_hafta is None:
        qayta_hafta = w                                               # ogohlantirish -> yangi (drift'dan keyingi) hafta bilan qayta o'qitish
        yangi_mal = data[data.w == w]                                 # real tizimda: avtomatik pipeline + validatsiya + kanareyka
        model = bias_oqit(pd.concat([ref.sample(frac=0.3, random_state=0), yangi_mal])); model_ver = "v2"
        ref = yangi_mal; ref_pred = model(ref)                        # monitoring etaloni ham yangilanadi

print("=== Monitoring paneli (haftalik) ===")
jad = pd.DataFrame(rows, columns=["hafta", "model", "RMSE", "PSI_baho", "PSI_film", "KS_p", "sovuq_start", "signal"])
print(jad.round(3).to_string(index=False))
print(f"\n=== Qayta o'qitish ===\nDrift {qayta_hafta}-haftada aniqlandi -> v2 (yangi hafta + eski ma'lumotning 30%) o'qitildi, etalon yangilandi.")
oldin = jad[(jad.hafta >= 8) & (jad.hafta <= qayta_hafta)].RMSE.mean(); keyin = jad[jad.hafta > qayta_hafta].RMSE.mean()
print(f"Drift davrida RMSE (v1): {oldin:.3f} | qayta o'qitilgandan keyin (v2): {keyin:.3f} | drift oldidan: {jad[jad.hafta < 8].RMSE.mean():.3f}")
print("Qoidalar: PSI_film > 0,2 yoki RMSE etalondan 10% yomon -> ogohlantirish; KS p < 0,01 — bashorat taqsimoti o'zgargan.")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
ax[0].plot(jad.hafta, jad.RMSE, marker="o"); ax[0].axvline(8, ls=":", c="gray", label="drift boshlanishi")
ax[0].axvline(qayta_hafta + 0.5, ls="--", c="tab:green", label="qayta o'qitish"); ax[0].set_title("Haftalik RMSE"); ax[0].legend()
ax[1].bar(jad.hafta - 0.2, jad.PSI_baho, 0.4, label="PSI baho"); ax[1].bar(jad.hafta + 0.2, jad.PSI_film, 0.4, label="PSI film")
ax[1].axhline(0.2, ls="--", c="tab:red"); ax[1].set_title("Ma'lumot drifti (PSI)"); ax[1].legend(); plt.tight_layout()
