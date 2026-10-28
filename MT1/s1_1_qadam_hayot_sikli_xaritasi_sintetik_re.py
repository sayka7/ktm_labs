"""MT1 — 1-qadam. Hayot sikli xaritasi, sintetik reytinglar va bazaviy modellar

AI mahsulot hayot siklining 8 bosqichi va ularni ushbu ishning qaysi qadami qamrashi jadvalda ko'rsatiladi. So'ngra Netflix uslubidagi sintetik reyting matritsasi yaratiladi (600 foydalanuvchi, 400 film, yashirin didlar, Zipf mashhurlik) va EDA o'tkaziladi: siyraklik, uzun dum, sovuq start xavfi. Ma'lumot **vaqt bo'yicha** bo'linadi (tasodifiy bo'lish kelajakdan ma'lumot sizdiradi) va ikki bazaviy model — global o'rtacha va regulyarizatsiyalangan bias-model — RMSE bo'yicha solishtiriladi. O'zgartirib ko'ring: `lambda`, mashhurlik darajasi 0,9, faollik taqsimoti.

Kutilgan natija: Siyraklik ~93%, eng mashhur 10% film reytinglarning ~40–45% ini oladi; RMSE: global o'rtacha ~1,09, bias-model ~0,95 (taxminiy). Ikki grafik: uzun dum va faollik.
"""
# 1-qadam. AI mahsulot hayot sikli: muammo ta'rifi, sintetik Netflix uslubidagi reytinglar, EDA va bazaviy modellar
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# --- Hayot sikli bosqichlari va ushbu mustaqil ishning qaysi qadami ularni qamrab olishi
bosqichlar = [
    ("1. Muammo va biznes maqsadi", "tomosha vaqti / qaytib kelish (retention) ni oshirish", "s1"),
    ("2. Ma'lumot yig'ish va EDA", "reytinglar, siyraklik, uzun dum, sovuq start", "s1"),
    ("3. Modellash", "bias-model, item-kNN, matrix factorization", "s1-s2"),
    ("4. Offline baholash", "RMSE, precision@k, recall@k, coverage", "s2"),
    ("5. Onlayn tajriba", "A/B test, statistik quvvat", "s3"),
    ("6. Boshqa sohalar", "Uber talab/narx, Amazon ombor", "s4-s5"),
    ("7. Monitoring va drift", "PSI, KS, sirpanuvchi RMSE, qayta o'qitish", "s6"),
    ("8. Joylash va hujjat", "API xizmat, kanareyka, model kartasi", "s7"),
]
print("=== Hayot sikli bosqichlari ===")
print(pd.DataFrame(bosqichlar, columns=["Bosqich", "Mazmun", "Qadam"]).to_string(index=False))

# --- Sintetik reyting generatori: yashirin omillar (did, janr) + foydalanuvchi/film siljishi + mashhurlik (Zipf)
def reyting_yarat(n_u=600, n_i=400, k=6, seed=42):
    r = np.random.default_rng(seed)
    P, Q = r.normal(0, 1, (n_u, k)), r.normal(0, 1, (n_i, k))     # haqiqiy didlar (biz ularni bilmaymiz)
    bu, bi = r.normal(0, 0.35, n_u), r.normal(0, 0.5, n_i)           # qattiqqo'l/yumshoq tomoshabin, yaxshi/yomon film
    pop = 1.0 / np.arange(1, n_i + 1) ** 0.9; pop = pop[r.permutation(n_i)]; pop /= pop.sum()
    faollik = np.clip(r.lognormal(3.0, 0.7, n_u).astype(int) + 5, 5, n_i // 2)
    qator = []
    for u in range(n_u):
        items = r.choice(n_i, size=faollik[u], replace=False, p=pop)
        s = 3.5 + bu[u] + bi[items] + P[u] @ Q[items].T / np.sqrt(k) * 0.9 + r.normal(0, 0.45, len(items))
        t = r.uniform(0, 365, len(items))                              # yil ichidagi kun
        qator += list(zip([u] * len(items), items, np.clip(np.round(s), 1, 5), t))
    return pd.DataFrame(qator, columns=["user", "item", "rating", "day"]), n_u, n_i

df, n_u, n_i = reyting_yarat()
print("\n=== Ma'lumot tahlili ===")
zichlik = len(df) / (n_u * n_i)
print(f"Reytinglar: {len(df)} | foydalanuvchi: {n_u} | film: {n_i} | zichlik: {zichlik:.2%} (siyraklik {1 - zichlik:.2%})")
print("Reytinglar taqsimoti:", df.rating.value_counts().sort_index().to_dict())
cnt = df.item.value_counts().values
top10 = cnt[: n_i // 10].sum() / cnt.sum()
print(f"Uzun dum: eng mashhur 10% film barcha reytinglarning {top10:.1%} ini oladi")
uc = df.user.value_counts()
print(f"Foydalanuvchi faolligi: mediana {uc.median():.0f}, min {uc.min()}, max {uc.max()}; <10 reytingli (sovuq start xavfi): {(uc < 10).sum()}")

# --- Vaqt bo'yicha bo'lish: oxirgi 60 kun — test (kelajakni bashorat qilamiz, tasodifiy bo'lish ma'lumot sizdiradi)
tr, te = df[df.day < 305], df[df.day >= 305]
te = te[te.user.isin(tr.user) & te.item.isin(tr.item)]
print(f"Train: {len(tr)}, test: {len(te)} (vaqt bo'yicha, 305-kundan keyin)")

# --- Bazaviy modellar: global o'rtacha va regulyarizatsiyalangan siljish (bias) modeli
def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
mu = tr.rating.mean()
natija = [("Global o'rtacha", rmse(te.rating, mu))]
bu, bi = np.zeros(n_u), np.zeros(n_i)
for lam in (0.0, 10.0):
    bu[:], bi[:] = 0, 0
    for _ in range(10):                                                 # ALS uslubidagi navbatma-navbat yangilash
        g = tr.assign(res=tr.rating - mu - bu[tr.user]).groupby("item").res
        bi[g.sum().index] = g.sum().values / (g.count().values + lam)
        g = tr.assign(res=tr.rating - mu - bi[tr.item]).groupby("user").res
        bu[g.sum().index] = g.sum().values / (g.count().values + lam)
    pred = np.clip(mu + bu[te.user] + bi[te.item], 1, 5)
    natija.append((f"Bias-model (lambda={lam:g})", rmse(te.rating, pred)))
print("\n=== Bazaviy modellar ===")
print(pd.DataFrame(natija, columns=["Model", "RMSE (test)"]).round(4).to_string(index=False))
print("Xulosa: regulyarizatsiya kam reytingli film/foydalanuvchida siljishni 0 ga tortadi va testda RMSE ni kamaytiradi.")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].plot(np.arange(1, n_i + 1), cnt, color="tab:red"); ax[0].set_yscale("log")
ax[0].set_title("Uzun dum: film mashhurligi"); ax[0].set_xlabel("Film reytingi (o'rin)"); ax[0].set_ylabel("Reytinglar soni")
ax[1].hist(uc.values, bins=30, color="tab:blue"); ax[1].set_title("Foydalanuvchi faolligi"); ax[1].set_xlabel("Reytinglar soni")
plt.tight_layout()
