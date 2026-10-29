"""MT1 — 1-qadam. Hayot sikli xaritasi, sintetik reytinglar va bazaviy modellar

AI mahsulot hayot siklining 8 bosqichi va ularni ushbu ishning qaysi qadami qamrashi jadvalda ko'rsatiladi. So'ngra Netflix uslubidagi sintetik reyting matritsasi yaratiladi (600 foydalanuvchi, 400 film, yashirin didlar, Zipf mashhurlik) va EDA o'tkaziladi: siyraklik, uzun dum, sovuq start xavfi. Ma'lumot **vaqt bo'yicha** bo'linadi (tasodifiy bo'lish kelajakdan ma'lumot sizdiradi) va ikki bazaviy model — global o'rtacha va regulyarizatsiyalangan bias-model — RMSE bo'yicha solishtiriladi. O'zgartirib ko'ring: `lambda`, mashhurlik darajasi 0,9, faollik taqsimoti.

Kutilgan natija: Siyraklik ~93%, eng mashhur 10% film reytinglarning ~40–45% ini oladi; RMSE: global o'rtacha ~1,09, bias-model ~0,95 (taxminiy). Ikki grafik: uzun dum va faollik.
"""
# 1-qadam. AI mahsulot hayot sikli: muammo ta'rifi, sintetik Netflix uslubidagi reytinglar, EDA va bazaviy modellar
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# TODO 1: hayot siklining 8 bosqichini (bosqich, mazmun, qaysi qadam) ro'yxat qilib, jadval ko'rinishida chiqaring
bosqichlar = [
    ("1. Muammo va biznes maqsadi", "...", "s1"),
    # ... qolgan 7 bosqich
]
print("=== Hayot sikli bosqichlari ===")
raise NotImplementedError("TODO 1: bosqichlar jadvalini to'ldiring va chiqaring")

# --- Sintetik reyting generatori (berilgan, o'zgartirmang)
def reyting_yarat(n_u=600, n_i=400, k=6, seed=42):
    r = np.random.default_rng(seed)
    P, Q = r.normal(0, 1, (n_u, k)), r.normal(0, 1, (n_i, k))
    bu, bi = r.normal(0, 0.35, n_u), r.normal(0, 0.5, n_i)
    pop = 1.0 / np.arange(1, n_i + 1) ** 0.9; pop = pop[r.permutation(n_i)]; pop /= pop.sum()
    faollik = np.clip(r.lognormal(3.0, 0.7, n_u).astype(int) + 5, 5, n_i // 2)
    qator = []
    for u in range(n_u):
        items = r.choice(n_i, size=faollik[u], replace=False, p=pop)
        s = 3.5 + bu[u] + bi[items] + P[u] @ Q[items].T / np.sqrt(k) * 0.9 + r.normal(0, 0.45, len(items))
        qator += list(zip([u] * len(items), items, np.clip(np.round(s), 1, 5), r.uniform(0, 365, len(items))))
    return pd.DataFrame(qator, columns=["user", "item", "rating", "day"]), n_u, n_i

df, n_u, n_i = reyting_yarat()
print("\n=== Ma'lumot tahlili ===")
# TODO 2: zichlik/siyraklik, eng mashhur 10% film ulushi (uzun dum) va <10 reytingli foydalanuvchilar sonini hisoblang
zichlik = top10 = None

# TODO 3: ma'lumotni VAQT bo'yicha bo'ling (day < 305 — train), testda faqat train'da bor user/item qolsin
tr = te = None

# TODO 4: rmse funksiyasi, global o'rtacha va lambda=0 va 10 bilan bias-model (bu, bi navbatma-navbat yangilanadi)
def rmse(a, b): ...
natija = []
print("\n=== Bazaviy modellar ===")
raise NotImplementedError("TODO 4: bazaviy modellarni o'qitib, RMSE jadvalini chiqaring")
