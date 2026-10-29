"""MT1 — 2-qadam. Tavsiya modellari: item-kNN, matrix factorization va offline metrikalar

Item-kNN (o'rtachadan farq bo'yicha kosinus o'xshashlik, K=30) va SGD bilan o'qitiladigan matrix factorization (k=8) quriladi. Reyting bashorati RMSE bilan, top-10 tavsiya esa precision@10, recall@10, katalog qamrovi (coverage) va yangilik (novelty) bilan baholanadi. Muhim saboq: vaqt bo'yicha offline testda foydalanuvchi faqat ko'rgan filmini baholaydi, shuning uchun mashhurlik modeli precision'da kuchli ko'rinadi (offline baholashning mashhurlik siljishi); gibrid model (MF + log mashhurlik) aniqlik va xilma-xillik o'rtasida murosa beradi.

Kutilgan natija: RMSE: item-kNN ~1,0, MF ~0,89; precision@10: mashhurlik ~0,09, gibrid ~0,09, MF ~0,02; coverage: item-kNN ~0,98, mashhurlik ~0,1 (taxminiy). MF o'qitish egri chizig'i.
"""
# 2-qadam. Tavsiya modellari: mashhurlik, item-kNN va matrix factorization (SGD); offline metrikalar
import warnings; warnings.filterwarnings("ignore")
import time, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def reyting_yarat(n_u=600, n_i=400, k=6, seed=42):
    """1-qadamdagi bilan bir xil sintetik Netflix uslubidagi generator (berilgan)."""
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
tr, te = df[df.day < 305], df[df.day >= 305]
te = te[te.user.isin(tr.user) & te.item.isin(tr.item)]
mu = tr.rating.mean()
R = np.zeros((n_u, n_i), np.float32); R[tr.user, tr.item] = tr.rating
M = R > 0
def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))

# TODO 1: item-kNN — foydalanuvchi o'rtachasidan farq matritsasi C, filmlar orasidagi kosinus o'xshashlik S, top-K=30 qo'shni
K = 30
def knn_baho(u, items):
    raise NotImplementedError("TODO 1: item-kNN bahosi")

# TODO 2: matrix factorization r = mu + bu + bi + p_u.q_i ni SGD bilan o'qiting (k=8, 25 epoxa), har epoxada test RMSE ni yozing
def mf_oqit(k=8, epoxa=25, lr=0.01, reg=0.05, seed=0):
    raise NotImplementedError("TODO 2: MF SGD")

P, Q, bu, bi, tarix = mf_oqit()
print("=== Reyting bashorati (RMSE) ===")
# TODO 3: global o'rtacha, item-kNN va MF uchun RMSE jadvalini chiqaring

# TODO 4: top-10 tavsiya (ko'rilganlarsiz), relevant = test reytingi >= 4; precision@10, recall@10, coverage, novelty
N = 10
modellar = {"Mashhurlik": None, "Item-kNN": None, "MF": None, "Gibrid MF+log(pop)": None}
print(f"\n=== Top-{N} tavsiya metrikalari ===")
