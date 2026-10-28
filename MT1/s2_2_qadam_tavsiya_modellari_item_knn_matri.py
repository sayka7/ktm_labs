"""MT1 — 2-qadam. Tavsiya modellari: item-kNN, matrix factorization va offline metrikalar

Item-kNN (o'rtachadan farq bo'yicha kosinus o'xshashlik, K=30) va SGD bilan o'qitiladigan matrix factorization (k=8) quriladi. Reyting bashorati RMSE bilan, top-10 tavsiya esa precision@10, recall@10, katalog qamrovi (coverage) va yangilik (novelty) bilan baholanadi. Muhim saboq: vaqt bo'yicha offline testda foydalanuvchi faqat ko'rgan filmini baholaydi, shuning uchun mashhurlik modeli precision'da kuchli ko'rinadi (offline baholashning mashhurlik siljishi); gibrid model (MF + log mashhurlik) aniqlik va xilma-xillik o'rtasida murosa beradi.

Kutilgan natija: RMSE: item-kNN ~1,0, MF ~0,89; precision@10: mashhurlik ~0,09, gibrid ~0,09, MF ~0,02; coverage: item-kNN ~0,98, mashhurlik ~0,1 (taxminiy). MF o'qitish egri chizig'i.
"""
# 2-qadam. Tavsiya modellari: mashhurlik, item-kNN va matrix factorization (SGD); offline metrikalar
import warnings; warnings.filterwarnings("ignore")
import time, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def reyting_yarat(n_u=600, n_i=400, k=6, seed=42):
    """1-qadamdagi bilan bir xil sintetik Netflix uslubidagi generator."""
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
R = np.zeros((n_u, n_i), np.float32); R[tr.user, tr.item] = tr.rating          # zich matritsa (kichik o'lcham)
M = R > 0
def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))

# --- 1) Item-kNN: o'rtachadan farq bo'yicha kosinus o'xshashlik, k eng yaqin qo'shni
t0 = time.perf_counter()
umean = np.where(M.sum(1) > 0, R.sum(1) / np.maximum(M.sum(1), 1), mu)
C = np.where(M, R - umean[:, None], 0.0)
norm = np.linalg.norm(C, axis=0) + 1e-9
S = (C.T @ C) / np.outer(norm, norm); np.fill_diagonal(S, 0)
K = 30
idx = np.argsort(-S, axis=1)[:, K:]; Sk = S.copy(); np.put_along_axis(Sk, idx, 0, axis=1)   # faqat top-K qo'shni
Sk = np.maximum(Sk, 0)
def knn_baho(u, items):
    w = Sk[items][:, M[u]]; c = C[u, M[u]]
    return umean[u] + (w @ c) / (w.sum(1) + 1e-6)
t_knn = time.perf_counter() - t0
p_knn = np.concatenate([knn_baho(u, g.item.values) for u, g in te.groupby("user")])
y_sorted = np.concatenate([g.rating.values for _, g in te.groupby("user")])

# --- 2) Matrix factorization: r = mu + bu + bi + p_u . q_i, SGD bilan
def mf_oqit(k=8, epoxa=25, lr=0.01, reg=0.05, seed=0):
    r = np.random.default_rng(seed)
    P, Q = r.normal(0, 0.1, (n_u, k)), r.normal(0, 0.1, (n_i, k)); bu, bi = np.zeros(n_u), np.zeros(n_i)
    U, I, Y = tr.user.values, tr.item.values, tr.rating.values
    tarix = []
    for ep in range(epoxa):
        for j in r.permutation(len(Y)):
            u, i = U[j], I[j]
            e = Y[j] - (mu + bu[u] + bi[i] + P[u] @ Q[i])
            bu[u] += lr * (e - reg * bu[u]); bi[i] += lr * (e - reg * bi[i])
            P[u], Q[i] = P[u] + lr * (e * Q[i] - reg * P[u]), Q[i] + lr * (e * P[u] - reg * Q[i])
        tarix.append(rmse(te.rating.values, np.clip(mu + bu[te.user] + bi[te.item] + np.sum(P[te.user] * Q[te.item], 1), 1, 5)))
    return P, Q, bu, bi, tarix
t0 = time.perf_counter(); P, Q, bu, bi, tarix = mf_oqit(); t_mf = time.perf_counter() - t0
print("=== Reyting bashorati (RMSE) ===")
print(pd.DataFrame([("Global o'rtacha", rmse(te.rating, mu), 0.0),
                    (f"Item-kNN (K={K})", rmse(y_sorted, np.clip(p_knn, 1, 5)), t_knn),
                    ("Matrix factorization (k=8)", tarix[-1], t_mf)],
                   columns=["Model", "RMSE", "O'qitish, s"]).round(4).to_string(index=False))

# --- Top-N tavsiya va reyting metrikalari: test reytingi >= 4 bo'lgan filmlar — «yoqqan» (relevant)
N = 10
releg = te[te.rating >= 4].groupby("user").item.apply(set)
def top_n(baho_vektor, u):
    s = baho_vektor.copy(); s[M[u]] = -np.inf                          # ko'rilganlarni tavsiya qilmaymiz
    return np.argpartition(-s, N)[:N]
pop_baho = M.sum(0).astype(float)
modellar = {
    "Mashhurlik": lambda u: pop_baho,
    "Item-kNN": lambda u: knn_baho(u, np.arange(n_i)),
    "MF": lambda u: mu + bu[u] + bi + Q @ P[u],
    # Gibrid: MF bahosi + mashhurlik logarifmi (ko'rilish ehtimoli). Offline testdagi «ko'rilganlik» siljishini hisobga oladi
    "Gibrid MF+log(pop)": lambda u: mu + bu[u] + bi + Q @ P[u] + 0.8 * np.log1p(pop_baho),
}
qator = []
for nom, f in modellar.items():
    prec, rec, tavsiya = [], [], set()
    for u, rel in releg.items():
        top = top_n(f(u), u); hit = len(rel & set(top))
        prec.append(hit / N); rec.append(hit / len(rel)); tavsiya |= set(top)
    novelty = np.mean([-np.log2((pop_baho[list(tavsiya)] + 1) / len(tr))])
    qator.append((nom, np.mean(prec), np.mean(rec), len(tavsiya) / n_i, novelty))
print(f"\n=== Top-{N} tavsiya metrikalari ===")
jad = pd.DataFrame(qator, columns=["Model", f"precision@{N}", f"recall@{N}", "coverage", "novelty, bit"]).round(4)
print(jad.to_string(index=False))
print("Izoh: vaqt bo'yicha offline testda foydalanuvchi faqat KO'RGAN filmini baholaydi, ko'rish esa mashhurlikka bog'liq —")
print("shuning uchun mashhurlik precision'da kuchli ko'rinadi (offline baholashning mashhurlik siljishi). Lekin uning coverage'i past:")
print("katalogning kichik qismi ko'rsatiladi. Gibrid model aniqlik va xilma-xillik o'rtasidagi murosani beradi; yakuniy hukm — A/B test (3-qadam).")

plt.figure(figsize=(5, 3.2)); plt.plot(range(1, len(tarix) + 1), tarix, marker=".")
plt.xlabel("Epoxa"); plt.ylabel("Test RMSE"); plt.title("MF o'qitish egri chizig'i"); plt.tight_layout()
