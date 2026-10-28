"""A6 — 6-qadam. SHAP TreeExplainer: global muhimlik, lokal izoh, additivlik

breast_cancer (569x30) ustida GradientBoosting o'qitiladi, TreeExplainer log-odds fazosida SHAP qiymatlarini beradi. Global o'rtacha |SHAP| top-10, bitta namuna uchun top-6 hissalar, additivlik tekshiruvi (bazaviy + sum(SHAP) = decision_function, farq ~1e-15) va 1% shovqinda tushuntirish barqarorligi chiqadi; beeswarm grafigi chiziladi.

Kutilgan natija: Aniqlik ~0.94; top belgilar: worst perimeter, worst concave points, mean concave points; additivlik farqi ~1e-15; barqarorlik ~0.99.
"""
# 6-qadam. SHAP (TreeExplainer) bilan daraxt modelini tushuntirish: global muhimlik, lokal izoh, additivlik
import numpy as np, pandas as pd, shap, matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

d = load_breast_cancer(as_frame=True)
X, y = d.data, d.target                                   # 569 namuna, 30 belgi, 1 = xavfsiz (benign)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
clf = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=0).fit(Xtr, ytr)
print(f"Model aniqligi (test): {clf.score(Xte, yte):.4f}")

ex = shap.TreeExplainer(clf)                              # daraxtlar uchun aniq va polinomial TreeSHAP
sv = ex.shap_values(Xte)                                  # log-odds fazosida, shakli (n, 30)
sv = sv[..., 1] if sv.ndim == 3 else sv
base = np.ravel(ex.expected_value)[-1]

print("=== SHAP: global muhimlik (o'rtacha |SHAP|) — top-10 ===")
imp = pd.Series(np.abs(sv).mean(0), index=X.columns).sort_values(ascending=False)
for i, (k, v) in enumerate(imp.head(10).items(), 1): print(f"{i:2d}. {k:28s} {v:.4f}")

i = 0
margin = clf.decision_function(Xte.iloc[[i]])[0]
print(f"\n=== Lokal tushuntirish: test namunasi #{i} (haqiqiy={yte.iloc[i]}, bashorat={clf.predict(Xte.iloc[[i]])[0]}) ===")
print(f"bazaviy qiymat (log-odds) = {base:.4f};  bazaviy + sum(SHAP) = {base + sv[i].sum():.4f};  model decision_function = {margin:.4f}")
print(f"Additivlik tekshiruvi: farq = {abs(base + sv[i].sum() - margin):.2e}")
contrib = pd.Series(sv[i], index=X.columns).sort_values(key=np.abs, ascending=False)
for k, v in contrib.head(6).items():
    print(f"  {k:28s} qiymat={Xte.iloc[i][k]:9.3f}  hissa={v:+.4f}  ({'xavfsiz tomon' if v > 0 else 'xavfli tomon'})")

# Barqarorlik: kichik shovqinda tushuntirish qanchalik o'zgaradi?
rng = np.random.default_rng(0)
Xn = Xte + rng.normal(0, 0.01, Xte.shape) * Xte.std().values
svn = ex.shap_values(Xn); svn = svn[..., 1] if svn.ndim == 3 else svn
top_same = np.mean([len(set(np.argsort(-np.abs(sv[j]))[:5]) & set(np.argsort(-np.abs(svn[j]))[:5])) / 5 for j in range(len(sv))])
print(f"\nTushuntirish barqarorligi: 1% shovqinda top-5 belgilarning o'rtacha mosligi = {top_same:.2f}")

shap.summary_plot(sv, Xte, max_display=12, show=False)
plt.title("SHAP beeswarm — GradientBoosting, breast_cancer"); plt.tight_layout()
print("Xulosa: TreeSHAP lokal aniqlik aksiomasini qondiradi (hissalar yig'indisi = bashorat), beeswarm global ko'rinish beradi.")
