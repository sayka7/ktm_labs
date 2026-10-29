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

# TODO 1: TreeExplainer bilan test to'plami SHAP qiymatlari (n, 30) va bazaviy qiymat (log-odds)
sv, base = None, None

print("=== SHAP: global muhimlik (o'rtacha |SHAP|) — top-10 ===")
# TODO 2: o'rtacha |SHAP| bo'yicha saralangan top-10 belgi
# TODO 3: test namunasi #0 uchun additivlik: base + sum(SHAP) va decision_function farqi ("Additivlik tekshiruvi: farq = ...");
#         eng katta 6 hissa (ishorasi bilan)
print(f"Additivlik tekshiruvi: farq = {abs(base + sv[0].sum()):.2e}")
# TODO 4: 1 % shovqin qo'shib top-5 belgilar mosligi (barqarorlik); beeswarm grafigi (shap.summary_plot, show=False)
