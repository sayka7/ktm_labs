"""MT6 — 5-qadam. Tushuntiriluvchanlik: permutatsion muhimlik vs SHAP, barqarorlik, surrogat daraxt, kontrfaktual

Mikrokredit sintetik ma'lumotida (7 belgi, shundan bittasi ataylab ta'sirsiz) GradientBoosting o'qitiladi. Global muhimlik ikki usulda (permutatsion va SHAP TreeExplainer) hisoblanib, Kendall tau bilan solishtiriladi; 5 bootstrap qayta o'qitishda SHAP top-3 barqarorligi tekshiriladi; chuqurligi 3 bo'lgan surrogat daraxtning sodiqligi (fidelity) va qoidalari chiqadi. Rad etilgan ariza uchun «tushuntirish xati» (eng salbiy 3 omil) va bitta omilli kontrfaktual izohlar yaratiladi.

Kutilgan natija: Aniqlik ~0,81; ikki usul top-4 da kelishadi (tau ~0,6), ta'sirsiz belgi ikkalasida ham eng past; bootstrap top-3 to'plami ~100% barqaror; surrogat sodiqligi ~0,89.
"""
# 5-qadam. Tushuntiriluvchanlik: permutatsion muhimlik vs SHAP, barqarorlik, surrogat daraxt va kontrfaktual izoh
# Ssenariy: mikrokredit arizasini baholash. Shaxsga qaror sababini tushuntirish (EU AI Act 86-modda, 13-modda ruhida).
import numpy as np, pandas as pd, shap
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
rng = np.random.default_rng(0); n = 3000
X = pd.DataFrame({
    "oylik_daromad_mln": np.round(rng.lognormal(1.3, 0.45, n), 2),
    "qarz_yuki_ulushi": np.round(rng.beta(2, 5, n), 3),
    "kredit_tarixi_oy": rng.integers(0, 120, n),
    "kechikishlar_soni": rng.poisson(0.8, n),
    "ish_staji_yil": np.round(rng.gamma(2, 2.5, n), 1),
    "yosh": rng.integers(20, 65, n),
    "tasodifiy_belgi": rng.normal(0, 1, n),          # haqiqatda ta'sirsiz — usullar buni "sezmasligi" kerak
})
logit = 1.8 * (0.9 * np.log(X.oylik_daromad_mln) - 4.0 * X.qarz_yuki_ulushi + 0.012 * X.kredit_tarixi_oy - 0.9 * X.kechikishlar_soni
         + 0.08 * X.ish_staji_yil + 0.3).values
y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)        # 1 = qaytaradi (tasdiq)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
gb = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=0).fit(Xtr, ytr)
print(f"Model aniqligi (test): {gb.score(Xte, yte):.3f}")

# TODO 1: permutatsion muhimlik (test to'plamida, n_repeats=8, accuracy) va SHAP TreeExplainer |o'rtacha| qiymatlari
# TODO 2: ikkala muhimlikni bitta jadvalga yig'ib, Kendall tau bilan reyting mosligini hisoblang
print("\n=== Global muhimlik: permutatsion vs SHAP ===")
raise NotImplementedError("TODO 1-2: global muhimlik")

# TODO 3: 5 ta bootstrap qayta o'qitishda SHAP top-3 belgilari va ularning mosligi (%)
# TODO 4: chuqurligi 3 bo'lgan surrogat daraxtni MODEL BASHORATLARIGA o'qitib, sodiqlik (fidelity) va qoidalarni chiqaring
print("=== Lokal izoh: rad etilgan ariza uchun tushuntirish xati ===")
# TODO 5: chegara atrofidagi rad etilgan ariza uchun eng salbiy 3 SHAP hissasi
# TODO 6: bitta omilni (qarz yuki, kechikishlar, daromad) o'zgartirib tasdiqqa (p >= 0.5) olib keluvchi qiymatni toping
