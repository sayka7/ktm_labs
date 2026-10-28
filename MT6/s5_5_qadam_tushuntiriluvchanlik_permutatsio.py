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

pi = permutation_importance(gb, Xte, yte, n_repeats=8, random_state=0, scoring="accuracy")
ex = shap.TreeExplainer(gb); sv = ex.shap_values(Xte); sv = sv[..., 1] if sv.ndim == 3 else sv
imp = pd.DataFrame({"permutatsion": pi.importances_mean, "perm_std": pi.importances_std, "SHAP_|o'rtacha|": np.abs(sv).mean(0)}, index=X.columns)
imp = imp.sort_values("SHAP_|o'rtacha|", ascending=False)
print("\n=== Global muhimlik: permutatsion vs SHAP ===")
print(imp.round(4).to_string())
tau, _ = kendalltau(imp["permutatsion"], imp["SHAP_|o'rtacha|"])
print(f"Reytinglar mosligi (Kendall tau) = {tau:.2f}")

print("\n=== Barqarorlik: 5 ta bootstrap qayta o'qitish, SHAP top-3 ===")
tops = []
for b in range(5):
    idx = rng.integers(0, len(Xtr), len(Xtr))
    m = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=b).fit(Xtr.iloc[idx], ytr[idx])
    s = shap.TreeExplainer(m).shap_values(Xte.iloc[:300]); s = s[..., 1] if s.ndim == 3 else s
    top = list(pd.Series(np.abs(s).mean(0), index=X.columns).sort_values(ascending=False).index[:3]); tops.append(top)
    print(f"  bootstrap {b + 1}: {top}")
agree = np.mean([set(t) == set(tops[0]) for t in tops])
print(f"top-3 to'plami mosligi: {agree:.0%}")

print("\n=== Global surrogat: chuqurligi 3 bo'lgan qaror daraxti modelni qanchalik takrorlaydi ===")
sur = DecisionTreeClassifier(max_depth=3, random_state=0).fit(Xtr, gb.predict(Xtr))
fid = (sur.predict(Xte) == gb.predict(Xte)).mean()
print(f"Sodiqlik (fidelity) = {fid:.3f}")
print(export_text(sur, feature_names=list(X.columns), decimals=2))

print("=== Lokal izoh: rad etilgan ariza uchun tushuntirish xati ===")
proba = gb.predict_proba(Xte)[:, 1]
i = int(np.argsort(np.abs(proba - 0.35))[0])          # chegara atrofidagi rad etilgan ariza
xi = Xte.iloc[[i]]; contrib = pd.Series(sv[i], index=X.columns).sort_values()
print(f"Tasdiqlanish ehtimoli = {proba[i]:.2f} (chegara 0.50) -> RAD ETILDI")
print("Qarorga eng ko'p salbiy ta'sir qilgan omillar:")
for k, v in contrib.head(3).items(): print(f"  - {k} = {xi[k].values[0]}  (hissa {v:+.3f} log-odds)")
print("Kontrfaktual (bitta omilni o'zgartirib tasdiqqa erishish):")
for feat, grid in [("qarz_yuki_ulushi", np.linspace(xi.qarz_yuki_ulushi.values[0], 0, 41)),
                   ("kechikishlar_soni", np.arange(xi.kechikishlar_soni.values[0], -1, -1)),
                   ("oylik_daromad_mln", np.linspace(xi.oylik_daromad_mln.values[0], xi.oylik_daromad_mln.values[0] * 3, 41))]:
    hit = None
    for v in grid:
        z = xi.copy(); z[feat] = v
        if gb.predict_proba(z)[0, 1] >= 0.5: hit = v; break
    print(f"  {feat}: {xi[feat].values[0]} -> " + (f"{hit:.3g} bo'lsa tasdiqlanadi" if hit is not None else "bu omil yetarli emas"))

fig, ax = plt.subplots(figsize=(7, 3.5)); y_ = np.arange(len(imp))
ax.barh(y_ - 0.2, imp["permutatsion"] / imp["permutatsion"].max(), 0.4, label="permutatsion (norm.)")
ax.barh(y_ + 0.2, imp["SHAP_|o'rtacha|"] / imp["SHAP_|o'rtacha|"].max(), 0.4, label="SHAP (norm.)")
ax.set_yticks(y_); ax.set_yticklabels(imp.index, fontsize=8); ax.invert_yaxis(); ax.legend(fontsize=8); ax.set_title("Global muhimlik"); plt.tight_layout()
print("\nIzoh: real tizimda tushuntirish xati yurist va UX mutaxassisi bilan tuziladi; kontrfaktual faqat o'zgartirish")
print("mumkin bo'lgan (yosh emas) omillar bo'yicha beriladi va izohlar sodiqligi muntazam tekshiriladi.")
