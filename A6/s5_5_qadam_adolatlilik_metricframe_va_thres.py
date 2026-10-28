"""A6 — 5-qadam. Adolatlilik: MetricFrame va ThresholdOptimizer

Sintetik kredit ma'lumotida sezgir atribut A, u bilan bog'liq vakil belgi (income) va tarixiy tarafkashlik (A=1 guruhda ijobiylarning 25% noto'g'ri rad etilgan) modellashtiriladi. Logistik regressiya uchun MetricFrame guruhlar kesimida aniqlik, tanlov ulushi, TPR, FPR beradi; DP va EO farqlari hisoblanadi. To'rt variant taqqoslanadi: A bilan, A siz (unawareness), ThresholdOptimizer DP va EO cheklovlari bilan.

Kutilgan natija: Baza: DP farqi ~0.25, EO ~0.27; A olib tashlanganda DP ~0.13 (proxy orqali saqlanadi); ThresholdOptimizer(DP) DP ni ~0.01 ga, (EO) EO ni ~0.02 ga tushiradi, aniqlik 1-3 bandga pasayadi.
"""
# 5-qadam. Adolatlilik: Fairlearn MetricFrame va ThresholdOptimizer (sintetik "kredit" ma'lumoti)
# Sezgir atribut A (0/1). Tarixiy tarafkashlik: A=1 guruhda belgilangan maqsad (y) past baholangan.
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fairlearn.metrics import (MetricFrame, selection_rate, true_positive_rate, false_positive_rate,
                               demographic_parity_difference, equalized_odds_difference)
from fairlearn.postprocessing import ThresholdOptimizer

rng = np.random.default_rng(0); n = 6000
A = rng.binomial(1, 0.4, n)                               # sezgir atribut (masalan, hudud/jins)
income = rng.normal(5.0 - 0.8 * A, 1.0, n)                # A bilan bog'liq (vakil belgi — proxy)
history = rng.normal(0, 1, n); debt = rng.normal(0, 1, n)
score = 0.9 * income + 1.0 * history - 0.8 * debt
y_true_ability = (score + rng.normal(0, 1, n) > 4.5).astype(int)   # haqiqiy to'lov qobiliyati
# tarixiy ma'lumotdagi yorliq: A=1 guruhda haqiqiy ijobiylarning 25% noto'g'ri "rad" deb yozilgan
flip = (A == 1) & (y_true_ability == 1) & (rng.random(n) < 0.25)
y = np.where(flip, 0, y_true_ability)
X = pd.DataFrame({"income": income, "history": history, "debt": debt, "A": A})
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
Atr, Ate = Xtr["A"].values, Xte["A"].values
print(f"Bazaviy ulush P(y=1|A): A=0 -> {y[A==0].mean():.3f}, A=1 -> {y[A==1].mean():.3f}")

def report(name, pred):
    mf = MetricFrame(metrics={"aniqlik": accuracy_score, "tanlov": selection_rate,
                              "TPR": true_positive_rate, "FPR": false_positive_rate},
                     y_true=yte, y_pred=pred, sensitive_features=Ate)
    dp = demographic_parity_difference(yte, pred, sensitive_features=Ate)
    eo = equalized_odds_difference(yte, pred, sensitive_features=Ate)
    print(f"\n--- {name}: umumiy aniqlik={accuracy_score(yte, pred):.4f}  DP farqi={dp:.3f}  EO farqi={eo:.3f}")
    print(mf.by_group.round(3).to_string())
    return accuracy_score(yte, pred), dp, eo

print("=== Adolatlilik tahlili (MetricFrame) ===")
rows = {}
clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
rows["Baza (A belgi sifatida kiradi)"] = report("Baza (A belgi sifatida kiradi)", clf.predict(Xte))

feats = ["income", "history", "debt"]                      # "bilmaslik orqali adolat": A olib tashlanadi
clf2 = LogisticRegression(max_iter=1000).fit(Xtr[feats], ytr)
rows["A olib tashlangan (unawareness)"] = report("A olib tashlangan (unawareness)", clf2.predict(Xte[feats]))

for c in ("demographic_parity", "equalized_odds"):
    to = ThresholdOptimizer(estimator=clf2, constraints=c, prefit=True, predict_method="predict_proba")
    to.fit(Xtr[feats], ytr, sensitive_features=Atr)
    pred = to.predict(Xte[feats], sensitive_features=Ate, random_state=0)
    rows[f"ThresholdOptimizer ({c})"] = report(f"ThresholdOptimizer ({c})", pred)

print("\n=== Yakuniy jadval ===")
print(f"{'Usul':42s} {'aniqlik':>8s} {'DP farqi':>9s} {'EO farqi':>9s}")
for k, (a, dp, eo) in rows.items(): print(f"{k:42s} {a:8.4f} {dp:9.3f} {eo:9.3f}")
print("\nXulosa: A ni olib tashlash farqni yo'qotmaydi (income — vakil belgi). Post-processing guruhga xos")
print("chegaralar bilan tanlangan mezonni kamaytiradi, lekin aniqlik biroz tushadi va inferensda A kerak bo'ladi.")
print("DP va EO ni bir vaqtda to'liq qondirib bo'lmaydi — mezon masala kontekstiga qarab tanlanadi.")
