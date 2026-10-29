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

# --- Berilgan ma'lumot generatori (o'zgartirmang)
rng = np.random.default_rng(0); n = 6000
A = rng.binomial(1, 0.4, n)                               # sezgir atribut
income = rng.normal(5.0 - 0.8 * A, 1.0, n)                # A bilan bog'liq (vakil belgi — proxy)
history = rng.normal(0, 1, n); debt = rng.normal(0, 1, n)
score = 0.9 * income + 1.0 * history - 0.8 * debt
y_true_ability = (score + rng.normal(0, 1, n) > 4.5).astype(int)
flip = (A == 1) & (y_true_ability == 1) & (rng.random(n) < 0.25)
y = np.where(flip, 0, y_true_ability)
X = pd.DataFrame({"income": income, "history": history, "debt": debt, "A": A})
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
Atr, Ate = Xtr["A"].values, Xte["A"].values

def report(name, pred):
    # TODO 1: MetricFrame (aniqlik, tanlov, TPR, FPR) guruhlar kesimida; DP va EO farqlari; (aniqlik, dp, eo) qaytaring
    raise NotImplementedError("TODO 1: report")

print("=== Adolatlilik tahlili (MetricFrame) ===")
rows = {}
# TODO 2: Baza — LogisticRegression barcha belgilar (A bilan)
# TODO 3: A olib tashlangan (unawareness) model
# TODO 4: ThresholdOptimizer (prefit, predict_proba) — demographic_parity va equalized_odds cheklovlari
rows["Baza (A belgi sifatida kiradi)"] = report("Baza", None)

print("\n=== Yakuniy jadval ===")
for k, (a, dp, eo) in rows.items(): print(f"{k:42s} {a:8.4f} {dp:9.3f} {eo:9.3f}")
