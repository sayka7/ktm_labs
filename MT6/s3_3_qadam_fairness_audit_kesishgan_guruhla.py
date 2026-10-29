"""MT6 — 3-qadam. Fairness audit: kesishgan guruhlar (jins x yosh), 4/5 qoidasi va yumshatish usullari

Ishga qabul (III-4 soha) sintetik ma'lumotida tarixiy qarorlar «ish tanaffusi» (jins vakili) va >45 yosh bo'yicha tarafkash. Fairlearn MetricFrame ikki sezgir atribut bilan 6 kesishgan guruh uchun tanlov ulushi, TPR, FPR beradi; disparate impact (eng kam/eng ko'p tanlov) 4/5 qoidasi bilan tekshiriladi. To'rt variant taqqoslanadi: bazaviy, vakil belgilarsiz, Kamiran–Calders qayta tortish (qo'lda) va ExponentiatedGradient (DemographicParity, kesishgan guruh bo'yicha).

Kutilgan natija: Bazaviy DI ~0,5 (4/5 qoidasidan o'tmaydi); vakil belgilarni olib tashlash DI ni yaxshilamaydi, qayta tortish ~0,7, ExpGrad ~0,77 ga ko'taradi, aniqlik 1–4 bandga pasayadi.
"""
# 3-qadam. Fairness audit: kesishgan guruhlar (jins x yosh), "4/5 qoidasi" va ikki yumshatish usuli
# Ssenariy: ishga qabulda nomzodlarni saralash (EU AI Act III-4 — yuqori xavf). Ma'lumot sintetik.
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate, demographic_parity_ratio, equalized_odds_difference
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

rng = np.random.default_rng(1); n = 6000
sex = rng.choice(["ayol", "erkak"], n, p=[0.45, 0.55]); age = rng.choice(["<30", "30-45", ">45"], n, p=[0.35, 0.45, 0.2])
age_num = np.where(age == "<30", rng.uniform(21, 30, n), np.where(age == "30-45", rng.uniform(30, 45, n), rng.uniform(45, 60, n)))
grad_year = np.round(2026 - (age_num - 22) + rng.normal(0, 1.5, n))     # bitirgan yil — yoshning vakil belgisi
exp_years = np.clip(rng.normal(6, 3, n) + 0.1 * (age_num - 30), 0, None)
skill = rng.normal(0, 1, n); edu = rng.integers(0, 3, n)
gap = ((sex == "ayol") & (rng.random(n) < 0.35)).astype(int)            # ish tanaffusi (masalan, dekret) — jinsning vakil belgisi
qual = 0.8 * skill + 0.25 * exp_years + 0.4 * edu + rng.normal(0, 1, n)
# tarixiy qarorlar: tanaffusli va >45 nomzodlar nohaq jazolangan (yorliq tarafkashligi)
hist = qual - 1.0 * gap - 1.2 * (age == ">45") > np.quantile(qual, 0.6)
X = pd.DataFrame({"tajriba": exp_years, "malaka_testi": skill, "talim": edu, "ish_tanaffusi": gap, "bitirgan_yil": grad_year})
A = pd.DataFrame({"jins": sex, "yosh": age})
Xtr, Xte, ytr, yte, Atr, Ate = train_test_split(X, hist.astype(int), A, test_size=0.4, random_state=0, stratify=hist)

# TODO 1: MetricFrame (tanlov ulushi, TPR, FPR) ni ikki sezgir atribut bo'yicha hisoblang;
#         DI = eng kam / eng ko'p tanlov ulushi; aniqlik, DP ratio (jins, yosh), EO farqi va 4/5 qoidasi natijasini qaytaring
def audit(name, pred):
    raise NotImplementedError("TODO 1: fairness audit")

# TODO 2: bazaviy LogisticRegression ni o'qitib, kesishgan guruhlar jadvalini va DI ni chiqaring
base = None
mf, r0 = audit("Bazaviy (LR)", base.predict(Xte))
print("=== Fairness audit: kesishgan guruhlar (jins x yosh), bazaviy model ===")

# TODO 3: Kamiran-Calders qayta tortish: w(guruh, yorliq) = P(guruh) * P(yorliq) / P(guruh, yorliq)
# TODO 4: ExponentiatedGradient + DemographicParity(difference_bound=0.03), kesishgan guruh bo'yicha
# TODO 5: vakil belgilarni (ish_tanaffusi, bitirgan_yil) olib tashlab o'qitish ("unawareness")
print("\n=== Yumshatish usullari taqqoslash ===")
# TODO 6: to'rt variant natijalarini bitta jadvalda chiqaring
