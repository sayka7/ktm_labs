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

def audit(name, pred):
    mf = MetricFrame(metrics={"tanlov": selection_rate, "TPR": true_positive_rate, "FPR": false_positive_rate},
                     y_true=yte, y_pred=pred, sensitive_features=Ate)
    sr = mf.by_group["tanlov"]
    di = sr.min() / sr.max()                               # disparate impact (eng kam / eng ko'p tanlov ulushi)
    return mf, dict(usul=name, aniqlik=accuracy_score(yte, pred), DI_kesishgan=di,
                    DP_ratio_jins=demographic_parity_ratio(yte, pred, sensitive_features=Ate["jins"]),
                    DP_ratio_yosh=demographic_parity_ratio(yte, pred, sensitive_features=Ate["yosh"]),
                    EO_diff=equalized_odds_difference(yte, pred, sensitive_features=Ate), qoida_4_5="o'tdi" if di >= 0.8 else "O'TMADI")

base = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
mf, r0 = audit("Bazaviy (LR)", base.predict(Xte))
print("=== Fairness audit: kesishgan guruhlar (jins x yosh), bazaviy model ===")
print(mf.by_group.round(3).to_string())
print(f"Eng past/eng yuqori tanlov ulushi = {r0['DI_kesishgan']:.2f}  (4/5 qoidasi: >= 0.80) -> {r0['qoida_4_5']}")

# ---- Yumshatish 1 (oldindan ishlov): Kamiran-Calders qayta tortish — (guruh, yorliq) juftliklari muvozanatlanadi ----
g = Atr["jins"] + "|" + Atr["yosh"]
w = np.ones(len(ytr))
for gg in g.unique():
    for yy in (0, 1):
        m = (g == gg).values & (ytr == yy)
        if m.sum(): w[m] = (g == gg).mean() * (ytr == yy).mean() / m.mean()
rw = LogisticRegression(max_iter=1000).fit(Xtr, ytr, sample_weight=w)
_, r1 = audit("Qayta tortish (reweighing)", rw.predict(Xte))
# ---- Yumshatish 2 (o'qitish ichida): ExponentiatedGradient + DemographicParity, kesishgan guruh bo'yicha ----
eg = ExponentiatedGradient(LogisticRegression(max_iter=1000), DemographicParity(difference_bound=0.03), max_iter=40)
eg.fit(Xtr, ytr, sensitive_features=g)
_, r2 = audit("ExpGrad (DP<=0.03)", eg.predict(Xte))
# ---- Taqqoslash uchun: vakil belgini olib tashlash ("unawareness") ----
PROXY = ["ish_tanaffusi", "bitirgan_yil"]
nb = LogisticRegression(max_iter=1000).fit(Xtr.drop(columns=PROXY), ytr)
_, r3 = audit("Vakil belgilarsiz", nb.predict(Xte.drop(columns=PROXY)))
tab = pd.DataFrame([r0, r3, r1, r2])
print("\n=== Yumshatish usullari taqqoslash ===")
print(tab.round(3).to_string(index=False))
print("\nEslatma: yorliqning o'zi tarafkash (tarixiy qarorlar), shuning uchun 'aniqlik' ham tarafkash yorliqqa nisbatan o'lchanmoqda.")

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.scatter(tab.DI_kesishgan, tab.aniqlik, s=60)
for _, r in tab.iterrows(): ax.annotate(r.usul, (r.DI_kesishgan, r.aniqlik), fontsize=7, xytext=(4, 4), textcoords="offset points")
ax.axvline(0.8, ls="--", c="r"); ax.set_xlabel("disparate impact (kesishgan)"); ax.set_ylabel("aniqlik"); ax.set_title("Aniqlik va adolatlilik murozasi"); ax.grid(alpha=.3)
plt.tight_layout()
print("\nIzoh: real auditda sezgir atributlarni yig'ish va ishlatishning huquqiy asosi (GDPR 9-modda, EU AI Act 10(5)-modda)")
print("hamda guruh kichikligi (statistik ishonchlilik, bootstrap oraliqlari) alohida tekshiriladi.")
