"""A7 — 6-qadam. Adolatlilik: fairlearn MetricFrame hisoboti va yumshatish

Test to'plamida jins va yosh guruhlari bo'yicha n, TPR, FPR, tanlov ulushi va AUC `MetricFrame` bilan chiqariladi, TPR farqi SLO (<= 0.05) bilan solishtiriladi, bootstrap bilan 95% ishonch oralig'i hisoblanadi. Ikki yumshatish yo'li taqqoslanadi: (A) `ThresholdOptimizer` guruhga xos chegaralar, (B) ko'lamni 19+ bilan cheklash va chegarani qayta tanlash. O'zgartirib ko'ring: cheklov turi, chegara, bolalar shovqin koeffitsienti generator ichida.

Kutilgan natija: Jins bo'yicha farq ~0.01 (OK), yosh bo'yicha ~0.18 (BUZILDI, 0-18 TPR ~0.70, keng ishonch oralig'i); A: farq ~0.02, lekin sezgirlik 0.77; B: farq ~0.03, sezgirlik 0.90.
"""
# 6-qadam. Adolatlilik: fairlearn MetricFrame bilan kesimlar hisoboti, bootstrap ishonch oralig'i va yumshatish
# Ikki yo'l taqqoslanadi: (A) guruhga xos chegara (ThresholdOptimizer, post-processing) va
# (B) ko'lamni cheklash (0–18 yosh guruhini tizimdan chiqarish, model kartasida «mo'ljallanmagan foydalanish»).
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from fairlearn.metrics import MetricFrame, true_positive_rate, false_positive_rate, selection_rate, count
from fairlearn.postprocessing import ThresholdOptimizer

FEATS = ["opacity", "lung_area", "ct_ratio", "texture", "intensity", "edge_density"]

def sintetik_bemorlar(n_bemor=1500, seed=0):
    """1-qadamdagi generator: bemor-darajali sintetik manifest (bolalar tasviri shovqinliroq)."""
    rng = np.random.default_rng(seed); rows = []
    for pid in range(n_bemor):
        age = int(np.clip(rng.normal(50, 18), 1, 95)); sex = "M" if rng.random() < 0.56 else "F"
        z = rng.normal(0, 1) + 0.02 * (age - 50)
        for k in range(int(rng.integers(1, 4))):
            e = rng.normal(0, 2.2 if age < 18 else 1.0, 6)
            f = [0.9*z+e[0], -0.6*z+e[1], 0.5*z+e[2], 0.7*z+e[3], 0.2*z+e[4], 0.4*z+e[5]]
            rows.append([f"{pid:08d}_{k:03d}.png", pid, age, sex, *np.round(f, 4), int(z + rng.normal(0, 0.6) > 0.3)])
    return pd.DataFrame(rows, columns=["image_id", "patient_id", "age", "sex", *FEATS, "finding"])

df = sintetik_bemorlar(4000)
df["age_group"] = pd.cut(df.age, [0, 18, 40, 60, 100], labels=["0-18", "19-40", "41-60", "61+"]).astype(str)
tr_i, te_i = next(GroupShuffleSplit(1, test_size=0.35, random_state=42).split(df, groups=df.patient_id))
tr, te = df.iloc[tr_i].reset_index(drop=True), df.iloc[te_i].reset_index(drop=True)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(tr[FEATS], tr.finding)
te = te.assign(score=model.predict_proba(te[FEATS])[:, 1])
THRESHOLD = 0.30; te["pred"] = (te.score >= THRESHOLD).astype(int)
SLO_GAP = 0.05

def hisobot(y, pred, score, sens, nom):
    """MetricFrame: kesimlar bo'yicha n, TPR, FPR, tanlov ulushi, AUC; TPR farqi (max-min)."""
    mf = MetricFrame(metrics={"n": count, "tpr": true_positive_rate, "fpr": false_positive_rate, "sel": selection_rate},
                     y_true=y, y_pred=pred, sensitive_features=sens)
    auc = pd.Series({g: roc_auc_score(y[sens == g], score[sens == g]) for g in np.unique(sens)}, name="auc")
    rep = mf.by_group.join(auc).round(3)
    gap = float(mf.difference()["tpr"])
    print(f"\n-- {nom}: kesim = {sens.name}")
    print(rep.to_string())
    print(f"TPR farqi (max-min) = {gap:.3f}  SLO <= {SLO_GAP}: {'BAJARILDI' if gap <= SLO_GAP else 'BUZILDI'}")
    return gap

def bootstrap_tpr(y, pred, sens, n=300, seed=0):
    """Har guruh uchun TPR 95% ishonch oralig'i (bemor emas, tasvir bo'yicha oddiy bootstrap)."""
    rng = np.random.default_rng(seed); out = {}
    for g in np.unique(sens):
        m = (sens == g) & (y == 1); yy, pp = y[m], pred[m]; vals = []
        for _ in range(n):
            i = rng.integers(0, len(yy), len(yy)); vals.append(pp[i].mean())
        out[g] = (round(pp.mean(), 3), round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3), int(m.sum()))
    return out

y, s, p = te.finding.values, te.score.values, te.pred.values
print("=== Dastlabki model (chegara 0.30) ===")
print(f"test: n={len(te)}, AUC={roc_auc_score(y, s):.3f}, sezgirlik={recall_score(y, p):.3f}, aniqlik={precision_score(y, p):.3f}")
gap_sex = hisobot(y, p, s, te.sex, "Dastlabki")
gap_age = hisobot(y, p, s, te.age_group, "Dastlabki")
print("\nTPR 95% ishonch oralig'i (yosh guruhi): ", {g: f"{v[0]} [{v[1]}; {v[2]}] n+={v[3]}" for g, v in bootstrap_tpr(y, p, te.age_group.values).items()})
print("-> 0-18 guruhi: namunalar kam, ishonch oralig'i keng, TPR past — bu guruh uchun model ishonchsiz")

# --- Yumshatish A: ThresholdOptimizer (yosh guruhiga xos chegara, TPR tenglik cheklovi) ---
print("\n=== Yumshatish A: guruhga xos chegara (ThresholdOptimizer, true_positive_rate_parity) ===")
to = ThresholdOptimizer(estimator=model, constraints="true_positive_rate_parity", objective="balanced_accuracy_score",
                        predict_method="predict_proba", prefit=True)
to.fit(tr[FEATS], tr.finding, sensitive_features=tr.age_group)
pA = to.predict(te[FEATS], sensitive_features=te.age_group, random_state=0).astype(int)
print(f"umumiy: sezgirlik={recall_score(y, pA):.3f}, aniqlik={precision_score(y, pA):.3f}")
gapA = hisobot(y, pA, s, te.age_group, "A: ThresholdOptimizer")
print("Guruh chegaralari (tasodifiy interpolyatsiya: p0*[>t0] + p1*[>t1]):")
for g, v in to.interpolated_thresholder_.interpolation_dict.items():
    print(f"   {g:<6} p0={float(v['p0']):.2f} t0={float(v['operation0'].threshold):.2f}   p1={float(v['p1']):.2f} t1={float(v['operation1'].threshold):.2f}")

# --- Yumshatish B: ko'lamni cheklash (18+ bemorlar), chegarani sezgirlik >= 0.90 uchun qayta tanlash ---
print("\n=== Yumshatish B: ko'lam cheklash — tizim faqat 19+ yoshdagi bemorlar uchun ===")
m = te.age >= 19; yB, sB = y[m], s[m]
tB = max(t for t in np.arange(0.05, 0.95, 0.01) if recall_score(yB, sB >= t) >= 0.90)
pB = (sB >= tB).astype(int)
print(f"chiqarilgan tasvirlar: {(~m).sum()} ({(~m).mean():.1%}), yangi chegara {tB:.2f}, sezgirlik={recall_score(yB, pB):.3f}, aniqlik={precision_score(yB, pB):.3f}")
gapB = hisobot(yB, pB, sB, te.age_group[m].reset_index(drop=True), "B: 19+ ko'lam")

print("\n=== Taqqoslash ===")
print(f"{'variant':<28} {'TPR farqi (yosh)':>16} {'sezgirlik':>10} {'aniqlik':>8}  SLO")
for nom, g, pp, yy in (("Dastlabki", gap_age, p, y), ("A: ThresholdOptimizer", gapA, pA, y), ("B: 19+ ko'lam cheklash", gapB, pB, yB)):
    print(f"{nom:<28} {g:>16.3f} {recall_score(yy, pp):>10.3f} {precision_score(yy, pp):>8.3f}  {'OK' if g <= SLO_GAP else 'BUZILDI'}")
print("\nQaror (ADR uchun): B yo'li halolroq — bolalar uchun ma'lumot yetarli emas; model kartasida «mo'ljallanmagan foydalanish: 18 yoshgacha».")
print("A yo'li esa kam ma'lumotli guruhga sun'iy past chegara qo'yib, yolg'on ijobiylarni oshiradi.")
