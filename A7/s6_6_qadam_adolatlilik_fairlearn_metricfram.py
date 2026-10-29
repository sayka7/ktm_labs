"""A7 — 6-qadam. Adolatlilik: fairlearn MetricFrame hisoboti va yumshatish

Test to'plamida jins va yosh guruhlari bo'yicha n, TPR, FPR, tanlov ulushi va AUC `MetricFrame` bilan chiqariladi, TPR farqi SLO (<= 0.05) bilan solishtiriladi, bootstrap bilan 95% ishonch oralig'i hisoblanadi. Ikki yumshatish yo'li taqqoslanadi: (A) `ThresholdOptimizer` guruhga xos chegaralar, (B) ko'lamni 19+ bilan cheklash va chegarani qayta tanlash. O'zgartirib ko'ring: cheklov turi, chegara, bolalar shovqin koeffitsienti generator ichida.

Kutilgan natija: Jins bo'yicha farq ~0.01 (OK), yosh bo'yicha ~0.18 (BUZILDI, 0-18 TPR ~0.70, keng ishonch oralig'i); A: farq ~0.02, lekin sezgirlik 0.77; B: farq ~0.03, sezgirlik 0.90.
"""
# 6-qadam. Adolatlilik: fairlearn MetricFrame bilan kesimlar hisoboti, bootstrap ishonch oralig'i va yumshatish
# (A) guruhga xos chegara (ThresholdOptimizer) va (B) ko'lamni cheklash (0–18 yosh guruhini chiqarish).
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
    """1-qadamdagi generator (berilgan, o'zgartirmang)."""
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

# TODO 1: MetricFrame (n, tpr, fpr, sel) + guruh AUC jadvali; TPR farqi (max-min) va SLO bajarildi/buzildi
def hisobot(y, pred, score, sens, nom):
    raise NotImplementedError("TODO 1: kesim hisoboti")

# TODO 2: har guruhning musbat namunalari ustida bootstrap (n=300) bilan TPR 95% oralig'i
def bootstrap_tpr(y, pred, sens, n=300, seed=0):
    raise NotImplementedError("TODO 2: bootstrap")

y, s, p = te.finding.values, te.score.values, te.pred.values
print("=== Dastlabki model (chegara 0.30) ===")
gap_sex = hisobot(y, p, s, te.sex, "Dastlabki")
gap_age = hisobot(y, p, s, te.age_group, "Dastlabki")

print("\n=== Yumshatish A: guruhga xos chegara (ThresholdOptimizer, true_positive_rate_parity) ===")
# TODO 3: prefit modeldan ThresholdOptimizer, train'da age_group bilan fit, testda bashorat va hisobot
gapA, pA = None, None

print("\n=== Yumshatish B: ko'lam cheklash — tizim faqat 19+ yoshdagi bemorlar uchun ===")
# TODO 4: age>=19 niqobi, sezgirlik >= 0.90 bo'ladigan eng katta chegara, hisobot
gapB, pB = None, None

print("\n=== Taqqoslash ===")
# TODO 5: uch variant jadvali: TPR farqi (yosh), sezgirlik, aniqlik, SLO; ADR uchun qaror
