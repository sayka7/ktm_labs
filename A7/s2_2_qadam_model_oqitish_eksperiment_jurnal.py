"""A7 — 2-qadam. Model o'qitish, eksperiment jurnali, model reestri va promote darvozasi

Uchta eksperiment (logreg, 2 belgili zaif baseline, HistGradientBoosting) o'qitiladi, chegara validatsiyada sezgirlik >= 0.90 bo'ladigan qilib tanlanadi. Har model /tmp/triage/registry ichiga pickle + SHA-256 xesh bilan JSON reestrga «staging» sifatida yoziladi. `promote` darvozasi SLO (AUC >= 0.75, recall >= 0.90, jins bo'yicha TPR farqi <= 0.05) tekshiradi va joriy production'dan yomon bo'lsa almashtirmaydi. O'zgartirib ko'ring: SLO qiymatlari, yangi eksperiment, xeshni buzish.

Kutilgan natija: Jadval: v1 logreg -> production, v2 -> rejected (auc < 0.75), v3 hgb -> staging (v1 yaxshiroq); reestr holati va butunlik tekshiruvi MOS.
"""
# 2-qadam. Model o'qitish, eksperiment jurnali, oddiy MODEL REESTRI va «promote» darvozasi
# MLflow o'rniga JSON fayl: har versiya uchun metrika, artefakt xeshi va bosqich (stage).
import json, os, hashlib, pickle, time
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, recall_score, precision_score, brier_score_loss

FEATS = ["opacity", "lung_area", "ct_ratio", "texture", "intensity", "edge_density"]

def sintetik_bemorlar(n_bemor=1500, seed=0):
    """1-qadamdagi generator (berilgan)."""
    rng = np.random.default_rng(seed); rows = []
    for pid in range(n_bemor):
        age = int(np.clip(rng.normal(50, 18), 1, 95)); sex = "M" if rng.random() < 0.56 else "F"
        z = rng.normal(0, 1) + 0.02 * (age - 50)
        for k in range(int(rng.integers(1, 4))):
            e = rng.normal(0, 2.2 if age < 18 else 1.0, 6)
            f = [0.9*z+e[0], -0.6*z+e[1], 0.5*z+e[2], 0.7*z+e[3], 0.2*z+e[4], 0.4*z+e[5]]
            rows.append([f"{pid:08d}_{k:03d}.png", pid, age, sex, *np.round(f, 4), int(z + rng.normal(0, 0.6) > 0.3)])
    return pd.DataFrame(rows, columns=["image_id", "patient_id", "age", "sex", *FEATS, "finding"])

def bolish(df):
    """Bemor bo'yicha train/val/test (berilgan)."""
    tr, rest = next(GroupShuffleSplit(1, test_size=0.3, random_state=42).split(df, groups=df.patient_id))
    df = df.copy(); df["split"] = "train"; df.iloc[rest, df.columns.get_loc("split")] = "test"
    df.loc[(df.split == "test") & (df.patient_id % 2 == 0), "split"] = "val"
    return df

df = bolish(sintetik_bemorlar(3000))
tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
Xtr, ytr, Xva, yva, Xte, yte = tr[FEATS].values, tr.finding.values, va[FEATS].values, va.finding.values, te[FEATS].values, te.finding.values
SLO = {"recall_min": 0.90, "auc_min": 0.75, "tpr_gap_max": 0.05}

# TODO 1: validatsiyada recall >= recall_min bo'ladigan ENG KATTA chegarani toping
def chegara_tanla(p, y, recall_min):
    raise NotImplementedError("TODO 1: chegara tanlash")

# TODO 2: auc, recall, precision, brier va jins (M/F) bo'yicha TPR farqini hisoblang
def baholash(model, X, y, meta, t):
    raise NotImplementedError("TODO 2: metrikalar")

REG_DIR = "/tmp/triage/registry"; os.makedirs(REG_DIR, exist_ok=True)
REG = os.path.join(REG_DIR, "registry.json")
reestr = json.load(open(REG)) if os.path.exists(REG) else {"name": "cxr-triage", "versions": []}

# TODO 3: modelni triage-v{N}.pkl ga pickle qiling, SHA-256 (16 belgi) hisoblang, reestrga stage="staging" bilan yozing
def royxatga_ol(model, nom, params, t, m_val, m_test):
    raise NotImplementedError("TODO 3: reestrga yozish")

# TODO 4: SLO buzilsa -> "rejected" (+reason); joriy production val AUC yuqoriroq bo'lsa -> staging qoladi;
#         aks holda eskisi "archived", yangisi "production"
def promote(yoz):
    raise NotImplementedError("TODO 4: promote darvozasi")

eksperimentlar = [
    ("logreg",       dict(C=1.0),                          make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500))),
    ("logreg-2feat", dict(C=1.0, feats=2),                 make_pipeline(ColumnTransformer([("sel", "passthrough", [4, 5])]), StandardScaler(), LogisticRegression(max_iter=500))),
    ("hgb",          dict(max_iter=150, lr=0.05, depth=3), HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=3, random_state=0)),
]
print("=== Eksperiment jurnali ===")
for nom, params, model in eksperimentlar:
    # TODO 5: o'qiting, chegarani VALIDATSIYADA tanlang, val/test metrikalarini hisoblang, reestrga yozing, promote qiling
    raise NotImplementedError("TODO 5: eksperiment sikli")

json.dump(reestr, open(REG, "w"), indent=1)
prod = [v for v in reestr["versions"] if v["stage"] == "production"]
print("\n=== Reestr holati ===")
# TODO 6: production artefakt xeshini qayta hisoblab reestrdagi bilan solishtiring -> "Butunlik tekshiruvi: MOS/BUZILGAN"
