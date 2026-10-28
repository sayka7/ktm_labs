"""A7 — 2-qadam. Model o'qitish, eksperiment jurnali, model reestri va promote darvozasi

Uchta eksperiment (logreg, 2 belgili zaif baseline, HistGradientBoosting) o'qitiladi, chegara validatsiyada sezgirlik >= 0.90 bo'ladigan qilib tanlanadi. Har model /tmp/triage/registry ichiga pickle + SHA-256 xesh bilan JSON reestrga «staging» sifatida yoziladi. `promote` darvozasi SLO (AUC >= 0.75, recall >= 0.90, jins bo'yicha TPR farqi <= 0.05) tekshiradi va joriy production'dan yomon bo'lsa almashtirmaydi. O'zgartirib ko'ring: SLO qiymatlari, yangi eksperiment, xeshni buzish.

Kutilgan natija: Jadval: v1 logreg -> production, v2 -> rejected (auc < 0.75), v3 hgb -> staging (v1 yaxshiroq); reestr holati va butunlik tekshiruvi MOS.
"""
# 2-qadam. Model o'qitish, eksperiment jurnali, oddiy MODEL REESTRI va «promote» darvozasi
# MLflow/Model Registry o'rniga JSON fayl: har versiya uchun metrika, artefakt xeshi va bosqich (stage).
# Model reestrga "staging" sifatida kiradi; SLO darvozasidan o'tsa — "production" ga ko'tariladi.
import json, os, hashlib, pickle, time
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, recall_score, precision_score, brier_score_loss

FEATS = ["opacity", "lung_area", "ct_ratio", "texture", "intensity", "edge_density"]

def sintetik_bemorlar(n_bemor=1500, seed=0):
    """1-qadamdagi generator: bemor-darajali sintetik manifest (har qadam mustaqil bo'lishi uchun qayta yaratiladi)."""
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
    """Bemor bo'yicha train/val/test (1-qadam bilan bir xil)."""
    tr, rest = next(GroupShuffleSplit(1, test_size=0.3, random_state=42).split(df, groups=df.patient_id))
    df = df.copy(); df["split"] = "train"; df.iloc[rest, df.columns.get_loc("split")] = "test"
    df.loc[(df.split == "test") & (df.patient_id % 2 == 0), "split"] = "val"
    return df

df = bolish(sintetik_bemorlar(3000))
tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
Xtr, ytr, Xva, yva, Xte, yte = tr[FEATS].values, tr.finding.values, va[FEATS].values, va.finding.values, te[FEATS].values, te.finding.values

# --- SLO (loyiha pasportidan): sezgirlik >= 0.90, AUC >= 0.75, jins bo'yicha TPR farqi <= 0.05 ---
SLO = {"recall_min": 0.90, "auc_min": 0.75, "tpr_gap_max": 0.05}

def chegara_tanla(p, y, recall_min):
    """Validatsiyada sezgirlik >= recall_min bo'ladigan ENG KATTA chegarani tanlaydi (aniqlikni maksimal saqlaydi)."""
    for t in np.arange(0.95, 0.01, -0.01):
        if recall_score(y, p >= t) >= recall_min:
            return round(float(t), 2)
    return 0.01

def baholash(model, X, y, meta, t):
    p = model.predict_proba(X)[:, 1]; pred = (p >= t).astype(int)
    tpr = {s: recall_score(y[meta.sex.values == s], pred[meta.sex.values == s]) for s in ("M", "F")}
    return dict(auc=round(roc_auc_score(y, p), 3), recall=round(recall_score(y, pred), 3),
                precision=round(precision_score(y, pred), 3), brier=round(brier_score_loss(y, p), 3),
                tpr_gap_sex=round(abs(tpr["M"] - tpr["F"]), 3))

# --- Reestr: /tmp/triage/registry/registry.json ---
REG_DIR = "/tmp/triage/registry"; os.makedirs(REG_DIR, exist_ok=True)
REG = os.path.join(REG_DIR, "registry.json")
reestr = json.load(open(REG)) if os.path.exists(REG) else {"name": "cxr-triage", "versions": []}

def royxatga_ol(model, nom, params, t, m_val, m_test):
    """Artefaktni saqlaydi, SHA-256 xeshini hisoblaydi va reestrga 'staging' bosqichida yozadi."""
    v = len(reestr["versions"]) + 1
    yol = os.path.join(REG_DIR, f"triage-v{v}.pkl")
    with open(yol, "wb") as f: pickle.dump(model, f)
    xesh = hashlib.sha256(open(yol, "rb").read()).hexdigest()[:16]
    yoz = dict(version=v, name=nom, params=params, threshold=t, val=m_val, test=m_test,
               artifact=yol, sha256=xesh, created=time.strftime("%Y-%m-%d %H:%M:%S"), stage="staging")
    reestr["versions"].append(yoz); return yoz

def promote(yoz):
    """Darvoza: SLO shartlari VALIDATSIYA metrikalarida bajarilsa -> production, aks holda rad etiladi."""
    m = yoz["val"]; sabab = []
    if m["recall"] < SLO["recall_min"]: sabab.append(f"recall {m['recall']} < {SLO['recall_min']}")
    if m["auc"] < SLO["auc_min"]: sabab.append(f"auc {m['auc']} < {SLO['auc_min']}")
    if m["tpr_gap_sex"] > SLO["tpr_gap_max"]: sabab.append(f"TPR farqi {m['tpr_gap_sex']} > {SLO['tpr_gap_max']}")
    if sabab:
        yoz["stage"] = "rejected"; yoz["reason"] = "; ".join(sabab); return False
    joriy = [v for v in reestr["versions"] if v["stage"] == "production"]
    if joriy and m["auc"] < joriy[-1]["val"]["auc"]:    # joriy production yaxshiroq bo'lsa — almashtirilmaydi
        yoz["stage"] = "staging"; yoz["reason"] = f"v{joriy[-1]['version']} val AUC {joriy[-1]['val']['auc']} yuqoriroq"; return False
    for old in joriy:                                   # faqat bitta production bo'ladi
        old["stage"] = "archived"
    yoz["stage"] = "production"; return True

# --- 3 ta eksperiment (rubrika: kamida 3 ta taqqoslash) ---
eksperimentlar = [
    ("logreg",        dict(C=1.0),                        make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500))),
    ("logreg-2feat",  dict(C=1.0, feats=2),               None),   # ataylab zaif: faqat 2 belgi
    ("hgb",           dict(max_iter=150, lr=0.05, depth=3), HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=3, random_state=0)),
]
print("=== Eksperiment jurnali ===")
print(f"{'v':>2} {'model':<13} {'chegara':>7} {'valAUC':>7} {'valRec':>7} {'valPrec':>7} {'TPRfarq':>7} {'testAUC':>7} {'stage':<10} sabab")
for nom, params, model in eksperimentlar:
    if model is None:   # zaif baseline: 2 ta belgi bilan
        from sklearn.compose import ColumnTransformer
        model = make_pipeline(ColumnTransformer([("sel", "passthrough", [4, 5])]), StandardScaler(), LogisticRegression(max_iter=500))
    t0 = time.perf_counter(); model.fit(Xtr, ytr); fit_s = time.perf_counter() - t0
    p_val = model.predict_proba(Xva)[:, 1]
    t = chegara_tanla(p_val, yva, SLO["recall_min"])
    m_val, m_test = baholash(model, Xva, yva, va, t), baholash(model, Xte, yte, te, t)
    yoz = royxatga_ol(model, nom, {**params, "fit_s": round(fit_s, 2)}, t, m_val, m_test)
    ok = promote(yoz)
    print(f"{yoz['version']:>2} {nom:<13} {t:>7} {m_val['auc']:>7} {m_val['recall']:>7} {m_val['precision']:>7} "
          f"{m_val['tpr_gap_sex']:>7} {m_test['auc']:>7} {yoz['stage']:<10} {yoz.get('reason', '')}")

json.dump(reestr, open(REG, "w"), indent=1)
prod = [v for v in reestr["versions"] if v["stage"] == "production"]
print("\n=== Reestr holati ===")
for v in reestr["versions"]:
    print(f"v{v['version']} {v['name']:<13} stage={v['stage']:<10} sha256={v['sha256']} chegara={v['threshold']}")
print("\nProduction versiya:", f"v{prod[-1]['version']} ({prod[-1]['name']})" if prod else "YO'Q")
print("Reestr fayli:", REG, "| artefaktlar:", sorted(os.listdir(REG_DIR)))

# Artefakt butunligi: xesh mos kelmasa — yuklash rad etiladi (runbook'dagi tekshiruv)
if prod:
    p = prod[-1]; hozir = hashlib.sha256(open(p["artifact"], "rb").read()).hexdigest()[:16]
    print("Butunlik tekshiruvi:", "MOS" if hozir == p["sha256"] else "BUZILGAN")
    print(f"Production test metrikalari: {p['test']}")
