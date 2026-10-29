"""A7 — 5-qadam. Monitoring: PSI/KS bilan drift aniqlash va ogohlantirish mantiqi

Validatsiya to'plami mos yozuv bo'ladi. 8 kunlik ishlab chiqarish oqimi simulyatsiya qilinadi: 4-kundan «yangi apparat» (opaklik kontrasti −30% va siljish, yorqinlik oshishi) kuchayib boradi. Har kun uchun belgilar va model ballari PSI, KS-test p-qiymati, HIGH ulushi hisoblanib, darajali qoida (OK/WARN/ALERT) ishlaydi; sezgirlik yorliq bilan qayta hisoblanib, driftning haqiqiy zarari ko'rsatiladi. O'zgartirib ko'ring: siljish kuchi, qoidalar chegarasi, chelaklar soni.

Kutilgan natija: 1–4 kun OK, 5-kundan ALERT (intensity/opacity PSI > 0.3), 8-kunda ball PSI ~0.23, KS p ~1e-14, sezgirlik 0.86 -> 0.68; kun × belgi PSI jadvali va grafik.
"""
# 5-qadam. Monitoring: PSI va KS-test bilan drift aniqlash, ogohlantirish mantiqi
# Mos yozuv (reference) — validatsiya to'plamidagi belgilar va model ballari.
# Simulyatsiya: yangi rentgen apparati -> 'opacity' va 'intensity' siljiydi, drift kun sayin kuchayadi.
import numpy as np, pandas as pd, json, warnings
warnings.filterwarnings("ignore")
from scipy.stats import ks_2samp
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, recall_score

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

# TODO 1: Population Stability Index — mos yozuv kvantillari bo'yicha chelaklar, sum((c-r)*ln(c/r))
def psi(ref, cur, bins=10):
    raise NotImplementedError("TODO 1: PSI")

df = sintetik_bemorlar(3000)
tr_i, rest = next(GroupShuffleSplit(1, test_size=0.3, random_state=42).split(df, groups=df.patient_id))
tr, ref = df.iloc[tr_i], df.iloc[rest]
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(tr[FEATS], tr.finding)
ref_scores = model.predict_proba(ref[FEATS])[:, 1]
THRESHOLD = 0.30
QOIDA = {"psi_warn": 0.10, "psi_alert": 0.20, "ks_p": 0.01, "pos_rate_rel": 0.25}

def yangi_apparat(df, kuch):
    """Boshqa apparat taqlidi (berilgan): kuch 0..1."""
    cur = df.copy()
    cur["opacity"] = cur["opacity"] * (1 - 0.30 * kuch) - 0.9 * kuch
    cur["intensity"] = cur["intensity"] + 0.9 * kuch
    cur["texture"] = cur["texture"] - 0.5 * kuch
    return cur

jurnal, psi_jadval = [], []
pos_ref = float((ref_scores >= THRESHOLD).mean())
print("\n=== Kunlik monitoring oynalari ===")
for kun in range(1, 9):
    kuch = 0.0 if kun <= 3 else min(1.0, (kun - 3) / 3)
    cur = yangi_apparat(sintetik_bemorlar(250, seed=100 + kun), kuch)
    # TODO 2: har belgi va model balli uchun PSI, ball uchun KS p-qiymati, HIGH ulushi va sezgirlikni hisoblang
    # TODO 3: darajali qoida: PSI>=0.2 -> ALERT, >=0.1 -> WARN; KS p<0.01 va HIGH ulushi ±25% darajani oshiradi
    # TODO 4: kunlik yozuvni jurnal ga qo'shing va satrni chiqaring
    raise NotImplementedError("TODO 2-4: kunlik monitoring")

print("\n=== Ogohlantirish xulosasi ===")
# TODO 5: birinchi ALERT kuni va sababi, eng ko'p siljigan belgi, sezgirlik 1-kun -> 8-kun
print("\n=== PSI belgilar bo'yicha (kun × belgi) ===")
json.dump(jurnal, open("/tmp/drift_log.json", "w"), indent=1)
