"""A7 — 7-qadam. Loyiha pasporti va model kartasi avtomatik generatsiyasi, rubrika bo'yicha o'z-o'zini baholash

Butun ketma-ketlik qisqa shaklda qayta bajariladi va barcha o'lchovlar (sizish, AUC, sezgirlik, chegara, parity, p50/p99, PSI, TPR farqlari, ko'lamdan tashqari guruh) bitta `O` lug'atiga yig'iladi. Shundan Markdown pasport + model kartasi (Mitchell va boshq. formati) va 10 mezonli rubrika jadvali avtomatik tuziladi: avtomatik tekshiriladigan mezonlar SLO bo'yicha to'liq/yarim ball oladi, qolganlari «qo'lda» belgilanadi. O'zgartirib ko'ring: SLO qiymatlari, model o'lchami, drift kuchi — hujjat o'zi qayta yoziladi.

Kutilgan natija: Markdown hujjat stdout'da va /tmp/triage/docs/model_card.md da; test AUC ~0.85, sezgirlik ~0.91, TPR farqlari SLO ichida, drift ALERT ishlaydi; jami ~84/100 (qo'lda mezonlar yarim ball).
"""
# 7-qadam. Yakuniy «loyiha pasporti» + model kartasi avtomatik generatsiyasi va rubrika bo'yicha o'z-o'zini baholash
# Butun ketma-ketlik qisqa shaklda qayta bajariladi (ma'lumot -> model -> ONNX -> drift -> adolatlilik),
# o'lchangan RAQAMLAR SLO bilan solishtiriladi va Markdown hujjat (docs/model_card.md) yoziladi.
# Hujjatdagi har raqam koddan olinadi — «qo'lda yozilgan» metrika yo'q.
import os, time, json, hashlib, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch, torch.nn as nn, onnxruntime as ort
from scipy.stats import ks_2samp
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from fairlearn.metrics import MetricFrame, true_positive_rate
torch.manual_seed(0)

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

def psi(ref, cur, bins=10):
    e = np.quantile(ref, np.linspace(0, 1, bins + 1)); e[0], e[-1] = -np.inf, np.inf
    r = np.histogram(ref, e)[0] / len(ref) + 1e-6; c = np.histogram(cur, e)[0] / len(cur) + 1e-6
    return float(np.sum((c - r) * np.log(c / r)))

SLO = {"auc": 0.75, "recall": 0.90, "tpr_gap": 0.05, "p99_ms": 800, "rps": 20, "psi_alert": 0.2}
O = {}   # o'lchovlar — pasportga kiradi

# TODO 1: ma'lumot — bemor bo'yicha train/val/test, O["leak"], n_img, n_pat, pos, data_sha; ko'lam = 19+
# TODO 2: MLP 6-32-16-1 o'qitish, validatsiyada sezgirlik >= 0.90 chegara; O: threshold, val_auc, test_auc, recall, precision
# TODO 3: ONNX eksport, O["onnx_sha"], parity, batch 1 p50/p99 (ms) va rps
# TODO 4: drift taqlidi (opacity*0.7-0.9, intensity+0.9): psi_normal, psi_drift, psi_feat_max, ks_p_drift
# TODO 5: MetricFrame bilan jins va yosh(19+) TPR farqlari; 0–18 guruh uchun kids_tpr, kids_n
raise NotImplementedError("TODO 1-5: o'lchovlarni O lug'atiga yig'ing")

# TODO 6: 10 mezonli rubrika: (nom, dalil, ok, ball); ok=None -> «qo'lda» yarim ball, ok=False -> yarim, True -> to'liq
rubrika = []
jami, satrlar = 0, []

# TODO 7: Markdown: sarlavha, "## 1. Pasport" jadvali, "## 2. Model kartasi", "## 3. Rubrika bo'yicha o'z-o'zini baholash"
#         jadvali va "**Jami: N/100**"; /tmp/triage/docs/model_card.md va measurements.json ga yozing
md = []
print("=== Loyiha pasporti va model kartasi ===")
print("\n".join(md))
