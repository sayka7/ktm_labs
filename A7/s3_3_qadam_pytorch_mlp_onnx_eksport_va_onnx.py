"""A7 — 3-qadam. PyTorch MLP -> ONNX eksport va onnxruntime bilan inferens tekshiruvi

Normalizatsiyasi grafga «tikilgan» kichik MLP 30 epoxa CPU'da o'qitiladi, dinamik batch bilan ONNX'ga eksport qilinadi, `onnx.checker` va PyTorch–ONNX parity (maks. farq < 1e-4) tekshiriladi, ikkala artefakt xeshi chiqariladi. Keyin PyTorch eager va ONNX Runtime kechikishi batch 1 va 64 uchun p50/p99 bilan taqqoslanadi. O'zgartirib ko'ring: qatlam o'lchamlari, `opset_version`, batch hajmi.

Kutilgan natija: Test AUC ~0.84; parity farqi ~1e-7 (MOS); ONNX Runtime batch 1 p50 ~10 µs, PyTorch ~40 µs; triage.onnx ~4 KB.
"""
# 3-qadam. PyTorch MLP -> ONNX eksport -> onnxruntime bilan inferens tekshiruvi va kechikish
# Sandbox'da 6 ta tasvir belgisi ustida kichik MLP. Eksportdan keyin parity va artefakt xeshi tekshiriladi.
import os, time, hashlib, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch, torch.nn as nn, onnx, onnxruntime as ort
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
torch.manual_seed(0); torch.set_num_threads(2)

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

df = sintetik_bemorlar(3000)
tr_i, te_i = next(GroupShuffleSplit(1, test_size=0.3, random_state=42).split(df, groups=df.patient_id))
tr, te = df.iloc[tr_i], df.iloc[te_i]
mu, sd = tr[FEATS].mean().values, tr[FEATS].std().values

# TODO 1: mu/sd ni register_buffer bilan saqlang, 6->32->16->1 MLP quring, forward'da (x-mu)/sd ni qo'llang
class TriageMLP(nn.Module):
    def __init__(self, mu, sd):
        super().__init__()
        raise NotImplementedError("TODO 1: model tuzilmasi")

model = TriageMLP(mu, sd)
Xtr = torch.tensor(tr[FEATS].values, dtype=torch.float32); ytr = torch.tensor(tr.finding.values, dtype=torch.float32)
Xte = torch.tensor(te[FEATS].values, dtype=torch.float32); yte = te.finding.values
print("=== O'qitish (CPU, 30 epoxa, mini-batch 128) ===")
# TODO 2: BCEWithLogitsLoss(pos_weight) + AdamW(lr=3e-3) bilan 30 epoxa mini-batch o'qitish, har 10 epoxada test AUC
model.eval()

os.makedirs("/tmp/triage/models", exist_ok=True)
ONNX = "/tmp/triage/models/triage.onnx"; PT = "/tmp/triage/models/triage.pt"
# TODO 3: state_dict saqlang; torch.onnx.export (input "features", output "logit", dinamik batch, opset 17);
#         onnx.checker bilan tekshiring va InferenceSession yarating
sess = None

print("\n=== ONNX eksport tekshiruvi ===")
# TODO 4: PyTorch va ONNX chiqishlari maks. farqini hisoblang -> "MOS (parity OK)" agar < 1e-4; ikkala AUC va fayl xeshlari

def olchov(fn, n=300):
    # TODO 5: 20 marta isitish, so'ng n marta o'lchab p50/p99 (mikrosekund) qaytaring
    raise NotImplementedError("TODO 5: kechikish o'lchovi")

print("\n=== Inferens kechikishi (mikrosekund) ===")
# TODO 6: batch 1 va 64 uchun PyTorch eager va ONNX Runtime p50/p99 jadvali; meta.json saqlang
