"""A7 — 3-qadam. PyTorch MLP -> ONNX eksport va onnxruntime bilan inferens tekshiruvi

Normalizatsiyasi grafga «tikilgan» kichik MLP 30 epoxa CPU'da o'qitiladi, dinamik batch bilan ONNX'ga eksport qilinadi, `onnx.checker` va PyTorch–ONNX parity (maks. farq < 1e-4) tekshiriladi, ikkala artefakt xeshi chiqariladi. Keyin PyTorch eager va ONNX Runtime kechikishi batch 1 va 64 uchun p50/p99 bilan taqqoslanadi. O'zgartirib ko'ring: qatlam o'lchamlari, `opset_version`, batch hajmi.

Kutilgan natija: Test AUC ~0.84; parity farqi ~1e-7 (MOS); ONNX Runtime batch 1 p50 ~10 µs, PyTorch ~40 µs; triage.onnx ~4 KB.
"""
# 3-qadam. PyTorch MLP -> ONNX eksport -> onnxruntime bilan inferens tekshiruvi va kechikish
# Haqiqiy loyihada DenseNet-121 tasvirdan ishlaydi; sandbox'da 6 ta tasvir belgisi ustida kichik MLP.
# Muhim: eksportdan keyin ONNX va PyTorch chiqishlari mos kelishi (parity) va artefakt xeshi tekshiriladi.
import os, time, hashlib, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch, torch.nn as nn, onnx, onnxruntime as ort
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
torch.manual_seed(0); torch.set_num_threads(2)

FEATS = ["opacity", "lung_area", "ct_ratio", "texture", "intensity", "edge_density"]

def sintetik_bemorlar(n_bemor=1500, seed=0):
    """1-qadamdagi generator: bemor-darajali sintetik manifest."""
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
mu, sd = tr[FEATS].mean().values, tr[FEATS].std().values          # normalizatsiya — modelga «tikiladi»

class TriageMLP(nn.Module):
    """Normalizatsiya + 2 qatlamli MLP. Chiqish — logit (sigmoid serverda yoki grafda)."""
    def __init__(self, mu, sd):
        super().__init__()
        self.register_buffer("mu", torch.tensor(mu, dtype=torch.float32))
        self.register_buffer("sd", torch.tensor(sd, dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x):
        return self.net((x - self.mu) / self.sd)

model = TriageMLP(mu, sd)
Xtr = torch.tensor(tr[FEATS].values, dtype=torch.float32); ytr = torch.tensor(tr.finding.values, dtype=torch.float32)
Xte = torch.tensor(te[FEATS].values, dtype=torch.float32); yte = te.finding.values
pos_w = torch.tensor([(1 - ytr.mean()) / ytr.mean()])
lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w); opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
print("=== O'qitish (CPU, 30 epoxa, mini-batch 128) ===")
t0 = time.perf_counter()
for ep in range(30):
    model.train(); perm = torch.randperm(len(Xtr))
    for i in range(0, len(Xtr), 128):
        b = perm[i:i+128]; opt.zero_grad(); loss = lossf(model(Xtr[b]).squeeze(1), ytr[b]); loss.backward(); opt.step()
    if ep % 10 == 9:
        model.eval()
        with torch.no_grad(): auc = roc_auc_score(yte, torch.sigmoid(model(Xte)).squeeze(1).numpy())
        print(f"epoxa {ep+1:2d}  loss={loss.item():.3f}  test AUC={auc:.3f}")
print(f"o'qitish vaqti: {time.perf_counter()-t0:.1f} s")
model.eval()

# --- ONNX eksport (dinamik batch) ---
os.makedirs("/tmp/triage/models", exist_ok=True)
ONNX = "/tmp/triage/models/triage.onnx"; PT = "/tmp/triage/models/triage.pt"
torch.save(model.state_dict(), PT)
dummy = torch.randn(1, 6)
torch.onnx.export(model, dummy, ONNX, input_names=["features"], output_names=["logit"],
                  dynamic_axes={"features": {0: "batch"}, "logit": {0: "batch"}}, opset_version=17, dynamo=False)
onnx.checker.check_model(onnx.load(ONNX))              # graf tuzilmasi to'g'riligi
sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])

print("\n=== ONNX eksport tekshiruvi ===")
with torch.no_grad(): ref = model(Xte).numpy()
out = sess.run(None, {"features": Xte.numpy()})[0]
farq = float(np.abs(out - ref).max())
print(f"graf kirish: {[i.name for i in sess.get_inputs()]}, shakl: {sess.get_inputs()[0].shape}")
print(f"operatorlar: {sorted({n.op_type for n in onnx.load(ONNX).graph.node})}")
print(f"PyTorch va ONNX chiqishlari orasidagi maks. farq: {farq:.2e}  -> {'MOS (parity OK)' if farq < 1e-4 else 'NOMOS'}")
print(f"test AUC (PyTorch) = {roc_auc_score(yte, ref):.4f}   test AUC (ONNX) = {roc_auc_score(yte, out):.4f}")
for yol in (PT, ONNX):
    print(f"{os.path.basename(yol):<12} {os.path.getsize(yol):>6} bayt  sha256={hashlib.sha256(open(yol,'rb').read()).hexdigest()[:16]}")

# --- Kechikish: PyTorch vs ONNX Runtime, batch 1 va 64 (p50/p99) ---
def olchov(fn, n=300):
    for _ in range(20): fn()                              # isitish (birinchi chaqiruvlar sekin)
    t = []
    for _ in range(n):
        s = time.perf_counter(); fn(); t.append((time.perf_counter() - s) * 1e6)
    return np.percentile(t, 50), np.percentile(t, 99)
print("\n=== Inferens kechikishi (mikrosekund) ===")
print(f"{'variant':<22} {'batch':>5} {'p50':>9} {'p99':>9}")
for bs in (1, 64):
    xb = Xte[:bs]; xn = xb.numpy()
    with torch.no_grad():
        p50, p99 = olchov(lambda: model(xb))
    print(f"{'PyTorch eager':<22} {bs:>5} {p50:>9.1f} {p99:>9.1f}")
    p50, p99 = olchov(lambda: sess.run(None, {"features": xn}))
    print(f"{'ONNX Runtime CPU':<22} {bs:>5} {p50:>9.1f} {p99:>9.1f}")

# Model metadata — serverga va model kartasiga kerak
meta = {"model_version": "triage-mlp-1.0", "onnx": ONNX, "input": "features[batch,6]", "feats": FEATS,
        "test_auc": round(float(roc_auc_score(yte, out)), 4), "parity_max_diff": farq}
json.dump(meta, open("/tmp/triage/models/meta.json", "w"), indent=1)
print("\nSaqlandi:", sorted(os.listdir("/tmp/triage/models")))
