"""A3 — 3-qadam. PyTorch'da INT8 dinamik kvantlash (Linear)

MLP (Linear og'ir model) `torch.ao.quantization.quantize_dynamic` bilan INT8 ga o'tkaziladi: kvantlash formulasi q = round(w/s) qo'lda tekshiriladi, hajm (state_dict baytlari), aniqlik va batch 1/256 kechikishi FP32 bilan solishtiriladi. Sandbox CPU'da AVX yo'qligi sababli `qnnpack` dvigateli majburan tanlanadi — real serverda bu qator kerak emas.

Kutilgan natija: Hajm ~3,8 marta kichik (335 → 89 KB), aniqlik o'zgarmaydi (±0,3 p.p.), tezlik bu CPU'da 0,7–1,0x (VNNI yo'q).
"""
# 3-qadam. PyTorch'da dinamik INT8 kvantlash (Linear qatlamlar): hajm, aniqlik, tezlik
import warnings; warnings.filterwarnings("ignore")
import io, time, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.ao.quantization import quantize_dynamic
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(0); torch.set_num_threads(2); torch.backends.nnpack.set_flags(False)
torch.backends.quantized.engine = "qnnpack"   # sandbox CPU'da AVX/VNNI yo'q (o'zgartirmang)

X, y = load_digits(return_X_y=True)
X = (X / 16.0).astype(np.float32)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
Xtr, Xte, ytr, yte = map(torch.tensor, (Xtr, Xte, ytr, yte))

# Berilgan MLP va uni o'qitish (o'zgartirmang)
model = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10))
opt = torch.optim.Adam(model.parameters(), 2e-3); lossf = nn.CrossEntropyLoss()
for ep in range(3):
    model.train(); perm = torch.randperm(len(Xtr))
    for i in range(0, len(Xtr), 64):
        idx = perm[i:i + 64]; opt.zero_grad(); lossf(model(Xtr[idx]), ytr[idx]).backward(); opt.step()
model.eval()

def aniqlik(m):
    with torch.inference_mode():
        return (m(Xte).argmax(1) == yte).float().mean().item() * 100

def hajm_kb(m):
    # TODO 3: state_dict ni xotiradagi buferga saqlab, hajmini KB da qaytaring
    raise NotImplementedError("TODO 3: hajm_kb")

print("\n=== PyTorch INT8 dinamik kvantlash ===")
# TODO 1: Linear qatlamlarni qint8 ga dinamik kvantlang va tuzilmani chiqaring
qmodel = None
# TODO 2: 1-qatlam vazni uchun s = max|w|/127, q = round(w/s) ni qo'lda hisoblab, tiklash xatosini chiqaring
# TODO 4: FP32 va INT8 uchun hajm, aniqlik, p50 (batch 1 va 256) jadvalini tuzing va "Hajm ... marta" ni chiqaring
raise NotImplementedError("TODO 1-4")
