"""A3 — 1-qadam. Etalon: PyTorch FP32 o'qitish va ONNX eksport

Kichik CNN sklearn digits to'plamida 3 epoxa o'qitiladi, batch 1 kechikishi (p50/p95) o'lchanadi, so'ngra model `torch.onnx.export` bilan dinamik paket o'qi va opset 17 bilan eksport qilinadi. `onnx.checker` grafikni tekshiradi, ONNX Runtime chiqishi PyTorch bilan `np.allclose` orqali solishtiriladi. O'zgartirib ko'ring: epoxa sonini, `dynamic_axes` ni olib tashlab kirish shaklini, `do_constant_folding=False` da tugunlar sonini.

Kutilgan natija: 3 epoxadan keyin ~94 % aniqlik, PyTorch eager p50 ~0,15 ms; ONNX grafikda 10 tugun (BatchNorm Conv'ga birlashgan), maksimal farq ~1e-6, «MOS».
"""
# 1-qadam. Etalon: kichik CNN'ni PyTorch'da o'qitish, ONNX'ga eksport qilish va moslikni tekshirish
import warnings; warnings.filterwarnings("ignore")
import os, time, numpy as np, torch, torch.nn as nn, onnx, onnxruntime as ort
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(0); np.random.seed(0)
torch.set_num_threads(2); torch.backends.nnpack.set_flags(False)

# Berilgan ma'lumot (o'zgartirmang): sklearn digits, 8x8 rasm, 10 sinf
X, y = load_digits(return_X_y=True)
X = (X / 16.0).astype(np.float32).reshape(-1, 1, 8, 8)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
Xtr, Xte, ytr, yte = map(torch.tensor, (Xtr, Xte, ytr, yte))

# Berilgan model (o'zgartirmang)
class KichikCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.belgilar = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2))
        self.tasnif = nn.Sequential(nn.Flatten(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
    def forward(self, x):
        return self.tasnif(self.belgilar(x))

def oqit(model, epoxa=3, lr=3e-3):
    # TODO 1: Adam + CrossEntropyLoss bilan paket 64 da o'qiting, har epoxadan keyin test aniqligini chiqaring
    raise NotImplementedError("TODO 1: o'qitish sikli")

print("=== PyTorch FP32 etalon ===")
model = KichikCNN()
acc_fp32 = oqit(model)
x1 = Xte[:1]
# TODO 2: batch 1 kechikishini o'lchang (isinish + 300 takror) va p50/p95 ni ms da chiqaring
raise NotImplementedError("TODO 2: latency p50/p95")

print("\n=== ONNX eksport ===")
# TODO 3: torch.onnx.export — dinamik paket o'qi, opset 17, dynamo=False; so'ng onnx.checker bilan tekshiring
# TODO 4: tugunlar soni va operatorlar ro'yxatini chiqaring
# TODO 5: ORT InferenceSession chiqishini PyTorch bilan np.allclose orqali solishtiring -> 'MOS' / 'MOS EMAS'
