"""MT2 — 1-qadam. Etalon model va o'lchov metodikasi

Sintetik HAR ma'lumoti (3 o'q x 128 nuqta, 6 sinf, sensor shovqini bilan) yaratiladi va HARNet (depthwise-separable 1D CNN, ~8 ming parametr) 5 epoxa o'qitiladi. Asosiy e'tibor — **to'g'ri o'lchash**: taymer aniqligi, isinish (birinchi chaqiruv sekin), bir konfiguratsiyani 5 marta takrorlab p50/p95/p99/p99.9 va variatsiya koeffitsienti (CV). PyTorch eager va TorchScript (trace + freeze) solishtiriladi; kechikish gistogrammasi uzun dumni ko'rsatadi.

Kutilgan natija: Test aniqligi ~0,92; 1-chaqiruv ~1,5 ms, keyingilari ~0,55 ms; TorchScript p50 ~0,37 ms (eager ~0,56 ms), CV < 1% (taxminiy).
"""
# 1-qadam. Etalon model (1D CNN, akselerometr) va to'g'ri o'lchov metodikasi: isinish, takrorlar, foizlar, tarqalish
import warnings; warnings.filterwarnings("ignore")
import time, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

torch.manual_seed(0); np.random.seed(0)
torch.set_num_threads(2); torch.backends.nnpack.set_flags(False)

# --- Sintetik akselerometr (HAR) ma'lumoti: 3 o'q x 128 nuqta (50 Hz, 2,56 s), 6 harakat sinfi
SINFLAR = ["tinch", "yurish", "yugurish", "zina_yuqori", "zina_past", "yiqilish"]
def har_yarat(n=3000, seed=0):
    r = np.random.default_rng(seed); t = np.arange(128) / 50.0
    X = np.zeros((n, 3, 128), np.float32); y = r.integers(0, 6, n)
    for j in range(n):
        c, f, ph = y[j], r.uniform(1.6, 2.4), r.uniform(0, 2 * np.pi)
        amp = {0: 0.05, 1: 0.6, 2: 1.0, 3: 0.65, 4: 0.65, 5: 0.3}[c] * r.uniform(0.7, 1.3)
        if c == 2: f *= 1.3
        s = amp * np.sin(2 * np.pi * f * t + ph)
        X[j, 0] = s; X[j, 1] = 0.5 * amp * np.sin(2 * np.pi * f * t + ph + 1.2); X[j, 2] = 1.0 + 0.6 * amp * np.cos(2 * np.pi * 2 * f * t + ph)
        if c == 3: X[j, 2] += np.linspace(0, 0.4, 128)          # yuqoriga — vertikal tezlanish trendi
        if c == 4: X[j, 2] -= np.linspace(0, 0.4, 128)
        if c == 5: k = r.integers(20, 100); X[j, :, k:k + 6] += r.normal(3.0, 0.8, (3, 1)); X[j, 2, k + 6:] -= 1.0   # zarba, keyin yotish
        X[j] += r.normal(0, 0.6, (3, 128))                      # sensor shovqini
    return X, y

# --- Model: depthwise-separable 1D CNN (MobileNet g'oyasi, chekka qurilmalar uchun)
class HARNet(nn.Module):
    def __init__(self, w=32):
        super().__init__()
        def ds(ci, co):  # depthwise (har kanal alohida) + pointwise (1x1) konvolyutsiya
            return nn.Sequential(nn.Conv1d(ci, ci, 5, padding=2, groups=ci), nn.Conv1d(ci, co, 1), nn.BatchNorm1d(co), nn.ReLU(), nn.MaxPool1d(2))
        self.f = nn.Sequential(nn.Conv1d(3, w, 5, padding=2), nn.BatchNorm1d(w), nn.ReLU(), nn.MaxPool1d(2),
                               ds(w, 2 * w), ds(2 * w, 2 * w), nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.head = nn.Linear(2 * w, 6)
    def forward(self, x):
        return self.head(self.f(x))

def oqit(model, Xtr, ytr, epoxa=5, lr=3e-3):
    opt = torch.optim.Adam(model.parameters(), lr); lf = nn.CrossEntropyLoss()
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
    for _ in range(epoxa):
        model.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 64):
            b = perm[i:i + 64]; opt.zero_grad(); lf(model(Xt[b]), yt[b]).backward(); opt.step()
    return model.eval()

X, y = har_yarat(3000)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

# TODO 1: HARNet ni o'qiting (vaqtini perf_counter bilan o'lchang), test aniqligini va parametrlar sonini hisoblang
model, t_oq, acc, n_par = None, None, None, None
raise NotImplementedError("TODO 1: model o'qitish, aniqlik va parametrlar soni")
print("=== Etalon model (PyTorch FP32) ===")
print(f"HARNet: {n_par} parametr ({n_par * 4 / 1024:.1f} KB FP32), o'qitish {t_oq:.1f} s, test aniqligi {acc:.3f}")

print("\n=== O'lchov metodikasi ===")
x1 = torch.tensor(Xte[:1])

# TODO 2: chaqir(fn, n) — fn ni n marta chaqirib, har chaqiruv vaqtini ms da numpy massivda qaytaring
def chaqir(fn, n):
    raise NotImplementedError("TODO 2: chaqiruvlar vaqtini o'lchash")

# TODO 3: yangi (sovuq) modelda 50 chaqiruv bilan isinish effektini ko'rsating: 1-chaqiruv, 2–10 va 40–50 o'rtachasi

# TODO 4: toliq_olch(fn) — isinishdan keyin 400 chaqiruv, p50/p95/p99/p99.9 va o'rtachani qaytaring
def toliq_olch(fn, takror=400, isinish=50):
    raise NotImplementedError("TODO 4: foizlar bo'yicha o'lchov")

# TODO 5: eager va TorchScript (trace + freeze) rejimlarini 5 martadan o'lchab jadval tuzing,
#         har rejim uchun p50 o'rtacha ± std va CV ni chiqaring
