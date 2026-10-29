"""A6 — 1-qadam. Bazaviy model va FGSM: aniqlikning epsilon'ga bog'liqligi

Digits ma'lumoti [0,1] piksel fazosiga keltiriladi (1 kulrang daraja = 1/16), MLP 300 qadam Adam bilan o'qitiladi (test ~97%). FGSM qo'lda yoziladi: `x + eps*sign(grad)`, [0,1] ga qirqish. 7 ta epsilon uchun aniqlik jadvali va mustahkamlik egri chizig'i chiziladi; shu eps'dagi tasodifiy ±eps shovqin bilan taqqoslanadi. Talaba model chuqurligini, epsilon to'rini va o'qitish qadamlarini o'zgartirib ko'radi.

Kutilgan natija: eps=0.05 da aniqlik ~0.77, eps=0.1 da ~0.3, eps=0.15 da ~0.1; tasodifiy shovqin eps=0.15 da ~0.93 qoldiradi. Grafik: keskin tushuvchi egri chiziq.
"""
# 1-qadam. Bazaviy model (digits 8x8, MLP) va FGSM hujumi: aniqlikning epsilon ga bog'liqligi
# Piksellar [0,1] oraliqqa keltiriladi (data/16) — 1 kulrang daraja = 1/16 = 0.0625.
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
torch.manual_seed(0); torch.set_num_threads(2)

d = load_digits()
Xtr, Xte, ytr, yte = train_test_split(d.data / 16.0, d.target, test_size=0.3, random_state=0, stratify=d.target)
Xtr, Xte = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Xte, dtype=torch.float32)
ytr, yte = torch.tensor(ytr), torch.tensor(yte)

def train(X, y, steps=300):
    m = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(steps):
        opt.zero_grad(); F.cross_entropy(m(X), y).backward(); opt.step()
    return m.eval()

def fgsm(model, x, y, eps):
    """x_adv = ? — bitta gradient qadami, [0,1] ga qirqiladi."""
    # TODO 1: x bo'yicha yo'qotish gradientini oling va eps * sign(grad) qadamini qo'yib, [0,1] ga qirqing
    raise NotImplementedError("TODO 1: FGSM")

def acc(model, x, y):
    with torch.no_grad(): return (model(x).argmax(1) == y).float().mean().item()

model = train(Xtr, ytr)
print(f"Toza aniqlik: o'quv={acc(model, Xtr, ytr):.3f}  test={acc(model, Xte, yte):.3f}")
EPS = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
print("=== FGSM: aniqlik vs epsilon ===")
res = []
# TODO 2: har eps uchun hujumli test to'plamida aniqlik va o'rtacha L_inf masofani chiqaring (res ga yig'ing)
for e in EPS:
    xa = fgsm(model, Xte, yte, e) if e > 0 else Xte
# TODO 3: eps=0.15 da tasodifiy +-eps shovqin bilan taqqoslang ("Taqqoslash eps=..." qatori) va egri chiziqni chizing
