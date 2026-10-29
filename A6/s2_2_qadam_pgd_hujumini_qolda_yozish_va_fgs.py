"""A6 — 2-qadam. PGD hujumini qo'lda yozish va FGSM bilan taqqoslash

PGD: tasodifiy boshlang'ich nuqta, K qadam `alpha*sign(grad)`, har qadimdan keyin eps-sharga proyeksiya va [0,1] qirqish. FGSM, PGD-10 (tasodifiy boshlanish bilan/siz) va PGD-40 bir xil epsilon to'rida solishtiriladi, har hujum uchun L_inf chegarasi `assert` bilan tekshiriladi. Talaba alpha, qadamlar soni va boshlang'ich nuqtaning ta'sirini ko'radi.

Kutilgan natija: PGD har eps'da FGSM dan past yoki teng aniqlik beradi (eps=0.15: FGSM ~0.11, PGD ~0.05; eps=0.2: PGD ~0.00). Assert o'tadi. 4 egri chiziqli grafik.
"""
# 2-qadam. PGD hujumini qo'lda amalga oshirish va FGSM bilan taqqoslash
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
    x = x.clone().detach().requires_grad_(True)
    grad, = torch.autograd.grad(F.cross_entropy(model(x), y), x)
    return (x + eps * grad.sign()).clamp(0, 1).detach()

def pgd(model, x, y, eps, alpha=None, steps=10, random_start=True):
    """Iterativ FGSM + proyeksiya: har qadamdan keyin x atrofidagi eps-sharga va [0,1] ga qaytariladi."""
    alpha = alpha or 2.5 * eps / steps
    # TODO 1: boshlang'ich nuqta — random_start bo'lsa x + U(-eps, eps), aks holda x; [0,1] ga qirqing
    # TODO 2: steps marta: gradient, alpha * sign(grad) qadami
    # TODO 3: har qadamdan keyin [x-eps, x+eps] sharga proyeksiya va [0,1] qirqish
    raise NotImplementedError("TODO 1-3: PGD")

def acc(model, x, y):
    with torch.no_grad(): return (model(x).argmax(1) == y).float().mean().item()

model = train(Xtr, ytr)
EPS = [0.02, 0.05, 0.1, 0.15, 0.2]
attacks = {"FGSM": lambda e: fgsm(model, Xte, yte, e),
           "PGD-10 (tasodifiy boshlanmasdan)": lambda e: pgd(model, Xte, yte, e, steps=10, random_start=False),
           "PGD-10": lambda e: pgd(model, Xte, yte, e, steps=10),
           "PGD-40": lambda e: pgd(model, Xte, yte, e, steps=40)}
print("=== Aniqlik: FGSM vs PGD (epsilon bo'yicha) ===")
curves = {}
for name, fn in attacks.items():
    row = []
    for e in EPS:
        torch.manual_seed(1); xa = fn(e)
        # TODO 4: L_inf chegarasini assert bilan tekshiring va aniqlikni row ga qo'shing
    curves[name] = row
print("L_inf chegarasi barcha hujumlarda saqlangan (assert o'tdi).")
