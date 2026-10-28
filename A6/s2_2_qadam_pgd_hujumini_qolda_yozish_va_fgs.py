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
    x0 = x.clone().detach()
    xa = (x0 + torch.empty_like(x0).uniform_(-eps, eps)) if random_start else x0.clone()
    xa = xa.clamp(0, 1)
    for _ in range(steps):
        xa.requires_grad_(True)
        grad, = torch.autograd.grad(F.cross_entropy(model(xa), y), xa)
        xa = xa.detach() + alpha * grad.sign()
        xa = torch.min(torch.max(xa, x0 - eps), x0 + eps).clamp(0, 1)   # proyeksiya
    return xa.detach()

def acc(model, x, y):
    with torch.no_grad(): return (model(x).argmax(1) == y).float().mean().item()

model = train(Xtr, ytr)
EPS = [0.02, 0.05, 0.1, 0.15, 0.2]
attacks = {"FGSM": lambda e: fgsm(model, Xte, yte, e),
           "PGD-10 (tasodifiy boshlanmasdan)": lambda e: pgd(model, Xte, yte, e, steps=10, random_start=False),
           "PGD-10": lambda e: pgd(model, Xte, yte, e, steps=10),
           "PGD-40": lambda e: pgd(model, Xte, yte, e, steps=40)}
print(f"Toza test aniqligi: {acc(model, Xte, yte):.3f}")
print("=== Aniqlik: FGSM vs PGD (epsilon bo'yicha) ===")
print(f"{'hujum':34s} " + " ".join(f"eps={e:<5}" for e in EPS))
curves = {}
for name, fn in attacks.items():
    row = []
    for e in EPS:
        torch.manual_seed(1); xa = fn(e)
        assert (xa - Xte).abs().max() <= e + 1e-6, "L_inf chegarasi buzildi!"   # tahdid modeli tekshiruvi
        row.append(acc(model, xa, yte))
    curves[name] = row
    print(f"{name:34s} " + " ".join(f"{a:9.3f}" for a in row))
print("L_inf chegarasi barcha hujumlarda saqlangan (assert o'tdi).")
print("Xulosa: PGD har doim FGSM dan kuchli yoki teng; qadamlar soni oshsa hujum kuchayadi, lekin narx K baravar.")
print("Agar qo'lda yozilgan PGD FGSM dan kuchsiz chiqsa — xato bor (eval() yo'q, detach yo'q yoki proyeksiya noto'g'ri).")

plt.figure(figsize=(6.5, 4))
for name, row in curves.items(): plt.plot(EPS, row, marker="o", label=name)
plt.xlabel("epsilon"); plt.ylabel("test aniqligi"); plt.ylim(0, 1.02); plt.grid(alpha=.3); plt.legend(fontsize=8)
plt.title("FGSM vs PGD — toza o'qitilgan model"); plt.tight_layout()
