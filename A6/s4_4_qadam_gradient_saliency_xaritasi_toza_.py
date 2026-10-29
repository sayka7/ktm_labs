"""A6 — 4-qadam. Gradient-saliency xaritasi: toza va hujum ostidagi raqamlar

MLP uchun Grad-CAM o'rniga gradient-saliency (|d logit/d x|, 8x8) hisoblanadi. 8 ta test raqami uchun toza rasm, uning saliency xaritasi, PGD (eps=0.15) bilan buzilgan rasm va uning xaritasi 4 qatorli panelda chiziladi; toza/adv xaritalar korrelyatsiyasi va entropiyasi jadvalda. Talaba tushuntirishning hujum ostida «ko'chishini» ko'radi.

Kutilgan natija: Adversarial rasmlar ko'zga toza bilan deyarli bir xil, lekin bashorat noto'g'ri; saliency korrelyatsiyasi ~0 atrofida (xarita ko'chgan). Panel rasmi.
"""
# 4-qadam. Gradient-saliency xaritasi (Grad-CAM ning MLP uchun muqobili): toza va hujum ostidagi raqamlar
# Saliency = |d logit_y / d x| — har piksel bashoratga qanchalik ta'sir qilishini ko'rsatadi.
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
torch.manual_seed(0); torch.set_num_threads(2)

d = load_digits()
Xtr, Xte, ytr, yte = train_test_split(d.data / 16.0, d.target, test_size=0.3, random_state=0, stratify=d.target)
Xtr, Xte = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Xte, dtype=torch.float32)
ytr, yte = torch.tensor(ytr), torch.tensor(yte)

m = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
opt = torch.optim.Adam(m.parameters(), lr=3e-3)
for _ in range(300):
    opt.zero_grad(); F.cross_entropy(m(Xtr), ytr).backward(); opt.step()
m.eval()

def pgd(model, x, y, eps, steps=10):
    alpha = 2.5 * eps / steps; x0 = x.clone().detach()
    xa = (x0 + torch.empty_like(x0).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        xa.requires_grad_(True)
        grad, = torch.autograd.grad(F.cross_entropy(model(xa), y), xa)
        xa = torch.min(torch.max(xa.detach() + alpha * grad.sign(), x0 - eps), x0 + eps).clamp(0, 1)
    return xa.detach()

def saliency(model, x, cls):
    """Bashorat qilingan sinf logitining kirish bo'yicha gradienti moduli (8x8)."""
    # TODO 1: har namuna uchun cls sinf logitini tanlab, kirish bo'yicha gradientning modulini qaytaring
    raise NotImplementedError("TODO 1: saliency")

def entropy(s):
    # TODO 2: xaritani ehtimollik taqsimotiga normallab, Shannon entropiyasini hisoblang (har namuna uchun)
    raise NotImplementedError("TODO 2: entropy")

N = 8; x, y = Xte[:N], yte[:N]
xa = pgd(m, x, y, eps=0.15)
with torch.no_grad(): p_clean, p_adv = m(x).argmax(1), m(xa).argmax(1)
s_clean, s_adv = saliency(m, x, p_clean), saliency(m, xa, p_adv)
# TODO 3: har namuna uchun toza va adv xaritalar Pearson korrelyatsiyasini hisoblang
print("=== Saliency xaritasi: toza vs adversarial (eps=0.15) ===")
# TODO 4: jadval (haqiqiy, toza, adv bashorat, korr, entropiyalar) va "O'rtacha korrelyatsiya=..." qatori; 4 qatorli panel rasmi
