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
    x = x.clone().detach().requires_grad_(True)
    logit = model(x)[torch.arange(len(x)), cls].sum()
    grad, = torch.autograd.grad(logit, x)
    return grad.abs()

def entropy(s):
    p = s / s.sum(1, keepdim=True); return -(p * torch.log(p + 1e-12)).sum(1)

N = 8; x, y = Xte[:N], yte[:N]
xa = pgd(m, x, y, eps=0.15)
with torch.no_grad(): p_clean, p_adv = m(x).argmax(1), m(xa).argmax(1)
s_clean, s_adv = saliency(m, x, p_clean), saliency(m, xa, p_adv)
corr = torch.tensor([np.corrcoef(s_clean[i].numpy(), s_adv[i].numpy())[0, 1] for i in range(N)])
print("=== Saliency xaritasi: toza vs adversarial (eps=0.15) ===")
print(f"{'#':>2s} {'haqiqiy':>7s} {'toza':>5s} {'adv':>4s} {'korr(toza,adv)':>15s} {'entropiya toza':>15s} {'entropiya adv':>14s}")
for i in range(N):
    print(f"{i:2d} {y[i].item():7d} {p_clean[i].item():5d} {p_adv[i].item():4d} {corr[i]:15.3f} {entropy(s_clean)[i]:15.3f} {entropy(s_adv)[i]:14.3f}")
print(f"O'rtacha korrelyatsiya={corr.mean():.3f}; o'rtacha entropiya: toza={entropy(s_clean).mean():.3f}, adv={entropy(s_adv).mean():.3f}")
print("Kuzatuv: hujumdan keyin xarita 'ko'chadi' (korrelyatsiya past) — tushuntirishning o'zi ham hujumga zaif;")
print("xarita tarqoqligi (entropiya) adversarial namunani aniqlash belgisi sifatida ishlatilishi mumkin.")

fig, ax = plt.subplots(4, N, figsize=(1.6 * N, 6.8))
for i in range(N):
    ax[0, i].imshow(x[i].view(8, 8), cmap="gray"); ax[0, i].set_title(f"toza: {p_clean[i].item()}", fontsize=9)
    ax[1, i].imshow(s_clean[i].view(8, 8), cmap="hot")
    ax[2, i].imshow(xa[i].view(8, 8), cmap="gray"); ax[2, i].set_title(f"adv: {p_adv[i].item()}", fontsize=9)
    ax[3, i].imshow(s_adv[i].view(8, 8), cmap="hot")
for a in ax.flat: a.axis("off")
for r, t in enumerate(("toza rasm", "saliency (toza)", "adversarial rasm", "saliency (adv)")): ax[r, 0].set_ylabel(t)
plt.suptitle("Gradient-saliency: toza va PGD (eps=0.15) namunalar"); plt.tight_layout()
