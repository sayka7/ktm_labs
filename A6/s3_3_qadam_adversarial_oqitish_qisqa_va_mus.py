"""A6 — 3-qadam. Adversarial o'qitish (qisqa) va mustahkamlik murozasi

Madry uslubi: har qadamda avval PGD-5 (eps=0.1) bilan adversarial o'quv paketi yasaladi, model aynan unda o'qitiladi. Toza va adversarial modellar FGSM/PGD-10 ostida 5 ta eps uchun taqqoslanadi (4 egri chiziq), o'qitish vaqti nisbati chiqadi. Talaba EPS_TRAIN, k va qadamlar sonini o'zgartirib, murozani o'rganadi.

Kutilgan natija: Adversarial o'qitish ~6x qimmat; PGD-10 eps=0.1 ostida aniqlik ~0.27 dan ~0.83 ga ko'tariladi; eps=0.2 da himoya zaiflashadi (~0.2).
"""
# 3-qadam. Adversarial o'qitish (Madry uslubi, qisqa) va mustahkamlikni taqqoslash
import time, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
torch.manual_seed(0); torch.set_num_threads(2)

d = load_digits()
Xtr, Xte, ytr, yte = train_test_split(d.data / 16.0, d.target, test_size=0.3, random_state=0, stratify=d.target)
Xtr, Xte = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Xte, dtype=torch.float32)
ytr, yte = torch.tensor(ytr), torch.tensor(yte)
EPS_TRAIN = 0.1                      # o'qitishdagi tahdid modeli

def new_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))

def pgd(model, x, y, eps, steps=10, random_start=True):
    alpha = 2.5 * eps / steps
    x0 = x.clone().detach()
    xa = (x0 + torch.empty_like(x0).uniform_(-eps, eps)).clamp(0, 1) if random_start else x0.clone()
    for _ in range(steps):
        xa.requires_grad_(True)
        grad, = torch.autograd.grad(F.cross_entropy(model(xa), y), xa)
        xa = torch.min(torch.max(xa.detach() + alpha * grad.sign(), x0 - eps), x0 + eps).clamp(0, 1)
    return xa.detach()

def fgsm(model, x, y, eps):
    x = x.clone().detach().requires_grad_(True)
    grad, = torch.autograd.grad(F.cross_entropy(model(x), y), x)
    return (x + eps * grad.sign()).clamp(0, 1).detach()

def train(adv=False, steps=300, k=5):
    """adv=True: har qadamda avval PGD-k bilan adversarial paket yasaladi, model aynan unda o'qitiladi (min-max)."""
    # TODO 1: model va Adam yarating, vaqtni o'lchang
    # TODO 2: har qadamda adv bo'lsa PGD-k (EPS_TRAIN) bilan o'quv paketini yasang (model eval holatida), aks holda Xtr
    # TODO 3: model.train() holatida shu paketda bir optimizatsiya qadamini bajaring; (model.eval(), vaqt) qaytaring
    raise NotImplementedError("TODO 1-3: adversarial o'qitish")

def acc(model, x, y):
    with torch.no_grad(): return (model(x).argmax(1) == y).float().mean().item()

clean, t_clean = train(adv=False)
robust, t_adv = train(adv=True)
EPS = [0.0, 0.05, 0.1, 0.15, 0.2]
print("=== Toza model vs adversarial o'qitilgan model ===")
curves = {k: [] for k in ("toza/FGSM", "toza/PGD", "adv/FGSM", "adv/PGD")}
# TODO 4: har eps va har model uchun FGSM va PGD-10 ostidagi aniqlikni curves ga yozing va jadvalni chiqaring
# TODO 5: "Muroza:" qatori — toza aniqlik va PGD-10 (eps=EPS_TRAIN) ostidagi aniqlik qanday o'zgardi
