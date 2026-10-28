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
    m = new_model(); opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    t0 = time.time()
    for _ in range(steps):
        x = pgd(m.eval(), Xtr, ytr, EPS_TRAIN, steps=k) if adv else Xtr
        m.train(); opt.zero_grad(); F.cross_entropy(m(x), ytr).backward(); opt.step()
    return m.eval(), time.time() - t0

def acc(model, x, y):
    with torch.no_grad(): return (model(x).argmax(1) == y).float().mean().item()

clean, t_clean = train(adv=False)
robust, t_adv = train(adv=True)
print(f"O'qitish vaqti: toza={t_clean:.1f}s  adversarial(PGD-5)={t_adv:.1f}s  (~{t_adv / t_clean:.0f}x qimmat)")
EPS = [0.0, 0.05, 0.1, 0.15, 0.2]
print("=== Toza model vs adversarial o'qitilgan model ===")
print(f"{'eps':>5s} | {'toza: FGSM':>10s} {'toza: PGD-10':>12s} | {'adv: FGSM':>9s} {'adv: PGD-10':>11s}")
curves = {k: [] for k in ("toza/FGSM", "toza/PGD", "adv/FGSM", "adv/PGD")}
for e in EPS:
    vals = []
    for name, m in (("toza", clean), ("adv", robust)):
        torch.manual_seed(1)
        af = acc(m, fgsm(m, Xte, yte, e) if e else Xte, yte); ap = acc(m, pgd(m, Xte, yte, e) if e else Xte, yte)
        curves[f"{name}/FGSM"].append(af); curves[f"{name}/PGD"].append(ap); vals += [af, ap]
    print(f"{e:5.2f} | {vals[0]:10.3f} {vals[1]:12.3f} | {vals[2]:9.3f} {vals[3]:11.3f}")
print(f"\nMuroza: toza aniqlik {curves['toza/PGD'][0]:.3f} -> {curves['adv/PGD'][0]:.3f}; "
      f"PGD-10 (eps={EPS_TRAIN}) ostida {curves['toza/PGD'][2]:.3f} -> {curves['adv/PGD'][2]:.3f}")
print("Kichik va oson to'plamda (digits) adversarial o'qitish regulyarizator vazifasini bajarib, toza aniqlikni")
print("pasaytirmasligi ham mumkin; CIFAR-10/ResNet kabi real masalalarda toza aniqlik odatda 5-15 bandga tushadi.")
print("Eslatma: himoya faqat o'qitilgan tahdid modeli (L_inf, eps<=0.1) doirasida; eps=0.2 da yana pasayadi.")

plt.figure(figsize=(6.5, 4))
for (k, v), st in zip(curves.items(), ("o--", "o-", "s--", "s-")): plt.plot(EPS, v, st, label=k)
plt.xlabel("epsilon"); plt.ylabel("test aniqligi"); plt.ylim(0, 1.02); plt.grid(alpha=.3); plt.legend()
plt.title("Adversarial o'qitish ta'siri (digits MLP)"); plt.tight_layout()
