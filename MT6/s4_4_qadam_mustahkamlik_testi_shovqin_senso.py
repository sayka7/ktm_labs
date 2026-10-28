"""MT6 — 4-qadam. Mustahkamlik testi: shovqin, sensor uzilishi, kalibrlash drifti va adversarial hujum

Zavod uskunasi holatini 6 sensor bo'yicha tasniflovchi uch model (LogReg, RandomForest, PyTorch MLP) ISO/IEC TR 24029-1 ruhidagi empirik testlardan o'tadi: Gauss shovqini, har bir sensorning uzilishi (o'rtacha qiymat bilan to'ldirish), harorat sensorining +2/5/10 C drifti va FGSM (MLP va LR uchun oq quti — LR gradienti analitik, RF uchun MLP'dan ko'chirilgan hujum). Natijalar oldindan belgilangan qabul mezonlari bilan OK/XATO jadvaliga keltiriladi.

Kutilgan natija: Toza aniqlik ~0,95–0,96; harorat sensori uzilishi ~11–13 p.p. pasayish beradi (mezon 10 — XATO); +10 C drift RF ni ~0,65 ga tushiradi; FGSM eps=0,2 da MLP mezondan o'tmaydi, ko'chirilgan hujum RF ga kuchsizroq.
"""
# 4-qadam. Mustahkamlik testi (ISO/IEC TR 24029-1 ruhida empirik baholash): shovqin, sensor uzilishi, drift, adversarial
# Ssenariy: zavod uskunasi holatini 6 ta sensor bo'yicha tasniflash (normal / podshipnik nuqsoni / qizib ketish). Ma'lumot sintetik.
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
torch.backends.nnpack.set_flags(False); torch.manual_seed(0); torch.set_num_threads(2)
rng = np.random.default_rng(0); n = 4000
SENS = ["harorat", "tebranish_rms", "tok", "bosim", "aylanish", "akustik"]
y = rng.choice(3, n, p=[0.6, 0.25, 0.15])
mu = np.array([[60, 2.0, 12, 5.0, 1450, 70], [63, 3.2, 13, 5.0, 1440, 76], [74, 2.3, 14.5, 4.6, 1450, 72]])
sd = np.array([4, 0.5, 1.0, 0.4, 15, 3])
Xraw = mu[y] + rng.normal(0, 1, (n, 6)) * sd
Xtr, Xte, ytr, yte = train_test_split(Xraw, y, test_size=0.4, random_state=0, stratify=y)
sc = StandardScaler().fit(Xtr); Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)

lr = LogisticRegression(max_iter=2000).fit(Ztr, ytr)
rf = RandomForestClassifier(200, max_depth=10, random_state=0, n_jobs=2).fit(Ztr, ytr)
mlp = nn.Sequential(nn.Linear(6, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3))
opt = torch.optim.Adam(mlp.parameters(), 3e-3); Xt, yt = torch.tensor(Ztr, dtype=torch.float32), torch.tensor(ytr)
for _ in range(400): opt.zero_grad(); F.cross_entropy(mlp(Xt), yt).backward(); opt.step()
mlp.eval()
def mlp_pred(Z):
    with torch.no_grad(): return mlp(torch.tensor(Z, dtype=torch.float32)).argmax(1).numpy()
MODELS = {"LogReg": lr.predict, "RandomForest": rf.predict, "MLP": mlp_pred}
acc = lambda f, Z: (f(Z) == yte).mean()

def fgsm_mlp(Z, eps):
    x = torch.tensor(Z, dtype=torch.float32, requires_grad=True)
    g, = torch.autograd.grad(F.cross_entropy(mlp(x), torch.tensor(yte)), x)
    return (x + eps * g.sign()).detach().numpy()
def fgsm_lr(Z, eps):
    """Multinomial LR uchun analitik gradient: dL/dx = W^T (p - onehot)."""
    P = lr.predict_proba(Z); P[np.arange(len(yte)), yte] -= 1
    return Z + eps * np.sign(P @ lr.coef_)

print("=== Toza aniqlik ===")
clean = {m: acc(f, Zte) for m, f in MODELS.items()}
print("  ".join(f"{m}={a:.3f}" for m, a in clean.items()))

print("\n=== 1) Gauss shovqini (sigma — standartlashgan birlikda) ===")
noise = {}
for s in (0.1, 0.3, 0.5, 1.0):
    Zn = Zte + rng.normal(0, s, Zte.shape); noise[s] = {m: acc(f, Zn) for m, f in MODELS.items()}
    print(f"sigma={s:<4} " + "  ".join(f"{m}={a:.3f}" for m, a in noise[s].items()))

print("\n=== 2) Bitta sensor uzilishi (qiymat o'rniga o'rtacha = 0 qo'yiladi) ===")
worst = {}
for j, nm in enumerate(SENS):
    Zm = Zte.copy(); Zm[:, j] = 0.0
    r = {m: acc(f, Zm) for m, f in MODELS.items()}
    for m in r: worst[m] = min(worst.get(m, 1), r[m])
    print(f"{nm:<14} " + "  ".join(f"{m}={a:.3f}" for m, a in r.items()))

print("\n=== 3) Kalibrlash drifti: harorat sensori +delta C ===")
drift = {}
for dC in (2, 5, 10):
    Xd = Xte.copy(); Xd[:, 0] += dC; Zd = sc.transform(Xd); drift[dC] = {m: acc(f, Zd) for m, f in MODELS.items()}
    print(f"+{dC:>2} C  " + "  ".join(f"{m}={a:.3f}" for m, a in drift[dC].items()))

print("\n=== 4) Adversarial (FGSM, L_inf, standartlashgan birlikda) ===")
EPS = [0.05, 0.1, 0.2, 0.3]; adv = {m: [] for m in MODELS}
for e in EPS:
    Za_mlp, Za_lr = fgsm_mlp(Zte, e), fgsm_lr(Zte, e)
    adv["MLP"].append(acc(mlp_pred, Za_mlp)); adv["LogReg"].append(acc(lr.predict, Za_lr))
    adv["RandomForest"].append(acc(rf.predict, Za_mlp))               # RF: gradient yo'q -> MLP dan ko'chirilgan hujum
    print(f"eps={e:<5} MLP(oq quti)={adv['MLP'][-1]:.3f}  LogReg(oq quti)={adv['LogReg'][-1]:.3f}  RF(ko'chirilgan)={adv['RandomForest'][-1]:.3f}")

print("\n=== Qabul mezonlari bo'yicha xulosa (pasayish, foiz punktlarda) ===")
CRIT = [("shovqin sigma=0.3", lambda m: clean[m] - noise[0.3][m], 5), ("eng yomon sensor uzilishi", lambda m: clean[m] - worst[m], 10),
        ("harorat drifti +5 C", lambda m: clean[m] - drift[5][m], 5), ("FGSM eps=0.2", lambda m: clean[m] - adv[m][2], 10)]
rows = []
for m in MODELS:
    row = {"model": m}
    for name, fn, lim in CRIT:
        d = 100 * fn(m); row[name] = f"{d:5.1f} {'OK' if d <= lim else 'XATO'}"
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))
print("Chegaralar: shovqin <= 5, sensor <= 10, drift <= 5, FGSM(eps=0.2) <= 10 p.p. (o'quv misoli uchun tanlangan)")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
for m in MODELS:
    ax[0].plot([0, .1, .3, .5, 1.0], [clean[m]] + [noise[s][m] for s in (0.1, 0.3, 0.5, 1.0)], "-o", label=m)
    ax[1].plot([0] + EPS, [clean[m]] + adv[m], "-o", label=m)
ax[0].set_xlabel("shovqin sigma"); ax[1].set_xlabel("FGSM eps"); ax[0].set_ylabel("aniqlik")
for a in ax: a.grid(alpha=.3); a.legend(fontsize=8)
plt.tight_layout()
print("\nIzoh: real tizimda testlar haqiqiy sensor yozuvlari, uskunaning ekspluatatsiya rejimlari va formal")
print("verifikatsiya (masalan, intervallar bilan chegaralash) bilan to'ldiriladi; natija texnik hujjatga kiritiladi.")
