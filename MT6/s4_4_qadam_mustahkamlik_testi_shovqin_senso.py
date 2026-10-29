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

# TODO 1: FGSM (MLP, oq quti): yo'qotish gradientini kirish bo'yicha olib, x + eps * sign(grad)
def fgsm_mlp(Z, eps):
    raise NotImplementedError("TODO 1: FGSM MLP uchun")

# TODO 2: multinomial LR uchun analitik gradient dL/dx = (p - onehot) W, so'ng x + eps * sign(...)
def fgsm_lr(Z, eps):
    raise NotImplementedError("TODO 2: FGSM LR uchun")

print("=== Toza aniqlik ===")
clean = {m: acc(f, Zte) for m, f in MODELS.items()}
print("  ".join(f"{m}={a:.3f}" for m, a in clean.items()))

# TODO 3: Gauss shovqini sigma = 0.1, 0.3, 0.5, 1.0 (standartlashgan birlikda) — har model aniqligi
# TODO 4: har bir sensor uzilishi: ustunni o'rtacha (=0) bilan almashtirib, eng yomon natijani saqlang
# TODO 5: harorat sensoriga +2/+5/+10 C drift (XOM qiymatga qo'shib, keyin standartlang)
# TODO 6: FGSM eps = 0.05..0.3: MLP va LR oq quti, RF — MLP dan ko'chirilgan hujum
Za = fgsm_mlp(Zte, 0.2)      # TODO 1 tayyor bo'lsa, shu yerdan hujumlarni boshlang
print("\n=== Qabul mezonlari bo'yicha xulosa (pasayish, foiz punktlarda) ===")
# TODO 7: pasayishni (p.p.) mezonlar bilan solishtirib OK/XATO jadvali: shovqin<=5, sensor<=10, drift<=5, FGSM(0.2)<=10
