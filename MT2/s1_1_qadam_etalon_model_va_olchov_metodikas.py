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
t0 = time.perf_counter(); model = oqit(HARNet(), Xtr, ytr); t_oq = time.perf_counter() - t0
with torch.no_grad():
    acc = (model(torch.tensor(Xte)).argmax(1).numpy() == yte).mean()
n_par = sum(p.numel() for p in model.parameters())
print("=== Etalon model (PyTorch FP32) ===")
print(f"HARNet: {n_par} parametr ({n_par * 4 / 1024:.1f} KB FP32), o'qitish {t_oq:.1f} s, test aniqligi {acc:.3f}")
print("Sinflar:", ", ".join(SINFLAR))

# --- O'lchov metodikasi
print("\n=== O'lchov metodikasi ===")
print(f"Taymer aniqligi (perf_counter): {time.get_clock_info('perf_counter').resolution * 1e9:.0f} ns")
x1 = torch.tensor(Xte[:1])
def chaqir(fn, n):
    t = np.empty(n)
    for i in range(n):
        a = time.perf_counter(); fn(); t[i] = time.perf_counter() - a
    return t * 1e3
m_fresh = HARNet().eval()
with torch.inference_mode():
    sovuq = chaqir(lambda: m_fresh(x1), 50)
print(f"Isinish effekti: 1-chaqiruv {sovuq[0]:.2f} ms, 2–10 o'rtacha {sovuq[1:10].mean():.3f} ms, 40–50 o'rtacha {sovuq[40:].mean():.3f} ms")

# Bir xil konfiguratsiyani 5 marta o'lchash — natijalar tarqalishi (o'rtacha emas, foizlar va CV)
def toliq_olch(fn, takror=400, isinish=50):
    chaqir(fn, isinish); t = chaqir(fn, takror)
    return [np.percentile(t, q) for q in (50, 95, 99, 99.9)] + [t.mean()]
rows = []
with torch.inference_mode():
    jit = torch.jit.trace(model, x1); jit = torch.jit.freeze(jit)
    for nom, fn in [("eager + no_grad", lambda: model(x1)), ("TorchScript freeze", lambda: jit(x1))]:
        for k in range(5):
            rows.append([nom, k + 1] + toliq_olch(fn))
df = pd.DataFrame(rows, columns=["Rejim", "Takror", "p50", "p95", "p99", "p99.9", "o'rtacha"])
print(df.round(3).to_string(index=False))
agg = df.groupby("Rejim")[["p50", "p99"]].agg(["mean", "std"])
for rj in agg.index:
    m50, s50 = agg.loc[rj, ("p50", "mean")], agg.loc[rj, ("p50", "std")]
    print(f"{rj}: p50 = {m50:.3f} ± {s50:.3f} ms (CV {s50 / m50:.1%}), p99 = {agg.loc[rj, ('p99', 'mean')]:.3f} ms")
print("Qoida: isinishsiz o'lchamang, kamida 3–5 marta takrorlang, o'rtacha emas p50/p99 ni va tarqalishni (CV) bering.")

with torch.inference_mode():
    t = chaqir(lambda: model(x1), 2000)
plt.figure(figsize=(6, 3.2)); plt.hist(t, bins=80, color="tab:blue"); plt.yscale("log")
for q, c in [(50, "g"), (99, "r")]: plt.axvline(np.percentile(t, q), c=c, ls="--", label=f"p{q}")
plt.xlabel("Kechikish, ms (batch 1)"); plt.ylabel("soni"); plt.title("Kechikish taqsimoti: uzun dum"); plt.legend(); plt.tight_layout()
