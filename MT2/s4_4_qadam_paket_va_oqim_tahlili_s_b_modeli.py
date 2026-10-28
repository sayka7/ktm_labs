"""MT2 — 4-qadam. Paket va oqim tahlili: S(b) modeli va dinamik paketlash simulyatsiyasi

Batch 1–128 uchun xizmat vaqti o'lchanadi va chiziqli model S(b) = c0 + c1·b moslanadi (c0 — har chaqiruv ustamasi, c1 — namuna narxi). So'ngra Puasson oqimi bilan navbat simulyatsiyasi: paketsiz, dinamik paketlash (b<=16, kutish 0,3 ms) va (b<=64, kutish 1 ms) siyosatlari turli yuklamada p50/p99 va throughput bo'yicha solishtiriladi (Triton dynamic_batching, OpenVINO avtomatik paketlash g'oyasi). Xuddi shu simulyatsiya GPU uslubidagi S(b) (katta c0, kichik c1 — faraziy) uchun takrorlanadi.

Kutilgan natija: Bu CPU'da c0 kichik (~0,02 ms), throughput batch bilan ~15k → ~18,5k inf/s; GPU uslubida dinamik paketlash SLO ichidagi yuklamani ~1000 dan ~60 000 so'rov/s gacha oshiradi (taxminiy).
"""
# 4-qadam. Paket (batch) va oqim tahlili: xizmat vaqti modeli S(b), dinamik paketlash simulyatsiyasi, SLO bo'yicha maksimal yuklama
import warnings; warnings.filterwarnings("ignore")
import time, numpy as np, pandas as pd, torch, torch.nn as nn, onnxruntime as ort
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ort.set_default_logger_severity(3)
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(2); torch.backends.nnpack.set_flags(False)

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

model = HARNet().eval()
torch.onnx.export(model, torch.randn(1, 3, 128), "har.onnx", input_names=["x"], output_names=["logits"],
                  dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
so = ort.SessionOptions(); so.intra_op_num_threads = 1
s = ort.InferenceSession("har.onnx", so, providers=["CPUExecutionProvider"])

# --- 1) Xizmat vaqti S(b): paket o'lchami bo'yicha o'lchov
rows = []
for b in (1, 2, 4, 8, 16, 32, 64, 128):
    xb = np.random.randn(b, 3, 128).astype(np.float32)
    for _ in range(10): s.run(None, {"x": xb})
    t = []
    for _ in range(max(30, 600 // b)):
        a = time.perf_counter(); s.run(None, {"x": xb}); t.append(time.perf_counter() - a)
    t = np.array(t) * 1e3
    rows.append((b, np.median(t), np.percentile(t, 99), b / np.median(t) * 1e3))
S = pd.DataFrame(rows, columns=["batch", "S_p50, ms", "S_p99, ms", "inf/s"])
c1, c0 = np.polyfit(S.batch, S["S_p50, ms"], 1)                     # S(b) ~ c0 + c1*b
print("=== Paket o'lchami: xizmat vaqti va throughput ===")
print(S.round(4).to_string(index=False))
print(f"Chiziqli model: S(b) = {c0:.4f} + {c1:.4f}*b ms  (c0 — har chaqiruvning qat'iy ustamasi, c1 — namuna narxi)")
print(f"Nazariy throughput chegarasi: 1/c1 = {1 / c1 * 1e3:.0f} inf/s; batch 1 da: {1 / (c0 + c1) * 1e3:.0f} inf/s")

# --- 2) Dinamik paketlash simulyatsiyasi (Triton/TensorRT-LLM/OpenVINO «auto-batching» g'oyasi)
def simul(lam, max_b, max_wait_ms, c0, c1, n=6000, seed=0):
    """Puasson oqimi (lam so'rov/s), bitta ishchi. Navbatdan max_b gacha olinadi; birinchi so'rov max_wait kutsa — jo'natiladi."""
    r = np.random.default_rng(seed)
    kel = np.cumsum(r.exponential(1000.0 / lam, n))                   # kelish vaqtlari, ms
    bosh, i, kech, bs = 0.0, 0, np.empty(n), []
    while i < n:
        start = max(bosh, kel[i])                                     # ishchi bo'shagan va so'rov kelgan payt
        t_lim = max(start, kel[i] + max_wait_ms)                      # paketni to'ldirish uchun kutish chegarasi
        j = min(i + max_b, int(np.searchsorted(kel, t_lim, side="right")))
        launch = max(start, kel[j - 1]) if j - i == max_b else t_lim  # paket to'lsa — darhol, aks holda muddat tugaganda
        tugash = launch + c0 + c1 * (j - i)
        kech[i:j] = tugash - kel[i:j]; bs.append(j - i); bosh = tugash; i = j
    return np.percentile(kech, 50), np.percentile(kech, 99), np.mean(bs), n / (bosh - kel[0]) * 1e3

SLO = 2.0                                                             # p99 <= 2 ms (taxminiy talab)
SIYOSAT = [("paketsiz", 1, 0.0), ("dinamik b<=16, kutish 0,3 ms", 16, 0.3), ("dinamik b<=64, kutish 1 ms", 64, 1.0)]
def jadval(c0_, c1_, yuklamalar):
    rows = []
    for lam in yuklamalar:
        for nom, mb, mw in SIYOSAT:
            p50, p99, ob, X = simul(lam, mb, mw, c0_, c1_)
            rows.append((lam, nom, p50, p99, ob, X, "ha" if p99 <= SLO else "yo'q"))
    D = pd.DataFrame(rows, columns=["yuklama", "siyosat", "p50, ms", "p99, ms", "o'rt. paket", "throughput", "SLO"])
    print(D.round(3).to_string(index=False))
    for nom, _, _ in SIYOSAT:
        ok = D[(D.siyosat == nom) & (D.SLO == "ha")]
        print(f"  {nom:30s}: SLO ni saqlagan maksimal yuklama ~ {ok.yuklama.max() if len(ok) else 0} so'rov/s")
    return D
print(f"\n=== Dinamik paketlash simulyatsiyasi: bu CPU (o'lchangan S(b)), SLO p99 <= {SLO} ms ===")
D = jadval(c0, c1, (2000, 6000, 10000, 14000, 20000))
# GPU'da qat'iy ustama (kernel ishga tushirish, PCIe nusxalash) katta, namuna narxi kichik — paketlash foydasi ancha katta
g0, g1 = 0.40, 0.004
print(f"\n=== Xuddi shu simulyatsiya: GPU uslubidagi S(b) = {g0} + {g1}*b ms (TAXMINIY, illyustrativ) ===")
G = jadval(g0, g1, (1000, 2000, 5000, 20000, 60000))
print("Xulosa: S(b) dagi c0/c1 nisbati katta bo'lsa (GPU, TensorRT) dinamik paketlash throughput'ni bir necha barobar oshiradi;")
print("bu CPU'da c0 kichik — paketlash faqat navbat to'lib qolganda yordam beradi. max_wait — kechikish va throughput o'rtasidagi tutqich.")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
ax[0].plot(S.batch, S["inf/s"], marker="o"); ax[0].set_xscale("log", base=2); ax[0].set_xlabel("batch"); ax[0].set_ylabel("inf/s"); ax[0].set_title("Throughput vs paket")
for nom in G.siyosat.unique():
    d = G[G.siyosat == nom]; ax[1].plot(d["yuklama"], d["p99, ms"].clip(upper=50), marker="o", label=nom)
ax[1].axhline(SLO, ls="--", c="r"); ax[1].set_yscale("log"); ax[1].set_xlabel("yuklama, so'rov/s"); ax[1].set_ylabel("p99, ms (GPU uslubi)"); ax[1].set_xscale("log"); ax[1].legend(fontsize=7); plt.tight_layout()
