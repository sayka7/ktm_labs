"""MT2 — 5-qadam. Parallellik: intra-op oqimlar, sessiyalar va LATENCY/THROUGHPUT rejimlari

Kengroq model (w=64) uchun 5 konfiguratsiya batch 1 va 32 da o'lchanadi: 1 sessiya/1 oqim, intra=2, umumiy sessiyaga 2 Python oqimi, 2 sessiya x 1 oqim va ortiqcha obuna (intra=2 x 2 oqim). Natija OpenVINO'ning NUM_STREAMS va PERFORMANCE_HINT (LATENCY/THROUGHPUT), TensorRT'ning bir nechta execution context + CUDA stream tushunchalari bilan jadvalda bog'lanadi.

Kutilgan natija: Eng past kechikish — intra=2 (1 oqim); eng yuqori throughput — 2 sessiya x 1 oqim (~2x); ortiqcha obunada p99 keskin oshadi (taxminiy).
"""
# 5-qadam. Parallellik: intra-op oqimlar, bir nechta so'rov oqimi (streams) va «latency» / «throughput» rejimlari
import warnings; warnings.filterwarnings("ignore")
import time, threading, numpy as np, pandas as pd, torch, torch.nn as nn, onnxruntime as ort
from concurrent.futures import ThreadPoolExecutor
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

# Kengroq model (w=64): oqimlararo parallellik sezilishi uchun hisob ko'proq bo'lsin
torch.onnx.export(HARNet(w=64).eval(), torch.randn(1, 3, 128), "har64.onnx", input_names=["x"], output_names=["logits"],
                  dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
def sessiya(intra, inter=1, parallel=False):
    so = ort.SessionOptions(); so.intra_op_num_threads = intra; so.inter_op_num_threads = inter
    if parallel: so.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    return ort.InferenceSession("har64.onnx", so, providers=["CPUExecutionProvider"])

def yuklama(sessiyalar, ishchi, batch, jami=1200):
    """ishchi ta Python oqimi navbatdagi so'rovlarni bajaradi; har ishchi o'z sessiyasini (yoki umumiy sessiyani) ishlatadi."""
    xb = np.random.randn(batch, 3, 128).astype(np.float32)
    for s in sessiyalar:
        for _ in range(20): s.run(None, {"x": xb})
    lat, lock, hisob = [], threading.Lock(), [0]
    def ish(k):
        s = sessiyalar[k % len(sessiyalar)]; mahalliy = []
        while True:
            with lock:
                if hisob[0] >= jami // batch: break
                hisob[0] += 1
            a = time.perf_counter(); s.run(None, {"x": xb}); mahalliy.append(time.perf_counter() - a)
        with lock: lat.extend(mahalliy)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(ishchi) as ex: list(ex.map(ish, range(ishchi)))
    dt = time.perf_counter() - t0; lat = np.array(lat) * 1e3
    return np.percentile(lat, 50), np.percentile(lat, 99), len(lat) * batch / dt

CFG = [("A: 1 sessiya, intra=1, 1 oqim (LATENCY)", lambda: [sessiya(1)], 1),
       ("B: 1 sessiya, intra=2, 1 oqim", lambda: [sessiya(2)], 1),
       ("C: 1 umumiy sessiya, intra=1, 2 oqim", lambda: [sessiya(1)], 2),
       ("D: 2 sessiya x intra=1, 2 oqim (THROUGHPUT)", lambda: [sessiya(1), sessiya(1)], 2),
       ("E: 1 sessiya, intra=2, 2 oqim (ortiqcha obuna)", lambda: [sessiya(2)], 2)]
rows = []
for batch in (1, 32):
    for nom, f, ish in CFG:
        p50, p99, X = yuklama(f(), ish, batch, jami=1200 if batch == 1 else 6400)
        rows.append((batch, nom, p50, p99, X))
df = pd.DataFrame(rows, columns=["batch", "Konfiguratsiya", "p50, ms", "p99, ms", "inf/s"])
df["nisbiy_X"] = df["inf/s"] / df.groupby("batch")["inf/s"].transform("first")
print("=== Parallellik konfiguratsiyalari (2 yadroli CPU) ===")
print(df.round(3).to_string(index=False))
for b in (1, 32):
    d = df[df.batch == b]; eng = d.loc[d["inf/s"].idxmax()]; tez = d.loc[d["p50, ms"].idxmin()]
    print(f"batch {b}: eng yuqori throughput — {eng.Konfiguratsiya} ({eng['inf/s']:.0f} inf/s); eng past p50 — {tez.Konfiguratsiya}")
print("\n=== OpenVINO / TensorRT dagi analoglari ===")
print(pd.DataFrame([
    ("intra_op_num_threads", "INFERENCE_NUM_THREADS", "— (GPU ichida avtomatik)"),
    ("bir nechta sessiya / oqim", "NUM_STREAMS, PERFORMANCE_HINT=THROUGHPUT", "bir nechta execution context + CUDA stream"),
    ("1 sessiya, 1 oqim", "PERFORMANCE_HINT=LATENCY", "1 context, batch 1"),
    ("dinamik paket (4-qadam)", "avtomatik paketlash (BATCH:GPU)", "Triton dynamic_batching"),
], columns=["ONNX Runtime (bu lab)", "OpenVINO", "TensorRT / Triton"]).to_string(index=False))
print("Qoida: yadrolar sonidan ko'p oqim (intra x ishchi > yadro) — ortiqcha obuna: p99 o'sadi, throughput oshmaydi.")

d = df[df.batch == 32]
plt.figure(figsize=(7, 3.3)); plt.barh([k.split(":")[0] for k in d.Konfiguratsiya], d["inf/s"], color="tab:green")
plt.xlabel("inf/s (batch 32)"); plt.title("Parallellik konfiguratsiyalari"); plt.tight_layout()
