"""A3 — 5-qadam. Yakuniy jadval, Pareto tahlili va grafik

6 ta konfiguratsiya (PyTorch FP32, PyTorch INT8 dinamik, ORT FP32, ORT INT8 dinamik, ORT statik MinMax/Percentile) bitta jadvalga yig'iladi: hajm, aniqlik, p50 (batch 1), throughput (batch 256), tezlanish. Pareto-optimal nuqtalar (latency kichik, aniqlik katta) aniqlanadi va ikki o'qli grafik chiziladi: latency–aniqlik va hajm–throughput.

Kutilgan natija: ORT variantlari PyTorch'dan 4–7 marta tez; Pareto chegarasida ORT FP32 va statik Percentile; grafikda qizil nuqtalar — Pareto.
"""
# 5-qadam. Yakuniy jadval: aniqlik vs tezlik vs hajm, Pareto tahlili va grafik
import warnings; warnings.filterwarnings("ignore")
import io, os, time, logging, contextlib, numpy as np, pandas as pd, torch, torch.nn as nn, onnxruntime as ort
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from torch.ao.quantization import quantize_dynamic as torch_qdyn
from onnxruntime.quantization import (quantize_dynamic, quantize_static, QuantType, QuantFormat,
                                      CalibrationDataReader, CalibrationMethod)
from onnxruntime.quantization.shape_inference import quant_pre_process
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

logging.disable(logging.WARNING)
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(2); torch.backends.nnpack.set_flags(False)
torch.backends.quantized.engine = "qnnpack"      # sandbox CPU'da AVX yo'q (3-qadamga qarang)

X, y = load_digits(return_X_y=True)
X = (X / 16.0).astype(np.float32).reshape(-1, 1, 8, 8)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
Ttr, Tte, Tytr, Tyte = torch.tensor(Xtr), torch.tensor(Xte), torch.tensor(ytr), torch.tensor(yte)

class KichikCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.belgilar = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2))
        self.tasnif = nn.Sequential(nn.Flatten(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
    def forward(self, x):
        return self.tasnif(self.belgilar(x))

model = KichikCNN(); opt = torch.optim.Adam(model.parameters(), 3e-3); lossf = nn.CrossEntropyLoss()
for ep in range(3):
    model.train(); perm = torch.randperm(len(Ttr))
    for i in range(0, len(Ttr), 64):
        idx = perm[i:i + 64]; opt.zero_grad(); lossf(model(Ttr[idx]), Tytr[idx]).backward(); opt.step()
model.eval()
torch.onnx.export(model, Tte[:1], "cnn_fp32.onnx", input_names=["input"], output_names=["logits"],
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
quant_pre_process("cnn_fp32.onnx", "cnn_prep.onnx")

class Reader(CalibrationDataReader):
    def __init__(self, arr, paket=32):
        self.it = iter([{"input": arr[i:i + paket]} for i in range(0, len(arr), paket)])
    def get_next(self):
        return next(self.it, None)

quantize_dynamic("cnn_prep.onnx", "cnn_dyn.onnx", weight_type=QuantType.QInt8)
for nomi, usul in (("minmax", CalibrationMethod.MinMax), ("pct", CalibrationMethod.Percentile)):
    with contextlib.redirect_stdout(io.StringIO()):
        quantize_static("cnn_prep.onnx", f"cnn_s_{nomi}.onnx", Reader(Xtr[:320]), quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True, calibrate_method=usul)

def olch(fn, x1, xb):
    """p50 latency (batch 1, ms) va throughput (batch 256, rasm/s)."""
    for _ in range(30): fn(x1)
    t = []
    for _ in range(300):
        a = time.perf_counter(); fn(x1); t.append(time.perf_counter() - a)
    for _ in range(5): fn(xb)
    a = time.perf_counter()
    for _ in range(20): fn(xb)
    return np.percentile(np.array(t) * 1e3, 50), 256 * 20 / (time.perf_counter() - a)

def torch_hajm(m):
    b = io.BytesIO(); torch.save(m.state_dict(), b); return b.tell() / 1024

rows = []
# PyTorch variantlari
qmodel = torch_qdyn(model, {nn.Linear}, dtype=torch.qint8)
for nomi, m in (("PyTorch FP32", model), ("PyTorch INT8 dinamik (Linear)", qmodel)):
    with torch.inference_mode():
        acc = (m(Tte).argmax(1) == Tyte).float().mean().item() * 100
        p50, thr = olch(lambda x: m(x), Tte[:1], Tte[:256])
    rows.append(dict(konfiguratsiya=nomi, hajm_KB=round(torch_hajm(m), 1), aniqlik=round(acc, 2), p50_ms=round(p50, 3), throughput=round(thr)))
# ONNX Runtime variantlari
for nomi, f in (("ORT FP32 (ALL)", "cnn_prep.onnx"), ("ORT INT8 dinamik", "cnn_dyn.onnx"),
                ("ORT INT8 statik MinMax", "cnn_s_minmax.onnx"), ("ORT INT8 statik Percentile", "cnn_s_pct.onnx")):
    so = ort.SessionOptions(); so.intra_op_num_threads = 2
    s = ort.InferenceSession(f, so, providers=["CPUExecutionProvider"])
    acc = (s.run(None, {"input": Xte})[0].argmax(1) == yte).mean() * 100
    p50, thr = olch(lambda x: s.run(None, {"input": x}), Xte[:1], Xte[:256])
    rows.append(dict(konfiguratsiya=nomi, hajm_KB=round(os.path.getsize(f) / 1024, 1), aniqlik=round(acc, 2), p50_ms=round(p50, 3), throughput=round(thr)))

df = pd.DataFrame(rows)
df["tezlanish"] = (df.p50_ms[0] / df.p50_ms).round(2)
print("=== Yakuniy jadval ===")
print(df.to_string(index=False))

# Pareto: latency kichik, aniqlik katta — hech kim ikkala ko'rsatkichda ham yaxshiroq bo'lmagan nuqtalar
def pareto(df):
    opt = []
    for i, r in df.iterrows():
        ustun = ((df.p50_ms <= r.p50_ms) & (df.aniqlik >= r.aniqlik) & ((df.p50_ms < r.p50_ms) | (df.aniqlik > r.aniqlik))).any()
        if not ustun: opt.append(r.konfiguratsiya)
    return opt
print("\n=== Pareto tahlili ===")
par = pareto(df)
print("Pareto-optimal (latency-aniqlik):", par)
eng_tez = df.loc[df.p50_ms.idxmin()]; eng_kichik = df.loc[df.hajm_KB.idxmin()]
print(f"Eng tez: {eng_tez.konfiguratsiya} ({eng_tez.p50_ms} ms, {eng_tez.tezlanish}x);  eng kichik: {eng_kichik.konfiguratsiya} ({eng_kichik.hajm_KB} KB)")
print("Tavsiya: tibbiy tasniflash (aniqlik birinchi) -> ORT FP32; chekka qurilma (hajm/energiya birinchi) -> statik INT8;")
print("  server CPU'da VNNI bo'lsa statik INT8 ham tezroq bo'ladi; GPU'da TensorRT FP16 odatda Pareto chegarasida.")

# Grafik: ikki o'q — latency vs aniqlik, hajm vs throughput
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for _, r in df.iterrows():
    rang = "tab:red" if r.konfiguratsiya in par else "tab:blue"
    ax[0].scatter(r.p50_ms, r.aniqlik, s=70, color=rang); ax[0].annotate(r.konfiguratsiya, (r.p50_ms, r.aniqlik), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax[1].scatter(r.hajm_KB, r.throughput, s=70, color=rang); ax[1].annotate(r.konfiguratsiya, (r.hajm_KB, r.throughput), fontsize=7, xytext=(4, 3), textcoords="offset points")
ax[0].set_xlabel("p50 latency, ms (batch 1)"); ax[0].set_ylabel("Test aniqlik, %"); ax[0].set_title("Latency - aniqlik (qizil = Pareto)"); ax[0].set_xscale("log"); ax[0].grid(alpha=.3)
ax[1].set_xlabel("Model hajmi, KB"); ax[1].set_ylabel("Throughput, rasm/s (batch 256)"); ax[1].set_title("Hajm - throughput"); ax[1].grid(alpha=.3)
plt.tight_layout()
