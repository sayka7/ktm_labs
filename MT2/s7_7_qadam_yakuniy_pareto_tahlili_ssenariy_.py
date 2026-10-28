"""MT2 — 7-qadam. Yakuniy Pareto tahlili, ssenariy bo'yicha tanlov va TensorRT/OpenVINO yo'l xaritasi

6 konfiguratsiya (PyTorch eager, ORT FP32 1/2 oqim, INT8 dinamik, INT8 statik 1/2 oqim) bitta jadvalga yig'iladi: aniqlik, p99 (batch 1), throughput (batch 32), µJ/inferens va hajm. To'rt mezonli Pareto chegarasi aniqlanadi va uch ssenariy (bilakuzuk, kasalxona serveri, real vaqt signali) uchun cheklovlar va maqsad funksiyasi bo'yicha konfiguratsiya tanlanadi. Oxirida GPU/NPU li stendda bajariladigan trtexec, ovc/benchmark_app va quvvat o'lchash buyruqlari beriladi.

Kutilgan natija: Pareto chegarasida INT8 statik konfiguratsiyalar; bilakuzuk uchun INT8 statik 1 oqim, server uchun eng yuqori throughput'li variant tanlanadi; grafikda qizil nuqtalar — Pareto (raqamlar taxminiy).
"""
# 7-qadam. Yakuniy tahlil: aniqlik-kechikish-energiya Pareto chegarasi, ssenariy bo'yicha tanlov va TensorRT/OpenVINO yo'l xaritasi
import warnings; warnings.filterwarnings("ignore")
import os, time, logging, numpy as np, pandas as pd, torch, torch.nn as nn, onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType, QuantFormat, CalibrationDataReader, CalibrationMethod
from onnxruntime.quantization.shape_inference import quant_pre_process
from sklearn.model_selection import train_test_split
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

logging.disable(logging.WARNING); ort.set_default_logger_severity(3)
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

X, y = har_yarat(3000)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
model = oqit(HARNet(), Xtr, ytr)
torch.onnx.export(model, torch.tensor(Xte[:1]), "fp32.onnx", input_names=["x"], output_names=["logits"],
                  dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
quant_pre_process("fp32.onnx", "prep.onnx")
class Kalib(CalibrationDataReader):
    def __init__(self): self.it = iter([{"x": Xtr[i:i + 32]} for i in range(0, 320, 32)])
    def get_next(self): return next(self.it, None)
quantize_dynamic("prep.onnx", "dyn.onnx", weight_type=QuantType.QInt8)
quantize_static("prep.onnx", "int8.onnx", Kalib(), quant_format=QuantFormat.QOperator, per_channel=True,
                activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, calibrate_method=CalibrationMethod.MinMax)

P_BOSH, P_MAX, YADRO = 6.0, 28.0, 2                                # 6-qadamdagi TAXMINIY quvvat modeli
def olchov(run, batch):
    xb1, xbb = Xte[:1], Xte[:batch]
    for _ in range(30): run(xb1)
    t = []
    for _ in range(400):
        a = time.perf_counter(); run(xb1); t.append(time.perf_counter() - a)
    c0, w0 = time.process_time(), time.perf_counter(); k = 200
    for _ in range(k): run(xbb)
    cpu, wall = time.process_time() - c0, time.perf_counter() - w0
    band = min(cpu / (wall * YADRO), 1.0)
    return np.percentile(np.array(t) * 1e3, 99), k * batch / wall, (P_BOSH + (P_MAX - P_BOSH) * band) * wall / (k * batch) * 1e6
rows = []
with torch.inference_mode():
    acc = (model(torch.tensor(Xte)).argmax(1).numpy() == yte).mean()
    p99, thr, uj = olchov(lambda x: model(torch.from_numpy(x)), 32)
torch.save(model.state_dict(), "m.pt")
rows.append(("PyTorch eager FP32", "2 oqim", acc, p99, thr, uj, os.path.getsize("m.pt") / 1024))
for nom, f, intra in [("ORT FP32", "fp32.onnx", 1), ("ORT FP32", "fp32.onnx", 2), ("ORT INT8 dinamik", "dyn.onnx", 1),
                      ("ORT INT8 statik", "int8.onnx", 1), ("ORT INT8 statik", "int8.onnx", 2)]:
    so = ort.SessionOptions(); so.intra_op_num_threads = intra
    s = ort.InferenceSession(f, so, providers=["CPUExecutionProvider"])
    acc = (s.run(None, {"x": Xte})[0].argmax(1) == yte).mean()
    p99, thr, uj = olchov(lambda x: s.run(None, {"x": x}), 32)
    rows.append((nom, f"{intra} oqim", acc, p99, thr, uj, os.path.getsize(f) / 1024))
T = pd.DataFrame(rows, columns=["Konfiguratsiya", "Oqim", "Aniqlik", "p99 b1, ms", "inf/s b32", "uJ/inf b32", "Hajm, KB"])
print("=== Yakuniy jadval ===")
print(T.round(4).to_string(index=False))

# --- Pareto: kechikish (min), energiya (min), aniqlik (max), throughput (max) — hech kim uni barcha mezonda ustun bo'lmasa, optimal
def pareto(df):
    v = np.c_[df["p99 b1, ms"], df["uJ/inf b32"], -df["Aniqlik"], -df["inf/s b32"]]
    return [not any((v[j] <= v[i]).all() and (v[j] < v[i]).any() for j in range(len(v)) if j != i) for i in range(len(v))]
T["Pareto"] = pareto(T)
print("\n=== Pareto tahlili (p99, energiya, aniqlik, throughput) ===")
print(T[["Konfiguratsiya", "Oqim", "Pareto"]].to_string(index=False))
print("Pareto-optimal:", "; ".join(T[T.Pareto].Konfiguratsiya + " (" + T[T.Pareto].Oqim + ")"))

# --- Ssenariylar: cheklovlar + maqsad funksiyasi
acc0 = T.Aniqlik.iloc[1]
SSEN = [("Bilakuzuk (yiqilish signali)", "hajm <= 20 KB, aniqlik >= FP32-1 p.p.", lambda d: d[(d["Hajm, KB"] <= 20) & (d.Aniqlik >= acc0 - 0.01)], "uJ/inf b32", True),
        ("Kasalxona serveri (1000 bemor)", "aniqlik >= FP32-0,5 p.p.", lambda d: d[d.Aniqlik >= acc0 - 0.005], "inf/s b32", False),
        ("Real vaqt signali", "p99 eng kichik", lambda d: d, "p99 b1, ms", True)]
print("\n=== Ssenariy bo'yicha tanlov ===")
for nom, shart, filt, mezon, kichik in SSEN:
    d = filt(T)
    if len(d) == 0: print(f"{nom}: shartni ({shart}) hech qaysi konfiguratsiya qanoatlantirmaydi"); continue
    tan = d.loc[d[mezon].idxmin() if kichik else d[mezon].idxmax()]
    print(f"{nom}: [{shart}] -> {tan.Konfiguratsiya} ({tan.Oqim}), {mezon} = {tan[mezon]:.3f}")

print("\n=== TensorRT / OpenVINO yo'l xaritasi (GPU/NPU li stendda bajariladi) ===")
for q in ["TensorRT: trtexec --onnx=fp32.onnx --fp16 --saveEngine=har_fp16.plan  (INT8 uchun --int8 + kalibrlash keshi)",
          "TensorRT: trtexec --loadEngine=har_fp16.plan --shapes=x:32x3x128 --dumpProfile  -> latency/throughput, qatlam profili",
          "OpenVINO: ovc fp32.onnx  -> IR (.xml/.bin); benchmark_app -m fp32.xml -d CPU -hint latency | throughput",
          "OpenVINO INT8: NNCF post-training quantization (kalibrlash to'plami ~300 namuna)",
          "Quvvat: nvidia-smi --query-gpu=power.draw --format=csv -lms 100  |  Jetson: tegrastats  |  CPU: RAPL (perf stat -e power/energy-pkg/)"]:
    print(" -", q)
print("Eslatma: bu labdagi energiya — model asosidagi baho; real tizimda yuqoridagi asboblar yoki tashqi quvvat o'lchagich bilan tekshiring.")

plt.figure(figsize=(6.5, 4))
for i, r in T.iterrows():
    plt.scatter(r["p99 b1, ms"], r["uJ/inf b32"], s=4000 * (r.Aniqlik - 0.85), c="tab:red" if r.Pareto else "tab:gray", alpha=0.7)
    plt.annotate(f"{r.Konfiguratsiya} ({r.Oqim})", (r["p99 b1, ms"], r["uJ/inf b32"]), fontsize=7)
plt.xscale("log"); plt.xlabel("p99 kechikish, ms (batch 1)"); plt.ylabel("uJ/inferens (taxminiy)")
plt.title("Pareto: qizil — optimal, doira o'lchami — aniqlik"); plt.tight_layout()
