"""MT2 — 3-qadam. Hisoblash aniqligi formatlari: FP32, FP16/BF16, INT8 dinamik va statik

O'qitilgan model 6 ko'rinishda solishtiriladi: ORT FP32, FP16 vazn yaxlitlash simulyatsiyasi (TensorRT --fp16 / OpenVINO FP16 IR ning sonli ta'siri), INT8 dinamik, INT8 statik QDQ (MinMax va Percentile) va QOperator (QLinearConv) formatlari, qo'shimcha ravishda PyTorch BF16 autocast. Aniqlikdan tashqari kvantlash xatosining sezgirroq o'lchovlari — FP32 bilan bashorat mosligi, max|Δlogit| va KL divergensiya — hamda hajm, batch 1 kechikishi va batch 64 throughput o'lchanadi.

Kutilgan natija: Aniqlik barcha formatlarda ~0,92 (±0,5 p.p.), moslik >= 99%; INT8 statik hajmi ~16–24 KB (FP32 ~33 KB), throughput ~1,9x; dinamik INT8 bu modelda ~0,4x sekin (taxminiy).
"""
# 3-qadam. Hisoblash aniqligi formatlari: FP32, FP16/BF16 (simulyatsiya), INT8 dinamik va statik (QDQ / QOperator)
import warnings; warnings.filterwarnings("ignore")
import os, io, time, logging, contextlib, numpy as np, pandas as pd, torch, torch.nn as nn, onnx, onnxruntime as ort
from onnx import numpy_helper
from onnxruntime.quantization import (quantize_dynamic, quantize_static, QuantType, QuantFormat, CalibrationDataReader, CalibrationMethod)
from onnxruntime.quantization.shape_inference import quant_pre_process
from sklearn.model_selection import train_test_split

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

# --- FP16 vazn simulyatsiyasi: vaznlarni float16 ga yaxlitlab qaytaramiz (TensorRT --fp16 / OpenVINO FP16 IR ning sonli ta'siri)
m = onnx.load("fp32.onnx")
for init in m.graph.initializer:
    w = numpy_helper.to_array(init)
    if w.dtype == np.float32: init.CopyFrom(numpy_helper.from_array(w.astype(np.float16).astype(np.float32), init.name))
onnx.save(m, "fp16sim.onnx")

class Kalib(CalibrationDataReader):
    def __init__(self, data): self.it = iter([{"x": data[i:i + 32]} for i in range(0, len(data), 32)])
    def get_next(self): return next(self.it, None)
kal = Xtr[:320]
quantize_dynamic("prep.onnx", "int8_dyn.onnx", weight_type=QuantType.QInt8)
quantize_static("prep.onnx", "int8_qdq_minmax.onnx", Kalib(kal), quant_format=QuantFormat.QDQ, per_channel=True,
                activation_type=QuantType.QInt8, weight_type=QuantType.QInt8, calibrate_method=CalibrationMethod.MinMax)
with contextlib.redirect_stdout(io.StringIO()):            # kalibrlash jarayoni chiqaradigan matnni yashiramiz
    quantize_static("prep.onnx", "int8_qdq_pct.onnx", Kalib(kal), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8, calibrate_method=CalibrationMethod.Percentile,
                    extra_options={"CalibPercentile": 99.99})
quantize_static("prep.onnx", "int8_qop.onnx", Kalib(kal), quant_format=QuantFormat.QOperator, per_channel=True,
                activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, calibrate_method=CalibrationMethod.MinMax)

def sess(f):
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    return ort.InferenceSession(f, so, providers=["CPUExecutionProvider"])
def lat(s, xb, n):
    for _ in range(20): s.run(None, {"x": xb})
    t = []
    for _ in range(n):
        a = time.perf_counter(); s.run(None, {"x": xb}); t.append(time.perf_counter() - a)
    return np.array(t)
def softmax(z): e = np.exp(z - z.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)

ref = sess("fp32.onnx").run(None, {"x": Xte})[0]; pref = softmax(ref)
rows = []
cfg = [("ORT FP32", "fp32.onnx"), ("FP16 vazn (simul.)", "fp16sim.onnx"), ("INT8 dinamik", "int8_dyn.onnx"),
       ("INT8 statik QDQ MinMax", "int8_qdq_minmax.onnx"), ("INT8 statik QDQ Percentile", "int8_qdq_pct.onnx"), ("INT8 statik QOperator", "int8_qop.onnx")]
for nom, f in cfg:
    s = sess(f); out = s.run(None, {"x": Xte})[0]; p = softmax(out)
    kl = float(np.mean(np.sum(pref * np.log((pref + 1e-9) / (p + 1e-9)), 1)))
    t1 = lat(s, Xte[:1], 300); t64 = lat(s, Xte[:64], 40)
    rows.append((nom, os.path.getsize(f) / 1024, (out.argmax(1) == yte).mean(), (out.argmax(1) == ref.argmax(1)).mean(),
                 np.abs(out - ref).max(), kl, np.percentile(t1, 50) * 1e3, 64 / np.median(t64)))
# BF16: PyTorch autocast (Intel AMX / ARM bf16 li CPU'larda tezlashadi; bu yerda faqat sonli ta'sir)
with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
    ob = model(torch.tensor(Xte)).float().numpy()
pb = softmax(ob)
rows.append(("BF16 autocast (PyTorch)", np.nan, (ob.argmax(1) == yte).mean(), (ob.argmax(1) == ref.argmax(1)).mean(),
             np.abs(ob - ref).max(), float(np.mean(np.sum(pref * np.log((pref + 1e-9) / (pb + 1e-9)), 1))), np.nan, np.nan))
print("=== Aniqlik formatlari taqqosi ===")
df = pd.DataFrame(rows, columns=["Konfiguratsiya", "Hajm, KB", "Aniqlik", "FP32 bilan moslik", "max|dlogit|", "KL", "p50 b1, ms", "inf/s b64"])
print(df.round(4).to_string(index=False))
print("\n=== Operatorlar (statik QDQ va QOperator) ===")
for f in ("int8_qdq_minmax.onnx", "int8_qop.onnx"):
    ops = pd.Series([n.op_type for n in onnx.load(f).graph.node]).value_counts()
    print(f, {k: int(v) for k, v in ops.items()})
print("Izoh: «moslik» va KL — kvantlash xatosining aniqlikdan sezgirroq o'lchovi (aniqlik tasodifan saqlanib qolishi mumkin).")
tz = df.set_index("Konfiguratsiya")["inf/s b64"]
print(f"Bu CPU'da: statik INT8 throughput FP32 ga nisbatan {tz['INT8 statik QOperator'] / tz['ORT FP32']:.1f}x, dinamik INT8 {tz['INT8 dinamik'] / tz['ORT FP32']:.1f}x")
print("(dinamik kvantlash har chaqiruvda faollashuv masshtabini hisoblaydi — kichik Conv modelda bu ustama foydadan katta).")
print("VNNI/AMX li Intel CPU (OpenVINO) yoki Tensor Core'li GPU (TensorRT) da INT8 farqi odatda kattaroq bo'ladi (taxminiy, o'lchab ko'rish shart).")
