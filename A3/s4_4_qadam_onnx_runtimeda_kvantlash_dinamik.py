"""A3 — 4-qadam. ONNX Runtime'da kvantlash: dinamik va statik (MinMax, Percentile, shovqin)

CNN ONNX modeli `quant_pre_process` dan o'tkazilib, `quantize_dynamic` va `quantize_static` (QDQ format) bilan kvantlanadi: per-channel MinMax, per-channel Percentile, per-tensor MinMax va ataylab noto'g'ri — tasodifiy shovqin bilan kalibrlash (A3 matnidagi 3-xato). Har variant uchun hajm, aniqlik, p50, throughput va Q/DQ tugunlar soni jadvalga yig'iladi.

Kutilgan natija: Hajm 17 → 11–13 KB; real kalibrlash aniqlikni saqlaydi (±0,3 p.p.), shovqinli kalibrlash 1 p.p. atrofida yo'qotadi; INT8 bu CPU'da tezroq emas.
"""
# 4-qadam. ONNX Runtime'da kvantlash: dinamik va statik (MinMax, Percentile, shovqin bilan kalibrlash)
import warnings; warnings.filterwarnings("ignore")
import io, os, time, logging, contextlib, numpy as np, pandas as pd, torch, torch.nn as nn, onnx, onnxruntime as ort
from onnxruntime.quantization import (quantize_dynamic, quantize_static, QuantType, QuantFormat,
                                      CalibrationDataReader, CalibrationMethod)
from onnxruntime.quantization.shape_inference import quant_pre_process
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

logging.disable(logging.WARNING)
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(2); torch.backends.nnpack.set_flags(False)

X, y = load_digits(return_X_y=True)
X = (X / 16.0).astype(np.float32).reshape(-1, 1, 8, 8)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
Ttr, Tte, Tytr = torch.tensor(Xtr), torch.tensor(Xte), torch.tensor(ytr)

class KichikCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.belgilar = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2))
        self.tasnif = nn.Sequential(nn.Flatten(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
    def forward(self, x):
        return self.tasnif(self.belgilar(x))

# O'qitish (2 epoxa) va eksport
model = KichikCNN(); opt = torch.optim.Adam(model.parameters(), 3e-3); lossf = nn.CrossEntropyLoss()
for ep in range(3):
    model.train(); perm = torch.randperm(len(Ttr))
    for i in range(0, len(Ttr), 64):
        idx = perm[i:i + 64]; opt.zero_grad(); lossf(model(Ttr[idx]), Tytr[idx]).backward(); opt.step()
model.eval()
torch.onnx.export(model, Tte[:1], "cnn_fp32.onnx", input_names=["input"], output_names=["logits"],
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
quant_pre_process("cnn_fp32.onnx", "cnn_prep.onnx")       # shakl inferensi + optimallashtirish (kvantlash sifati uchun)

class Reader(CalibrationDataReader):
    """Kalibrlash to'plami: ORT har safar bitta {kirish: massiv} lug'at so'raydi."""
    def __init__(self, arr, paket=32):
        self.it = iter([{"input": arr[i:i + paket]} for i in range(0, len(arr), paket)])
    def get_next(self):
        return next(self.it, None)

def statik(chiqish, kalib, usul, per_channel=True):
    with contextlib.redirect_stdout(io.StringIO()):                # kalibrator loglarini yashiramiz
        quantize_static("cnn_prep.onnx", chiqish, Reader(kalib), quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                        per_channel=per_channel, calibrate_method=usul)

print("=== ONNX Runtime kvantlash ===")
kalib = Xtr[:320]                                                   # 320 ta real namuna
shovqin = (np.random.rand(320, 1, 8, 8) * 8).astype(np.float32)     # 3-xato: real taqsimotga mos kelmaydigan shovqin (0..8, real 0..1)
quantize_dynamic("cnn_prep.onnx", "cnn_dyn.onnx", weight_type=QuantType.QInt8)
statik("cnn_s_minmax.onnx", kalib, CalibrationMethod.MinMax)
statik("cnn_s_pct.onnx", kalib, CalibrationMethod.Percentile)
statik("cnn_s_pt.onnx", kalib, CalibrationMethod.MinMax, per_channel=False)
statik("cnn_s_noise.onnx", shovqin, CalibrationMethod.MinMax)

def baho(fayl):
    so = ort.SessionOptions(); so.intra_op_num_threads = 2
    s = ort.InferenceSession(fayl, so, providers=["CPUExecutionProvider"])
    acc = (s.run(None, {"input": Xte})[0].argmax(1) == yte).mean() * 100
    x1 = Xte[:1]
    for _ in range(30): s.run(None, {"input": x1})
    t = []
    for _ in range(300):
        a = time.perf_counter(); s.run(None, {"input": x1}); t.append(time.perf_counter() - a)
    xb = Xte[:256]
    for _ in range(5): s.run(None, {"input": xb})
    a = time.perf_counter()
    for _ in range(20): s.run(None, {"input": xb})
    thr = 256 * 20 / (time.perf_counter() - a)
    ops = [n.op_type for n in onnx.load(fayl).graph.node]
    return dict(hajm_KB=round(os.path.getsize(fayl) / 1024, 1), aniqlik=round(acc, 2),
                p50_ms=round(np.percentile(np.array(t) * 1e3, 50), 3), throughput=round(thr),
                QDQ=sum(o in ("QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear") for o in ops))

rows = []
for nomi, f in (("FP32", "cnn_prep.onnx"), ("Dinamik INT8", "cnn_dyn.onnx"),
                ("Statik, per-channel, MinMax", "cnn_s_minmax.onnx"),
                ("Statik, per-channel, Percentile", "cnn_s_pct.onnx"),
                ("Statik, per-tensor, MinMax", "cnn_s_pt.onnx"),
                ("Statik, SHOVQIN bilan kalibrlash", "cnn_s_noise.onnx")):
    rows.append(dict(konfiguratsiya=nomi, **baho(f)))
df = pd.DataFrame(rows)
df["farq_pp"] = (df.aniqlik - df.aniqlik[0]).round(2)
print(df.to_string(index=False))
print("\nOperatorlar (dinamik):", [n.op_type for n in onnx.load("cnn_dyn.onnx").graph.node][:9], "...")
print("Operatorlar (statik QDQ):", [n.op_type for n in onnx.load("cnn_s_minmax.onnx").graph.node][:8], "...")
print("\nXulosa: statik kvantlash kalibrlash to'plamiga bog'liq — noto'g'ri taqsimotli shovqin bilan kalibrlangan model aniqlikni yo'qotadi;")
print("  per-tensor vs per-channel farqi kichik modelda sezilmasligi mumkin, ResNet-50 da 1-2 p.p. beradi.")
print("  INT8 tezlanishi bu CPU'da (VNNI yo'q) kutilmaydi; VNNI/AMX serverda statik INT8 2-4x beradi, GPU'da TensorRT INT8 ishlatiladi.")
