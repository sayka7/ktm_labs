"""A3 — 6-qadam. Energiya bahosi, nusxalar rejasi va GPU/TensorRT yo'l xaritasi

RAPL/nvidia-smi yo'qligi sababli quvvat MODEL bilan baholanadi: P = P_bosh + (P_to'liq − P_bosh)·CPU_bandlik, bandlik `time.process_time()/devor vaqti` dan olinadi. FP32 va INT8 dinamik, 1 va 2 oqim, batch 1 va 64 uchun µJ/inferens (to'liq va dinamik) hamda inferens/s/W hisoblanadi; Little qonuni bilan 5000 so'rov/s uchun nusxalar va quvvat; oxirida GPU'da TensorRT FP16/INT8 va quvvat o'lchash qadamlari (buyruqlar va kutilgan raqamlar).

Kutilgan natija: Batch 64 da energiya batch 1 dan 5–15 marta kam; 2 oqim batch 1 da energiya ortadi (bandlik 2x, tezlik oshmaydi); ustunli grafik.
"""
# 6-qadam. Energiya bahosi (CPU vaqti asosida), nusxalar rejasi va GPU/TensorRT yo'l xaritasi
import warnings; warnings.filterwarnings("ignore")
import time, math, logging, numpy as np, pandas as pd, torch, torch.nn as nn, onnxruntime as ort
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process

logging.disable(logging.WARNING)
torch.manual_seed(0); torch.set_num_threads(2); torch.backends.nnpack.set_flags(False)

# Energiya latency kabi vaznlarga bog'liq emas — modelni o'qitmaymiz
class KichikCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.belgilar = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2))
        self.tasnif = nn.Sequential(nn.Flatten(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
    def forward(self, x):
        return self.tasnif(self.belgilar(x))

model = KichikCNN().eval(); x1 = torch.randn(1, 1, 8, 8)
torch.onnx.export(model, x1, "cnn_fp32.onnx", input_names=["input"], output_names=["logits"],
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
quant_pre_process("cnn_fp32.onnx", "cnn_prep.onnx")
quantize_dynamic("cnn_prep.onnx", "cnn_dyn.onnx", weight_type=QuantType.QInt8)

# --- Quvvat modeli. Sandbox'da RAPL/nvidia-smi yo'q, shuning uchun quvvatni MODEL bilan baholaymiz:
#     P = P_bosh + (P_tolik - P_bosh) * CPU_bandlik.  Real labda bu qiymatlar RAPL (CPU) yoki NVML (GPU) dan o'qiladi.
P_TOLIK_W = 15.0      # 2 yadro to'liq yuklangan server CPU ulushi (taxmin; o'zgartirib ko'ring)
P_BOSH_W = 5.0        # bo'sh holat

def energiya(fn, batch, takror):
    for _ in range(20): fn()
    w0, c0 = time.perf_counter(), time.process_time()      # devor vaqti va CPU vaqti (barcha oqimlar yig'indisi)
    for _ in range(takror): fn()
    wall, cpu = time.perf_counter() - w0, time.process_time() - c0
    bandlik = min(cpu / wall / 2, 1.0)                        # 2 yadroga nisbatan
    P = P_BOSH_W + (P_TOLIK_W - P_BOSH_W) * bandlik
    n = batch * takror
    return dict(latency_ms=wall / takror * 1e3, throughput=n / wall, cpu_bandlik=bandlik * 100,
                P_W=P, uJ_tolik=P * wall / n * 1e6, uJ_dinamik=(P - P_BOSH_W) * wall / n * 1e6, inf_s_W=n / wall / P)

print("=== Energiya bahosi ===")
rows = []
for nomi, fayl in (("ORT FP32", "cnn_prep.onnx"), ("ORT INT8 dinamik", "cnn_dyn.onnx")):
    for th in (1, 2):
        so = ort.SessionOptions(); so.intra_op_num_threads = th
        s = ort.InferenceSession(fayl, so, providers=["CPUExecutionProvider"])
        for b in (1, 64):
            xb = np.random.rand(b, 1, 8, 8).astype(np.float32)
            r = energiya(lambda: s.run(None, {"input": xb}), b, 200 if b == 1 else 60)
            rows.append(dict(konfiguratsiya=nomi, oqim=th, batch=b, **{k: round(v, 3 if k == "latency_ms" else 1) for k, v in r.items()}))
df = pd.DataFrame(rows); print(df.to_string(index=False))
eng = df.loc[df.uJ_tolik.idxmin()]
print(f"\nEng tejamkor: {eng.konfiguratsiya}, {eng.oqim} oqim, batch {eng.batch}: {eng.uJ_tolik} uJ/inferens (to'liq; 1 mJ = 1000 uJ), {eng.inf_s_W} inferens/s/W")
print("Kuzatuv: batch 64 da inferens boshiga energiya batch 1 ga nisbatan ancha kam — doimiy xarajat (Python, sessiya chaqiruvi) taqsimlanadi.")
print("  uJ_tolik — TCO uchun (bo'sh quvvat ham hisobga olinadi), uJ_dinamik — algoritm samaradorligi uchun. ResNet-50 da bu 10-100 mJ bo'ladi.")

# --- Little qonuni bo'yicha rejalashtirish
print("\n=== Nusxalar rejasi (Little qonuni) ===")
lam = 5000                                                   # kerakli yuklama, so'rov/s
rejalar = [(r.konfiguratsiya, r.latency_ms, r.P_W) for _, r in df[(df.batch == 1) & (df.oqim == 2)].iterrows()]
rejalar.append(("ResNet-50 FP32 CPU (taxmin 20 ms, 15 W)", 20.0, 15.0))
rejalar.append(("ResNet-50 INT8 VNNI CPU (taxmin 8 ms, 15 W)", 8.0, 15.0))
for nomi, lat, P in rejalar:
    N = lam * lat / 1e3
    print(f"  {nomi:<44} {lam} x {lat:.3f} ms = {N:7.1f} parallel -> {math.ceil(N):4d} nusxa (2 yadro), quvvat ~{math.ceil(N) * P:.0f} W")

# --- GPU/TensorRT'da bu qadam qanday bo'ladi (sandbox'da GPU yo'q — faqat yo'l xaritasi)
print("\n=== GPU'da bu qadam qanday bo'ladi ===")
gpu = pd.DataFrame([
    dict(bosqich="Dvigatel qurish", buyruq="trtexec --onnx=cnn_fp32.onnx --saveEngine=fp32.plan --minShapes=input:1x1x8x8 --optShapes=input:32x1x8x8 --maxShapes=input:256x1x8x8"),
    dict(bosqich="FP16", buyruq="trtexec ... --fp16   (Tensor Core, ResNet-50 da ~2.7x, aniqlik -0.1 p.p.)"),
    dict(bosqich="INT8", buyruq="trtexec --onnx=cnn_s_pct.onnx --int8 --fp16   (Q/DQ tugunli ONNX kalibrlashsiz qabul qilinadi)"),
    dict(bosqich="Quvvat", buyruq="nvidia-smi --query-gpu=power.draw --format=csv -lms 100  (parallel oqimda, 10-30 s davomida o'rtacha)"),
    dict(bosqich="Energiya", buyruq="mJ/rasm = P_ortacha / throughput * 1000;  rasm/s/W = throughput / P_ortacha"),
])
for _, r in gpu.iterrows():
    print(f"  {r.bosqich:<16} {r.buyruq}")
print("  Kutilgan (T4, ResNet-50, batch 8): FP32 119 mJ -> FP16 45 mJ -> INT8 27 mJ/rasm; quvvat deyarli o'zgarmaydi (~70 W), energiya throughput'ga proporsional.")

# Grafik: mJ/inferens ustunlari
fig, ax = plt.subplots(figsize=(8, 4))
lab = [f"{r.konfiguratsiya}\n{r.oqim} oqim, b={r.batch}" for _, r in df.iterrows()]
ax.bar(range(len(df)), df.uJ_tolik, color=["tab:blue" if "FP32" in l else "tab:orange" for l in lab], label="to'liq")
ax.bar(range(len(df)), df.uJ_dinamik, color="k", alpha=.35, width=.4, label="dinamik")
ax.set_xticks(range(len(df))); ax.set_xticklabels(lab, fontsize=7); ax.set_ylabel("uJ / inferens (model bo'yicha)")
ax.set_yscale("log"); ax.set_title("Energiya bahosi: to'liq va dinamik"); ax.legend(); ax.grid(axis="y", alpha=.3)
plt.tight_layout()
