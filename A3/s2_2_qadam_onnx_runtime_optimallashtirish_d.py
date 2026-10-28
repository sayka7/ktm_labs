"""A3 — 2-qadam. ONNX Runtime: optimallashtirish darajalari, paket, oqimlar, Little qonuni

ONNX Runtime'ning 4 ta grafik optimallashtirish darajasi (DISABLE_ALL, BASIC, EXTENDED, ALL) batch 1 da o'lchanadi va optimallashtirilgan grafikdagi tugunlar soni sanaladi; keyin paket 1/16/64/256 uchun throughput va Little qonuni (N = X·R), intra_op oqimlar 1 va 2 taqqoslanadi (GPU parallelligi o'rniga). Oxirida 2000 so'rov/s talabi uchun nusxalar soni rejalashtiriladi.

Kutilgan natija: ORT ALL PyTorch eager'dan 6–9 marta tez; EXTENDED/ALL da tugunlar 10 → 7–8; batch 256 da throughput 10+ marta yuqori, latency ham oshadi; 2 oqim ~1,9x.
"""
# 2-qadam. ONNX Runtime: grafik optimallashtirish darajalari, paket o'lchami, oqimlar soni va Little qonuni
import warnings; warnings.filterwarnings("ignore")
import time, math, numpy as np, pandas as pd, torch, torch.nn as nn, onnx, onnxruntime as ort

torch.manual_seed(0); torch.set_num_threads(2); torch.backends.nnpack.set_flags(False)

# Latency vaznlarga bog'liq emas, shuning uchun bu qadamda model o'qitilmaydi — faqat arxitektura.
class KichikCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.belgilar = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2))
        self.tasnif = nn.Sequential(nn.Flatten(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
    def forward(self, x):
        return self.tasnif(self.belgilar(x))

model = KichikCNN().eval()
x1 = torch.randn(1, 1, 8, 8)
torch.onnx.export(model, x1, "cnn_fp32.onnx", input_names=["input"], output_names=["logits"],
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)

def olch(fn, takror=300, isinish=30):
    """fn() ni takror marta chaqirib, kechikish taqsimotini (ms) qaytaradi."""
    for _ in range(isinish): fn()
    t = []
    for _ in range(takror):
        a = time.perf_counter(); fn(); t.append(time.perf_counter() - a)
    t = np.array(t) * 1e3
    return dict(p50=np.percentile(t, 50), p95=np.percentile(t, 95), p99=np.percentile(t, 99), mean=t.mean())

def ort_sess(opt_level, threads=2, saqla=None):
    so = ort.SessionOptions()
    so.graph_optimization_level = opt_level
    so.intra_op_num_threads = threads
    so.log_severity_level = 3                            # faqat xatolar
    if saqla: so.optimized_model_filepath = saqla        # optimallashtirilgan grafikni faylga yozib tugunlarni sanaymiz
    return ort.InferenceSession("cnn_fp32.onnx", so, providers=["CPUExecutionProvider"])

print("=== ONNX Runtime o'lchovlari ===")
print("Mavjud provayderlar:", ort.get_available_providers(), "(GPU yo'q -> faqat CPU)")
L = ort.GraphOptimizationLevel
qatorlar = []
# 0) PyTorch eager etalon
with torch.inference_mode():
    r = olch(lambda: model(x1))
qatorlar.append(dict(dvigatel="PyTorch eager", opt="-", tugun="-", **{k: round(v, 3) for k, v in r.items()}))
# 1) ORT 4 ta optimallashtirish darajasi, batch 1
xn = x1.numpy()
for nomi, lvl in (("DISABLE_ALL", L.ORT_DISABLE_ALL), ("BASIC", L.ORT_ENABLE_BASIC),
                  ("EXTENDED", L.ORT_ENABLE_EXTENDED), ("ALL", L.ORT_ENABLE_ALL)):
    s = ort_sess(lvl, saqla=f"opt_{nomi}.onnx")
    tugun = len(onnx.load(f"opt_{nomi}.onnx").graph.node)
    r = olch(lambda: s.run(None, {"input": xn}))
    qatorlar.append(dict(dvigatel="ORT CPU", opt=nomi, tugun=tugun, **{k: round(v, 3) for k, v in r.items()}))
df = pd.DataFrame(qatorlar)
print("\nBatch 1, 2 oqim (ms):"); print(df.to_string(index=False))
tez = df.loc[0, "p50"] / df.loc[4, "p50"]
print(f"ORT ALL PyTorch eager'dan {tez:.1f} marta tez (Python xarajati yo'q, grafik oldindan tuzilgan).")

# 2) Paket o'lchami va throughput (Little qonuni: tizimdagi so'rovlar = throughput x latency)
print("\n--- Paket o'lchami (ORT ALL, 2 oqim) ---")
s = ort_sess(L.ORT_ENABLE_ALL)
qat = []
for b in (1, 16, 64, 256):
    xb = np.random.rand(b, 1, 8, 8).astype(np.float32)
    r = olch(lambda: s.run(None, {"input": xb}), takror=100)
    thr = b / (r["mean"] / 1e3)
    qat.append(dict(batch=b, p50_ms=round(r["p50"], 3), p99_ms=round(r["p99"], 3), throughput=round(thr),
                    little_N=round(thr * r["mean"] / 1e3, 1)))
dfb = pd.DataFrame(qat); print(dfb.to_string(index=False))
print("little_N = throughput x latency = paketdagi so'rovlar soni (Little qonuni sinxron siklda aynan batch'ga teng).")

# 3) Oqimlar soni: 1 va 2 (GPU parallelligi o'rniga CPU parallelligi)
print("\n--- intra_op oqimlar: 1 vs 2 (batch 256) ---")
xb = np.random.rand(256, 1, 8, 8).astype(np.float32)
nat = {}
for th in (1, 2):
    s = ort_sess(L.ORT_ENABLE_ALL, threads=th)
    nat[th] = olch(lambda: s.run(None, {"input": xb}), takror=100)["mean"]
    print(f"  {th} oqim: o'rtacha {nat[th]:.3f} ms  -> {256 / nat[th] * 1e3:,.0f} rasm/s")
print(f"  Tezlanish 2 oqimda: {nat[1] / nat[2]:.2f}x (ideal 2.0x; farq — sinxronizatsiya va kichik model)")

# 4) Rejalashtirish: 2000 so'rov/s va p99 <= 2 ms talabi uchun nechta parallel nusxa kerak?
lam, p99 = 2000, df.loc[4, "p99"]
print(f"\nLittle qonuni bo'yicha reja ({lam} so'rov/s, p99 bo'yicha):")
for nomi, lat in (("bizning CNN", p99), ("ResNet-50 CPU (~20 ms)", 20.0), ("ResNet-50 T4 INT8 (~0.4 ms/rasm)", 0.4)):
    N = lam * lat / 1e3
    print(f"  {nomi:<34} {lam} x {lat:.3f} ms = {N:6.1f} parallel ishlov -> kamida {math.ceil(N)} nusxa/oqim")
print("GPU'da: ORT CUDAExecutionProvider / TensorRT'da batch 32 da throughput 5-10 marta o'sadi, latency 10+ marta oshadi.")
