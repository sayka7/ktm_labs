"""MT2 — 2-qadam. ONNX eksport, grafik optimallashtirish va operatorlar profili

Model ONNX'ga (opset 17, dinamik batch) eksport qilinadi. ORT ning 4 ta optimallashtirish darajasi uchun optimallashtirilgan grafik faylga saqlanib, tugunlar va operatorlar ro'yxati (Conv+Relu → FusedConv) solishtiriladi — bu TensorRT'ning engine qurishdagi va OpenVINO'ning kompilyatsiyadagi qatlam birlashtirishiga o'xshaydi. So'ngra `enable_profiling` bilan har operatorning vaqt ulushi hisoblanadi (trtexec --dumpProfile analogi) va Amdahl qonuni bo'yicha optimallashtirish nuqtasi aniqlanadi.

Kutilgan natija: 14 tugun → 11 (FusedConv:3); ORT PyTorch eager'dan ~8 marta tez (~0,07 ms); profilda Conv/FusedConv vaqtning ~80% ini oladi (taxminiy).
"""
# 2-qadam. ONNX eksport, ORT grafik optimallashtirish (fusion) va operatorlar bo'yicha profil
import warnings; warnings.filterwarnings("ignore")
import os, json, time, glob, numpy as np, pandas as pd, torch, torch.nn as nn, onnx, onnxruntime as ort

torch.manual_seed(0); np.random.seed(0)
torch.set_num_threads(2); torch.backends.nnpack.set_flags(False)
ort.set_default_logger_severity(3)                 # ORT ogohlantirishlarini yashiramiz

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

# Kechikish va profil vaznlarga bog'liq emas — bu qadamda model o'qitilmaydi (faqat arxitektura)
model = HARNet().eval(); x1 = torch.randn(1, 3, 128)
torch.onnx.export(model, x1, "har.onnx", input_names=["x"], output_names=["logits"],
                  dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
m = onnx.load("har.onnx"); onnx.checker.check_model(m)
def operatorlar(mm):
    s = pd.Series([n.op_type for n in mm.graph.node]).value_counts(); return {k: int(v) for k, v in s.items()}
print("=== ONNX eksport ===")
print(f"Fayl: {os.path.getsize('har.onnx') / 1024:.1f} KB, tugunlar: {len(m.graph.node)}, operatorlar: {operatorlar(m)}")

# --- Optimallashtirilgan grafikni faylga saqlab, fusion natijasini ko'ramiz (TensorRT «layer fusion» ning CPU analogi)
rows = []
xb = np.random.randn(1, 3, 128).astype(np.float32)
for nom, lv in [("DISABLE_ALL", ort.GraphOptimizationLevel.ORT_DISABLE_ALL), ("BASIC", ort.GraphOptimizationLevel.ORT_ENABLE_BASIC),
                ("EXTENDED", ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED), ("ALL", ort.GraphOptimizationLevel.ORT_ENABLE_ALL)]:
    so = ort.SessionOptions(); so.graph_optimization_level = lv; so.intra_op_num_threads = 1
    so.optimized_model_filepath = f"har_{nom}.onnx"
    s = ort.InferenceSession("har.onnx", so, providers=["CPUExecutionProvider"])
    for _ in range(50): s.run(None, {"x": xb})
    t = []
    for _ in range(500):
        a = time.perf_counter(); s.run(None, {"x": xb}); t.append(time.perf_counter() - a)
    t = np.array(t) * 1e3; om = onnx.load(f"har_{nom}.onnx")
    rows.append((nom, len(om.graph.node), np.percentile(t, 50), np.percentile(t, 99), ", ".join(f"{k}:{v}" for k, v in operatorlar(om).items())))
print("\n=== Optimallashtirish darajalari (batch 1, 1 oqim) ===")
print(pd.DataFrame(rows, columns=["Daraja", "Tugun", "p50, ms", "p99, ms", "Operatorlar"]).round(4).to_string(index=False))
with torch.inference_mode():
    tt = []
    for i in range(550):
        a = time.perf_counter(); model(x1); tt.append(time.perf_counter() - a)
pt50 = np.percentile(np.array(tt[50:]) * 1e3, 50)
print(f"Taqqos: PyTorch eager p50 = {pt50:.4f} ms -> ORT ALL {pt50 / rows[-1][2]:.1f} marta tez (Python/dispatcher ustamasi yo'q)")
print("Izoh: BatchNorm eksportdayoq (eval + constant folding) Conv vazniga qo'shilgan; EXTENDED da Conv+Relu -> FusedConv.")
print("TensorRT ham xuddi shunday (va kengroq) qatlam birlashtirishni engine qurishda, OpenVINO esa model kompilyatsiyasida bajaradi.")

# --- Profil: qaysi operator qancha vaqt oladi (TensorRT: trtexec --dumpProfile, OpenVINO: benchmark_app -pc analogi)
so = ort.SessionOptions(); so.enable_profiling = True; so.intra_op_num_threads = 1
s = ort.InferenceSession("har.onnx", so, providers=["CPUExecutionProvider"])
xb64 = np.random.randn(64, 3, 128).astype(np.float32)
for _ in range(100): s.run(None, {"x": xb64})
prof = s.end_profiling()
ev = [e for e in json.load(open(prof)) if e.get("cat") == "Node" and e["name"].endswith("_kernel_time")]
df = pd.DataFrame([(e["args"]["op_name"], e["dur"]) for e in ev], columns=["op", "dur_us"])
agg = df.groupby("op").dur_us.agg(["sum", "count"]).sort_values("sum", ascending=False)
agg["ulush"] = agg["sum"] / agg["sum"].sum(); agg["o'rtacha_us"] = agg["sum"] / agg["count"]
print("\n=== ORT profil (operatorlar bo'yicha, batch 64, 100 chaqiruv) ===")
print(agg[["count", "o'rtacha_us", "ulush"]].round(3).to_string())
eng = agg.index[0]
print(f"Eng og'ir operator: {eng} ({agg.ulush.iloc[0]:.0%}) — optimallashtirishni shundan boshlash kerak (Amdahl qonuni).")
for f in glob.glob("onnxruntime_profile_*.json"): os.remove(f)
