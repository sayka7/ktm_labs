"""A4 — 2-qadam. Yuklama sinovi: ketma-ket va parallel so'rovlar, Little qonuni, paketlash

`hey -c N` o'rniga `ThreadPoolExecutor` bilan 1, 2, 4, 8, 16 parallel mijoz simulyatsiya qilinadi: throughput, p50/p95/p99 va Little qonuni (N = X·R) tekshiriladi; to'liq kechikish ichidagi sof inferens ulushi ko'rsatiladi; 64 ta alohida so'rov va bitta paketli so'rov solishtiriladi; p95 ≤ 50 ms SLO tekshiriladi. Grafik: throughput va kechikish.

Kutilgan natija: c=1: p50 ~2–3 ms, ~400 so'rov/s (sof inferens 0,1 ms — qolgani HTTP/JSON); throughput 2 yadroda to'yinadi, p99 mijozlar bilan chiziqli o'sadi; paket 20–30x tezroq.
"""
# 2-qadam. Yuklama sinovi: ketma-ket va parallel so'rovlar, p50/p95/p99, throughput, Little qonuni
import warnings; warnings.filterwarnings("ignore")
import os, time, numpy as np, pandas as pd, torch, torch.nn as nn, onnxruntime as ort
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel
from sklearn.datasets import load_digits

torch.manual_seed(0); torch.set_num_threads(2); torch.backends.nnpack.set_flags(False)

# Model: latency uchun o'qitish shart emas — tasodifiy vaznli MLP -> ONNX
model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10)).eval()
torch.onnx.export(model, torch.zeros(1, 64), "digits.onnx", input_names=["input"], output_names=["logits"],
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
X, _ = load_digits(return_X_y=True)

state = {"ready": False, "session": None}

@asynccontextmanager
async def lifespan(app):
    so = ort.SessionOptions(); so.intra_op_num_threads = 1       # har so'rov 1 oqim — parallel so'rovlar bir-biriga xalaqit bermasin
    state["session"] = ort.InferenceSession("digits.onnx", so, providers=["CPUExecutionProvider"])
    state["ready"] = True
    yield

app = FastAPI(lifespan=lifespan)

class Sorov(BaseModel):
    pixels: list[float]

@app.post("/predict")
def predict(s: Sorov):
    if not state["ready"]: raise HTTPException(503)
    t0 = time.perf_counter()
    x = (np.asarray(s.pixels, np.float32) / 16.0)[None]
    logits = state["session"].run(None, {"input": x})[0][0]
    return {"label": int(logits.argmax()), "inference_ms": (time.perf_counter() - t0) * 1e3}

@app.post("/predict_batch")             # paketli so'rov: bir HTTP chaqiruvda N ta rasm
def predict_batch(s: list[Sorov]):
    x = np.asarray([q.pixels for q in s], np.float32) / 16.0
    return {"labels": state["session"].run(None, {"input": x})[0].argmax(1).tolist()}

def foiz(t):
    t = np.array(t) * 1e3
    return dict(p50=np.percentile(t, 50), p95=np.percentile(t, 95), p99=np.percentile(t, 99))

print("=== Yuklama sinovi ===")
with TestClient(app) as c:
    body = {"pixels": X[0].tolist()}
    for _ in range(50): c.post("/predict", json=body)                   # isinish

    # 1) Ketma-ket 300 so'rov: to'liq kechikish va uning ichidagi sof inferens
    t, inf = [], []
    for _ in range(300):
        a = time.perf_counter(); r = c.post("/predict", json=body); t.append(time.perf_counter() - a)
        inf.append(r.json()["inference_ms"])
    f = foiz(t)
    print(f"Ketma-ket (c=1): p50 {f['p50']:.2f} ms  p95 {f['p95']:.2f} ms  p99 {f['p99']:.2f} ms  -> {1 / np.mean(t):.0f} so'rov/s")
    print(f"  shundan sof inferens p50 {np.percentile(inf, 50):.3f} ms; qolgani ({f['p50'] - np.percentile(inf, 50):.2f} ms) — HTTP, JSON, pydantic validatsiya")

    # 2) Parallel mijozlar (hey -c N o'rniga ThreadPoolExecutor): throughput va kechikish
    rows = []
    for cc in (1, 2, 4, 8, 16):
        n = 60 * cc
        def bitta(_):
            a = time.perf_counter(); c.post("/predict", json=body); return time.perf_counter() - a
        a0 = time.perf_counter()
        with ThreadPoolExecutor(cc) as ex: tt = list(ex.map(bitta, range(n)))
        wall = time.perf_counter() - a0; f = foiz(tt); thr = n / wall
        rows.append(dict(mijozlar=cc, sorov_s=round(thr), p50_ms=round(f["p50"], 2), p95_ms=round(f["p95"], 2), p99_ms=round(f["p99"], 2),
                         little_N=round(thr * np.mean(tt), 1)))
    df = pd.DataFrame(rows); print("\nParallel mijozlar:"); print(df.to_string(index=False))
    print("little_N = throughput x o'rtacha kechikish = tizimdagi so'rovlar soni (Little qonuni) — mijozlar soniga yaqin bo'lishi kerak.")
    print("Kuzatuv: 2 yadro va GIL tufayli throughput tez to'yinadi, mijozlar ko'paygani sari faqat navbat (p99) o'sadi.")

    # 3) Paketlash: 64 ta rasm bitta so'rovda vs 64 ta alohida so'rov
    paket = [{"pixels": X[i].tolist()} for i in range(64)]
    a = time.perf_counter()
    for _ in range(10): c.post("/predict_batch", json=paket)
    tb = (time.perf_counter() - a) / 10
    a = time.perf_counter()
    for _ in range(10):
        for q in paket: c.post("/predict", json=q)
    ts = (time.perf_counter() - a) / 10
    print(f"\n64 rasm: alohida so'rovlar {ts * 1e3:.1f} ms, bitta paketli so'rov {tb * 1e3:.1f} ms -> {ts / tb:.1f}x tezroq (lekin paket to'lguncha kutish kerak).")

    # 4) SLO tekshiruvi
    SLO = 50.0
    print(f"\nSLO p95 <= {SLO} ms: ", ", ".join(f"c={int(r.mijozlar)}: {'OK' if r.p95_ms <= SLO else 'BUZILDI'}" for _, r in df.iterrows()))

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(df.mijozlar, df.sorov_s, "o-"); ax[0].set_xlabel("parallel mijozlar"); ax[0].set_ylabel("so'rov/s"); ax[0].set_title("Throughput"); ax[0].grid(alpha=.3)
ax[1].plot(df.mijozlar, df.p50_ms, "o-", label="p50"); ax[1].plot(df.mijozlar, df.p95_ms, "s-", label="p95"); ax[1].plot(df.mijozlar, df.p99_ms, "^-", label="p99")
ax[1].axhline(SLO, color="r", ls="--", label="SLO"); ax[1].set_xlabel("parallel mijozlar"); ax[1].set_ylabel("ms"); ax[1].set_title("Kechikish"); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout()
print("\nReal labda: hey -z 30s -c 8 -m POST ... yoki locust; bu yerda TestClient + oqimlar. Docker --cpus=2 cheklovi sandbox'ning 2 yadrosiga mos.")
