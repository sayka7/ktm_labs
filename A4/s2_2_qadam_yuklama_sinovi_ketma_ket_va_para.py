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

# --- Berilgan xizmat (1-qadamdagi kabi, soddalashtirilgan) — o'zgartirmang
model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10)).eval()
torch.onnx.export(model, torch.zeros(1, 64), "digits.onnx", input_names=["input"], output_names=["logits"],
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
X, _ = load_digits(return_X_y=True)
state = {"ready": False, "session": None}

@asynccontextmanager
async def lifespan(app):
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    state["session"] = ort.InferenceSession("digits.onnx", so, providers=["CPUExecutionProvider"])
    state["ready"] = True
    yield

app = FastAPI(lifespan=lifespan)
class Sorov(BaseModel):
    pixels: list[float]

@app.post("/predict")
def predict(s: Sorov):
    t0 = time.perf_counter()
    logits = state["session"].run(None, {"input": (np.asarray(s.pixels, np.float32) / 16.0)[None]})[0][0]
    return {"label": int(logits.argmax()), "inference_ms": (time.perf_counter() - t0) * 1e3}

@app.post("/predict_batch")
def predict_batch(s: list[Sorov]):
    x = np.asarray([q.pixels for q in s], np.float32) / 16.0
    return {"labels": state["session"].run(None, {"input": x})[0].argmax(1).tolist()}

def foiz(t):
    # TODO 1: soniyadagi kechikishlar ro'yxatidan ms dagi p50, p95, p99 lug'atini qaytaring
    raise NotImplementedError("TODO 1: foiz")

print("=== Yuklama sinovi ===")
with TestClient(app) as c:
    body = {"pixels": X[0].tolist()}
    for _ in range(50): c.post("/predict", json=body)                   # isinish
    # TODO 2: 300 ta ketma-ket so'rov: to'liq kechikish va javobdagi inference_ms ni yig'ing, p50/p95/p99 va so'rov/s ni chiqaring
    # TODO 3: cc in (1,2,4,8,16) uchun ThreadPoolExecutor(cc) bilan 60*cc so'rov; throughput, p50/p95/p99 va little_N ni df ga yig'ing,
    #         "Parallel mijozlar:" sarlavhasi bilan chop eting
    df = None
    # TODO 4: 64 rasmni bitta /predict_batch so'rovida va 64 ta alohida /predict so'rovida yuborib vaqtlarini solishtiring
    # TODO 5: SLO = 50 ms; har c uchun p95 <= SLO ekanini tekshirib "SLO p95 <= ..." qatorini chiqaring
    raise NotImplementedError("TODO 2-5: yuklama sinovi")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(df.mijozlar, df.sorov_s, "o-"); ax[0].set_xlabel("parallel mijozlar"); ax[0].set_ylabel("so'rov/s"); ax[0].grid(alpha=.3)
ax[1].plot(df.mijozlar, df.p95_ms, "s-", label="p95"); ax[1].axhline(50, color="r", ls="--", label="SLO"); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout()
