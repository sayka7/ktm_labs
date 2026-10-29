"""A4 — 1-qadam. FastAPI inferens xizmati: probe'lar, predict, metrikalar

Kichik MLP ONNX'ga eksport qilinadi va FastAPI xizmatiga o'raladi: `lifespan` ichida model yuklanadi va isitiladi, `/healthz` (liveness) va `/readyz` (readiness) ajratilgan, `/predict` JSON so'rovni qabul qilib top-3 va vaqt taqsimotini (preprocess/inference) qaytaradi, `/metrics` Prometheus matn formatida so'rovlar, xatolar, kechikish kvantillarini beradi. TestClient model yuklanmasdan oldin (`/readyz` → 503) va keyin (`with` blok) chaqiriladi.

Kutilgan natija: Yuklanmasdan oldin /readyz 503, /healthz 200; keyin 200; 5 namunadan 4–5 tasi to'g'ri, 200 namunada ~90 % aniqlik; noto'g'ri so'rovlar 422; /metrics matni.
"""
# 1-qadam. FastAPI inferens xizmati: /healthz, /readyz, /predict, /metrics — jarayon ichida TestClient bilan
import warnings; warnings.filterwarnings("ignore")
import os, time, json, numpy as np, torch, torch.nn as nn, onnxruntime as ort
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.testclient import TestClient
from pydantic import BaseModel
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(0); torch.set_num_threads(2); torch.backends.nnpack.set_flags(False)

# --- Berilgan: A3 dan "optimallashtirilgan model" — kichik MLP -> digits.onnx (o'zgartirmang)
X, y = load_digits(return_X_y=True); X = (X / 16.0).astype(np.float32)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))
opt = torch.optim.Adam(model.parameters(), 3e-3); lossf = nn.CrossEntropyLoss()
Ttr, Tytr = torch.tensor(Xtr), torch.tensor(ytr)
for ep in range(4):
    perm = torch.randperm(len(Ttr))
    for i in range(0, len(Ttr), 64):
        idx = perm[i:i + 64]; opt.zero_grad(); lossf(model(Ttr[idx]), Tytr[idx]).backward(); opt.step()
model.eval()
torch.onnx.export(model, Ttr[:1], "digits.onnx", input_names=["input"], output_names=["logits"],
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)

# --- 12 omilli ilova: sozlamalar muhit o'zgaruvchilaridan
MODEL_PATH = os.getenv("MODEL_PATH", "digits.onnx")
ORT_THREADS = int(os.getenv("ORT_THREADS", "2"))
state = {"ready": False, "session": None, "sorovlar": 0, "xatolar": 0, "kechikish": []}

@asynccontextmanager
async def lifespan(app):
    # TODO 1: ort.InferenceSession yarating (ORT_THREADS oqim), bitta nol so'rov bilan isiting, state["ready"]=True qiling
    raise NotImplementedError("TODO 1: modelni yuklash va isitish")
    yield
    state["ready"] = False

app = FastAPI(title="digits-infer", lifespan=lifespan)

class Sorov(BaseModel):
    pixels: list[float]          # 64 ta qiymat, 0..16

# TODO 2: /healthz (har doim 200) va /readyz (model tayyor bo'lmasa 503) endpointlarini yozing

@app.post("/predict")
def predict(s: Sorov):
    # TODO 3: 64 piksel tekshiruvi (422), /16 normallash, session.run, top-3 va timing_ms (preprocess/inference) qaytaring
    raise NotImplementedError("TODO 3: predict")

@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    # TODO 4: Prometheus matni: infer_requests_total, infer_errors_total, infer_latency_ms (p50/p95), infer_ready
    raise NotImplementedError("TODO 4: metrics")

print("=== FastAPI xizmati ===")
sovuq = TestClient(app)              # lifespan ishga tushmagan — model hali yuklanmagan
r = sovuq.get("/readyz"); print(f"Model yuklanmasdan oldin GET /readyz -> {r.status_code}")
print(f"                         GET /healthz -> {sovuq.get('/healthz').status_code}")

with TestClient(app) as c:            # with-blok lifespan'ni ishga tushiradi
    print("GET /readyz ->", c.get("/readyz").status_code, c.get("/readyz").json())
    # TODO 5: 5 ta Xte namunasini (Xte[i]*16) yuboring, bashoratni yte bilan solishtiring; noto'g'ri so'rovlar (3 piksel, {"foo":1}) kodini chiqaring;
    #         200 namunada aniqlikni hisoblang, /metrics matnini va "OpenAPI yo'llari" ni chop eting
