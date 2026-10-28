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

# --- A3 dan "optimallashtirilgan model": kichik MLP -> ONNX (digits, 8x8). Real labda bu resnet50_int8.onnx
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

# --- 12 omilli ilova: barcha sozlamalar muhit o'zgaruvchilaridan (Docker/K8s'da env orqali beriladi)
MODEL_PATH = os.getenv("MODEL_PATH", "digits.onnx")
ORT_THREADS = int(os.getenv("ORT_THREADS", "2"))
state = {"ready": False, "session": None, "sorovlar": 0, "xatolar": 0, "kechikish": []}

@asynccontextmanager
async def lifespan(app):
    """startup: model yuklash + isinish; shutdown: tozalash. K8s readiness probe shu holatga qaraydi."""
    t0 = time.perf_counter()
    so = ort.SessionOptions(); so.intra_op_num_threads = ORT_THREADS
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    state["session"] = ort.InferenceSession(MODEL_PATH, so, providers=["CPUExecutionProvider"])
    state["session"].run(None, {"input": np.zeros((1, 64), np.float32)})      # isinish — birinchi so'rov sekin bo'lmasin
    state["ready"] = True
    print(f"[startup] model yuklandi: {MODEL_PATH}, {ORT_THREADS} oqim, {(time.perf_counter() - t0) * 1e3:.0f} ms")
    yield
    state["ready"] = False
    print("[shutdown] xizmat to'xtadi")

app = FastAPI(title="digits-infer", lifespan=lifespan)

class Sorov(BaseModel):
    pixels: list[float]          # 64 ta qiymat, 0..16

@app.get("/healthz")             # liveness: jarayon tirik
def healthz():
    return {"ok": True}

@app.get("/readyz")              # readiness: model yuklangan va so'rov qabul qila oladi
def readyz():
    if not state["ready"]:
        raise HTTPException(503, "model not loaded")
    return {"ready": True, "model": os.path.basename(MODEL_PATH), "threads": ORT_THREADS}

@app.post("/predict")
def predict(s: Sorov):
    if not state["ready"]:
        raise HTTPException(503, "not ready")
    if len(s.pixels) != 64:
        state["xatolar"] += 1
        raise HTTPException(422, "64 ta piksel kerak")
    t0 = time.perf_counter()
    x = (np.asarray(s.pixels, np.float32) / 16.0)[None]                        # oldindan ishlov
    t1 = time.perf_counter()
    logits = state["session"].run(None, {"input": x})[0][0]                    # inferens
    t2 = time.perf_counter()
    top = logits.argsort()[-3:][::-1]
    state["sorovlar"] += 1; state["kechikish"].append((t2 - t0) * 1e3)
    return {"top3": [{"label": int(i), "score": round(float(logits[i]), 3)} for i in top],
            "timing_ms": {"preprocess": round((t1 - t0) * 1e3, 3), "inference": round((t2 - t1) * 1e3, 3)}}

@app.get("/metrics", response_class=PlainTextResponse)   # Prometheus matn formati (4 signal: so'rovlar, xatolar, kechikish, resurs)
def metrics():
    k = state["kechikish"] or [0.0]
    return (f"# TYPE infer_requests_total counter\ninfer_requests_total {state['sorovlar']}\n"
            f"# TYPE infer_errors_total counter\ninfer_errors_total {state['xatolar']}\n"
            f"# TYPE infer_latency_ms summary\ninfer_latency_ms{{quantile=\"0.5\"}} {np.percentile(k, 50):.3f}\n"
            f"infer_latency_ms{{quantile=\"0.95\"}} {np.percentile(k, 95):.3f}\n"
            f"# TYPE infer_ready gauge\ninfer_ready {int(state['ready'])}\n")

print("=== FastAPI xizmati ===")
# Lifespan ishga tushmagan mijoz — Pod endi yaratilgan, model hali yuklanmagan holat
sovuq = TestClient(app)
r = sovuq.get("/readyz"); print(f"Model yuklanmasdan oldin GET /readyz -> {r.status_code} {r.json()}  (K8s trafik yubormaydi)")
print(f"                         GET /healthz -> {sovuq.get('/healthz').status_code}  (Pod tirik — o'ldirilmaydi)")

with TestClient(app) as c:            # with-blok lifespan'ni ishga tushiradi (startup -> yield -> shutdown)
    print("GET /readyz ->", c.get("/readyz").status_code, c.get("/readyz").json())
    print("\nPOST /predict (5 ta test namunasi):")
    togri = 0
    for i in range(5):
        r = c.post("/predict", json={"pixels": (Xte[i] * 16).tolist()})
        j = r.json(); bashorat = j["top3"][0]["label"]; togri += bashorat == yte[i]
        print(f"  #{i} haqiqiy {yte[i]}  bashorat {bashorat}  {'OK' if bashorat == yte[i] else 'XATO'}  timing {j['timing_ms']}")
    print("Noto'g'ri so'rov (3 ta piksel) ->", c.post("/predict", json={"pixels": [1, 2, 3]}).status_code)
    print("Yaroqsiz JSON             ->", c.post("/predict", json={"foo": 1}).status_code)
    # Butun test to'plami bo'yicha aniqlik — xizmat orqali
    acc = np.mean([c.post("/predict", json={"pixels": (Xte[i] * 16).tolist()}).json()["top3"][0]["label"] == yte[i] for i in range(200)])
    print(f"\nXizmat orqali 200 namunada aniqlik: {acc * 100:.1f} %")
    print("\nGET /metrics:"); print(c.get("/metrics").text)
    print("OpenAPI yo'llari:", sorted(p for p in c.get("/openapi.json").json()["paths"]))
print("\nIzoh: real labda bu xizmat uvicorn bilan 8000-portda, Docker ichida ishlaydi; bu yerda TestClient uni jarayon ichida chaqiradi.")
