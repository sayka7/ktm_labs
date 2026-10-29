"""A7 — 4-qadam. FastAPI xizmati: /predict, /healthz, Prometheus metrikalari va audit jurnali

Xizmat kodi haqiqiy loyihadagidek yoziladi: API kalit, pydantic kontrakt, kirish nazorati, ONNX Runtime inferens, Prometheus matn formatidagi Counter/Histogram (o'z amalga oshirishimiz), audit jurnali (kirishning xeshi, jins/yosh sarlavhadan). TestClient bilan 300 so'rov yuboriladi — to'liq va server ichidagi kechikish p50/p99 taqqoslanadi, /metrics chiqishi va jurnal o'qiladi. O'zgartirib ko'ring: chegara, gistogramma chelaklari, noto'g'ri so'rov turlari.

Kutilgan natija: /healthz ok; 401/400/422 holatlar; 300 so'rov p50 ~4 ms (TestClient) vs ~0.1 ms (server ichida); /metrics da counter va 2 histogram; jurnalda 300 yozuv.
"""
# 4-qadam. FastAPI inferens xizmati: /predict, /healthz, /metrics (Prometheus matn formati), audit jurnali
# Docker/uvicorn o'rniga TestClient — xizmat kodi aynan shunday, faqat jarayon ichida chaqiriladi.
import os, time, json, uuid, hashlib, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch, torch.nn as nn, onnxruntime as ort
from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field
torch.manual_seed(0)

FEATS = ["opacity", "lung_area", "ct_ratio", "texture", "intensity", "edge_density"]

def sintetik_bemorlar(n_bemor=1500, seed=0):
    """1-qadamdagi generator (berilgan)."""
    rng = np.random.default_rng(seed); rows = []
    for pid in range(n_bemor):
        age = int(np.clip(rng.normal(50, 18), 1, 95)); sex = "M" if rng.random() < 0.56 else "F"
        z = rng.normal(0, 1) + 0.02 * (age - 50)
        for k in range(int(rng.integers(1, 4))):
            e = rng.normal(0, 2.2 if age < 18 else 1.0, 6)
            f = [0.9*z+e[0], -0.6*z+e[1], 0.5*z+e[2], 0.7*z+e[3], 0.2*z+e[4], 0.4*z+e[5]]
            rows.append([f"{pid:08d}_{k:03d}.png", pid, age, sex, *np.round(f, 4), int(z + rng.normal(0, 0.6) > 0.3)])
    return pd.DataFrame(rows, columns=["image_id", "patient_id", "age", "sex", *FEATS, "finding"])

# --- Model artefakti (berilgan): tez o'qitiladigan MLP -> ONNX, chiqishi ehtimol ---
df = sintetik_bemorlar(1500)
X = torch.tensor(df[FEATS].values, dtype=torch.float32); y = torch.tensor(df.finding.values, dtype=torch.float32)
mu, sd = X.mean(0), X.std(0)
net = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 1))
opt = torch.optim.Adam(net.parameters(), lr=5e-3); lossf = nn.BCEWithLogitsLoss()
for ep in range(60):
    opt.zero_grad(); loss = lossf(net((X - mu) / sd).squeeze(1), y); loss.backward(); opt.step()
class Wrap(nn.Module):
    def __init__(s): super().__init__(); s.net = net; s.register_buffer("mu", mu); s.register_buffer("sd", sd)
    def forward(s, x): return torch.sigmoid(s.net((x - s.mu) / s.sd))
os.makedirs("/tmp/triage/models", exist_ok=True); os.makedirs("/tmp/triage/logs", exist_ok=True)
ONNX = "/tmp/triage/models/triage.onnx"
torch.onnx.export(Wrap().eval(), torch.randn(1, 6), ONNX, input_names=["features"], output_names=["prob"],
                  dynamic_axes={"features": {0: "b"}, "prob": {0: "b"}}, opset_version=17, dynamo=False)

# TODO 1: Counter (inc(**labels), expose()) va Histogram (observe(x), expose(): _bucket{le=..}, +Inf, _sum, _count)
class Counter:
    def __init__(s, name, help_, labels): raise NotImplementedError("TODO 1: Counter")
class Histogram:
    def __init__(s, name, help_, buckets): raise NotImplementedError("TODO 1: Histogram")

REQ = Counter("triage_requests_total", "so'rovlar soni holat bo'yicha", ["status"])
LAT = Histogram("triage_latency_seconds", "to'liq so'rov kechikishi", [.00005, .0001, .0002, .0005, .001, .002, .005, .01])
SCORE = Histogram("triage_score", "bashorat ehtimoli taqsimoti", [i / 10 for i in range(1, 11)])

MODEL_VERSION, THRESHOLD, API_KEY = "triage-mlp-1.0", 0.30, os.environ.get("TRIAGE_API_KEY", "demo-key")
AUDIT = "/tmp/triage/logs/predictions.jsonl"
app = FastAPI(title="CXR Triage API", version=MODEL_VERSION)
sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])

# TODO 2: pydantic kontrakt — features: aynan 6 ta float
class Kirish(BaseModel):
    features: list

@app.post("/predict")
def predict(k: Kirish, x_api_key: str = Header(None), x_patient_sex: str = Header(None), x_patient_age: int = Header(None)):
    # TODO 3: API kalit -> 401; NaN yoki |x|>15 -> 400; ONNX inferens, chegara bo'yicha HIGH/ROUTINE;
    #         metrikalarni yangilang; audit jurnaliga (kirish xeshi bilan) JSON satr yozing
    raise NotImplementedError("TODO 3: /predict")

@app.get("/healthz")
def healthz():
    return {"status": "ok", "model": MODEL_VERSION, "onnx_loaded": sess is not None, "threshold": THRESHOLD}

@app.get("/metrics")
def metrics():
    return Response("\n".join(REQ.expose() + LAT.expose() + SCORE.expose()) + "\n", media_type="text/plain; version=0.0.4")

c = TestClient(app)
print("=== /healthz ===", c.get("/healthz").json())
# TODO 4: namunaviy so'rov va 401/400/422 holatlarini ko'rsating

# TODO 5: jurnal va metrikalarni tozalab, 300 ta so'rov yuboring; to'liq va server ichidagi p50/p99 ni solishtiring
print(f"\n=== Yuk sinovi: 300 so'rov ===")

print("\n=== /metrics (Prometheus matn formati) ===")
print(c.get("/metrics").text.strip())

# TODO 6: audit jurnalini pd.read_json(lines=True) bilan o'qing: yozuvlar soni, HIGH ulushi jins bo'yicha
print("\n=== Audit jurnali ===")
