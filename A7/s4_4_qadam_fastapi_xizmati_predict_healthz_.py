"""A7 — 4-qadam. FastAPI xizmati: /predict, /healthz, Prometheus metrikalari va audit jurnali

Xizmat kodi haqiqiy loyihadagidek yoziladi: API kalit, pydantic kontrakt, kirish nazorati, ONNX Runtime inferens, Prometheus matn formatidagi Counter/Histogram (o'z amalga oshirishimiz), audit jurnali (kirishning xeshi, jins/yosh sarlavhadan). TestClient bilan 300 so'rov yuboriladi — to'liq va server ichidagi kechikish p50/p99 taqqoslanadi, /metrics chiqishi va jurnal o'qiladi. O'zgartirib ko'ring: chegara, gistogramma chelaklari, noto'g'ri so'rov turlari.

Kutilgan natija: /healthz ok; 401/400/422 holatlar; 300 so'rov p50 ~4 ms (TestClient) vs ~0.1 ms (server ichida); /metrics da counter va 2 histogram; jurnalda 300 yozuv.
"""
# 4-qadam. FastAPI inferens xizmati: /predict, /healthz, /metrics (Prometheus matn formati), audit jurnali
# Docker/uvicorn o'rniga TestClient — xizmat kodi aynan shunday, faqat tarmoq o'rniga jarayon ichida chaqiriladi.
import os, time, json, uuid, hashlib, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch, torch.nn as nn, onnxruntime as ort
from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field
torch.manual_seed(0)

FEATS = ["opacity", "lung_area", "ct_ratio", "texture", "intensity", "edge_density"]

def sintetik_bemorlar(n_bemor=1500, seed=0):
    """1-qadamdagi generator: bemor-darajali sintetik manifest."""
    rng = np.random.default_rng(seed); rows = []
    for pid in range(n_bemor):
        age = int(np.clip(rng.normal(50, 18), 1, 95)); sex = "M" if rng.random() < 0.56 else "F"
        z = rng.normal(0, 1) + 0.02 * (age - 50)
        for k in range(int(rng.integers(1, 4))):
            e = rng.normal(0, 2.2 if age < 18 else 1.0, 6)
            f = [0.9*z+e[0], -0.6*z+e[1], 0.5*z+e[2], 0.7*z+e[3], 0.2*z+e[4], 0.4*z+e[5]]
            rows.append([f"{pid:08d}_{k:03d}.png", pid, age, sex, *np.round(f, 4), int(z + rng.normal(0, 0.6) > 0.3)])
    return pd.DataFrame(rows, columns=["image_id", "patient_id", "age", "sex", *FEATS, "finding"])

# --- Model artefakti (3-qadam qisqacha): tez o'qitiladigan MLP -> ONNX ---
df = sintetik_bemorlar(1500)
X = torch.tensor(df[FEATS].values, dtype=torch.float32); y = torch.tensor(df.finding.values, dtype=torch.float32)
mu, sd = X.mean(0), X.std(0)
net = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 1))
opt = torch.optim.Adam(net.parameters(), lr=5e-3); lossf = nn.BCEWithLogitsLoss()
for ep in range(60):
    opt.zero_grad(); loss = lossf(net((X - mu) / sd).squeeze(1), y); loss.backward(); opt.step()
class Wrap(nn.Module):
    def __init__(s): super().__init__(); s.net = net; s.register_buffer("mu", mu); s.register_buffer("sd", sd)
    def forward(s, x): return torch.sigmoid(s.net((x - s.mu) / s.sd))      # grafning o'zi ehtimol qaytaradi
os.makedirs("/tmp/triage/models", exist_ok=True); os.makedirs("/tmp/triage/logs", exist_ok=True)
ONNX = "/tmp/triage/models/triage.onnx"
torch.onnx.export(Wrap().eval(), torch.randn(1, 6), ONNX, input_names=["features"], output_names=["prob"],
                  dynamic_axes={"features": {0: "b"}, "prob": {0: "b"}}, opset_version=17, dynamo=False)

# --- Prometheus uslubidagi metrikalar (prometheus_client o'rniga minimal amalga oshirish) ---
class Counter:
    def __init__(s, name, help_, labels): s.name, s.help, s.labels_, s.v = name, help_, labels, {}
    def inc(s, **lab): k = tuple(lab[l] for l in s.labels_); s.v[k] = s.v.get(k, 0) + 1
    def expose(s):
        out = [f"# HELP {s.name} {s.help}", f"# TYPE {s.name} counter"]
        for k, n in sorted(s.v.items()):
            out.append(f"{s.name}{{{','.join(f'{l}=\"{x}\"' for l, x in zip(s.labels_, k))}}} {n}")
        return out
class Histogram:
    def __init__(s, name, help_, buckets): s.name, s.help, s.b, s.cnt, s.sum, s.vals = name, help_, buckets, [0]*len(buckets), 0.0, []
    def observe(s, x):
        s.sum += x; s.vals.append(x)
        for i, ub in enumerate(s.b):
            if x <= ub: s.cnt[i] += 1
    def expose(s):
        out = [f"# HELP {s.name} {s.help}", f"# TYPE {s.name} histogram"]
        out += [f'{s.name}_bucket{{le="{ub}"}} {c}' for ub, c in zip(s.b, s.cnt)]
        out += [f'{s.name}_bucket{{le="+Inf"}} {len(s.vals)}', f"{s.name}_sum {s.sum:.6f}", f"{s.name}_count {len(s.vals)}"]
        return out

REQ = Counter("triage_requests_total", "so'rovlar soni holat bo'yicha", ["status"])
LAT = Histogram("triage_latency_seconds", "to'liq so'rov kechikishi", [.00005, .0001, .0002, .0005, .001, .002, .005, .01])
SCORE = Histogram("triage_score", "bashorat ehtimoli taqsimoti", [i / 10 for i in range(1, 11)])

# --- Xizmat ---
MODEL_VERSION, THRESHOLD, API_KEY = "triage-mlp-1.0", 0.30, os.environ.get("TRIAGE_API_KEY", "demo-key")
AUDIT = "/tmp/triage/logs/predictions.jsonl"
app = FastAPI(title="CXR Triage API", version=MODEL_VERSION)
sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])

class Kirish(BaseModel):
    """API kontrakti: 6 ta belgi (haqiqiy tizimda — tasvir fayli); qiymatlar chegaralangan."""
    features: list[float] = Field(min_length=6, max_length=6)

@app.post("/predict")
def predict(k: Kirish, x_api_key: str = Header(None), x_patient_sex: str = Header(None), x_patient_age: int = Header(None)):
    if x_api_key != API_KEY:
        REQ.inc(status="unauthorized"); raise HTTPException(401, "API kalit noto'g'ri")
    t0 = time.perf_counter()
    x = np.asarray(k.features, dtype=np.float32)[None]
    if not np.isfinite(x).all() or np.abs(x).max() > 15:            # kirish nazorati (sxema bilan bir xil chegara)
        REQ.inc(status="bad_input"); raise HTTPException(400, "belgilar diapazondan tashqarida")
    p = float(sess.run(None, {"features": x})[0][0, 0]); flag = p >= THRESHOLD
    dt = time.perf_counter() - t0
    LAT.observe(dt); SCORE.observe(p); REQ.inc(status="ok")
    rec = {"id": str(uuid.uuid4())[:8], "ts": round(time.time(), 3), "model": MODEL_VERSION, "score": round(p, 4),
           "flag": bool(flag), "latency_ms": round(dt * 1000, 2), "sex": x_patient_sex, "age": x_patient_age,
           "input_sha": hashlib.sha256(x.tobytes()).hexdigest()[:16]}   # kirishning o'zi emas, xeshi saqlanadi
    with open(AUDIT, "a") as f: f.write(json.dumps(rec) + "\n")
    return {"id": rec["id"], "score": rec["score"], "priority": "HIGH" if flag else "ROUTINE",
            "model_version": MODEL_VERSION, "disclaimer": "Faqat navbatni tartiblash uchun; tashxis emas."}

@app.get("/healthz")
def healthz():
    return {"status": "ok", "model": MODEL_VERSION, "onnx_loaded": sess is not None, "threshold": THRESHOLD}

@app.get("/metrics")
def metrics():
    return Response("\n".join(REQ.expose() + LAT.expose() + SCORE.expose()) + "\n", media_type="text/plain; version=0.0.4")

# --- Sinov: TestClient bilan 300 so'rov (Locust yuk sinovining jarayon ichidagi muqobili) ---
c = TestClient(app)
print("=== /healthz ===", c.get("/healthz").json())
print("=== /predict namunasi ===")
r = c.post("/predict", json={"features": df[FEATS].iloc[0].tolist()},
           headers={"X-API-Key": "demo-key", "X-Patient-Sex": "F", "X-Patient-Age": "54"})
print(r.status_code, r.json())
print("kalitsiz so'rov ->", c.post("/predict", json={"features": [0]*6}).status_code,
      "| noto'g'ri kirish ->", c.post("/predict", json={"features": [99]*6}, headers={"X-API-Key": "demo-key"}).status_code,
      "| 5 ta belgi ->", c.post("/predict", json={"features": [0]*5}, headers={"X-API-Key": "demo-key"}).status_code)

if os.path.exists(AUDIT): os.remove(AUDIT)
REQ.v.clear(); LAT.__init__(LAT.name, LAT.help, LAT.b); SCORE.__init__(SCORE.name, SCORE.help, SCORE.b)
sample = df.sample(300, random_state=1)
t_all = []
for _, row in sample.iterrows():
    s = time.perf_counter()
    c.post("/predict", json={"features": row[FEATS].tolist()},
           headers={"X-API-Key": "demo-key", "X-Patient-Sex": row.sex, "X-Patient-Age": str(row.age)})
    t_all.append((time.perf_counter() - s) * 1000)
c.post("/predict", json={"features": [0]*6})          # bitta ruxsatsiz — metrikada ko'rinsin
print(f"\n=== Yuk sinovi: 300 so'rov ===")
print(f"to'liq (TestClient) kechikish: p50={np.percentile(t_all,50):.2f} ms  p99={np.percentile(t_all,99):.2f} ms  "
      f"o'tkazuvchanlik ~{1000/np.mean(t_all):.0f} so'rov/s (bitta oqim)")
print(f"server ichidagi kechikish:     p50={np.percentile(LAT.vals,50)*1000:.3f} ms  p99={np.percentile(LAT.vals,99)*1000:.3f} ms")
print("-> tizim kechikishi >> model kechikishi (HTTP/JSON qayta ishlash ulushi katta)")

print("\n=== /metrics (Prometheus matn formati) ===")
print(c.get("/metrics").text.strip())

log = pd.read_json(AUDIT, lines=True)
print("\n=== Audit jurnali ===")
print(f"fayl: {AUDIT}, yozuvlar: {len(log)}, ustunlar: {list(log.columns)}")
print(log[["id", "model", "score", "flag", "latency_ms", "sex", "age", "input_sha"]].head(3).to_string(index=False))
print("HIGH ulushi:", round(log.flag.mean(), 3), "| jins bo'yicha HIGH ulushi:", log.groupby("sex").flag.mean().round(3).to_dict())
print("\nOpenAPI yo'llari:", sorted(p for p in c.get("/openapi.json").json()["paths"]))
