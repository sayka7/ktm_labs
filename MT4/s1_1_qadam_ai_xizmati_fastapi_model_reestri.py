"""MT4 — 1-qadam. AI xizmati: FastAPI, model reestri, API kalit, probe'lar va avtomatik testlar

Loyiha — «tumor-api»: sklearn breast_cancer (30 belgi) bo'yicha o'sma turini tasniflovchi xizmat. Ikki model versiyasi (v1 — logistik regressiya, v2 — tasodifiy o'rmon) o'qitilib, SHA-256 xeshi bilan reestrga (/tmp/models) yoziladi. Xizmat `lifespan` ichida modelni butunlik tekshiruvi va isitish bilan yuklaydi; /startupz, /livez, /readyz alohida, /v1/predict va /v1/predict/batch API kalit (x-api-key) talab qiladi, har javobga x-request-id va x-model-version sarlavhalari qo'shiladi, /metrics Prometheus matn formatida. Oxirida CI'dagi kabi 11 ta avtomatik test TestClient bilan bajariladi. MODEL_VERSION, MAX_BATCH ni o'zgartirib ko'ring.

Kutilgan natija: v1 aniqligi ~0,96, v2 ~0,94; 11/11 PASS; ketma-ket ~300-400 so'rov/s; /metrics da predictions_total 133 va model_info.
"""
# 1-qadam. AI xizmati: FastAPI, model reestri (v1/v2), API kalit, so'rov ID, batch, probe'lar va avtomatik testlar
# Loyiha: "tumor-api" — ko'krak bezi o'smasi belgilarini (sklearn breast_cancer, 30 belgi) tasniflovchi xizmat.
# Docker/K8s o'rniga xizmat jarayon ichida TestClient bilan sinaladi — konteynerda ham aynan shu kod ishlaydi.
import warnings; warnings.filterwarnings("ignore")
import hashlib, json, os, pickle, time, uuid
from contextlib import asynccontextmanager
from typing import List

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import PlainTextResponse
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field, field_validator
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# --- Model reestri: ikki versiya o'qitiladi va /tmp/models/<versiya>/model.pkl ga yoziladi (real loyihada — NAS/S3) ---
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
REESTR = "/tmp/models"
for ver, est in (("v1", make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))),
                 ("v2", RandomForestClassifier(n_estimators=60, random_state=0))):
    est.fit(Xtr, ytr); os.makedirs(f"{REESTR}/{ver}", exist_ok=True)
    blob = pickle.dumps(est); open(f"{REESTR}/{ver}/model.pkl", "wb").write(blob)
    meta = {"versiya": ver, "aniqlik": float(est.score(Xte, yte)), "sha256": hashlib.sha256(blob).hexdigest()[:16]}
    json.dump(meta, open(f"{REESTR}/{ver}/meta.json", "w"))
    print(f"reestr: {ver} aniqlik={meta['aniqlik']:.4f} sha256={meta['sha256']}")

# --- Sozlamalar muhit o'zgaruvchilaridan (12 omil) ---
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
API_KEY = os.getenv("API_KEY", "maxfiy-demo-kalit")      # real loyihada — K8s Secret
MAX_BATCH = int(os.getenv("MAX_BATCH", "32"))
S = {"model": None, "meta": None, "startup": False, "ready": False, "n": 0, "xato": 0, "lat": [], "boshlangan": time.time()}


@asynccontextmanager
async def lifespan(app):
    yo = f"{REESTR}/{MODEL_VERSION}"
    S["meta"] = json.load(open(f"{yo}/meta.json"))
    blob = open(f"{yo}/model.pkl", "rb").read()
    assert hashlib.sha256(blob).hexdigest()[:16] == S["meta"]["sha256"], "model fayli buzilgan!"   # butunlik tekshiruvi
    S["model"] = pickle.loads(blob)
    S["model"].predict_proba(np.zeros((1, 30)))           # isitish (warm-up)
    S["startup"] = S["ready"] = True
    yield
    S["ready"] = False                                    # SIGTERM: trafikni qabul qilishni to'xtatish (drain)


app = FastAPI(title="tumor-api", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def sorov_id(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    t0 = time.perf_counter(); resp = await call_next(request)
    resp.headers["x-request-id"] = rid
    resp.headers["x-model-version"] = MODEL_VERSION
    if request.url.path.startswith("/v1/"):
        S["lat"].append(time.perf_counter() - t0)
    return resp


class Namuna(BaseModel):
    belgilar: List[float] = Field(..., description="30 ta son belgisi")

    @field_validator("belgilar")
    @classmethod
    def uzunlik(cls, v):
        if len(v) != 30: raise ValueError("aynan 30 ta belgi kerak")
        if not all(np.isfinite(v)): raise ValueError("NaN/inf mumkin emas")
        return v


class Paket(BaseModel):
    namunalar: List[Namuna]


def kalit(x_api_key):
    if x_api_key != API_KEY:
        S["xato"] += 1; raise HTTPException(401, "API kalit noto'g'ri")


@app.get("/startupz")
def startupz():
    return {"ok": S["startup"]}


@app.get("/livez")
def livez():
    return {"ok": True, "uptime_s": round(time.time() - S["boshlangan"], 1)}


@app.get("/readyz")
def readyz():
    if not S["ready"]: raise HTTPException(503, "tayyor emas")
    return {"ok": True, "model": S["meta"]["versiya"]}


@app.get("/version")
def version():
    return {"xizmat": app.version, **S["meta"]}


@app.post("/v1/predict")
def predict(n: Namuna, x_api_key: str = Header(None)):
    kalit(x_api_key)
    p = S["model"].predict_proba(np.array([n.belgilar]))[0]
    S["n"] += 1
    return {"sinf": "xavfsiz" if p[1] >= 0.5 else "xavfli", "ehtimol": round(float(max(p)), 4)}


@app.post("/v1/predict/batch")
def batch(pk: Paket, x_api_key: str = Header(None)):
    kalit(x_api_key)
    if len(pk.namunalar) > MAX_BATCH: raise HTTPException(413, f"paket <= {MAX_BATCH}")
    P = S["model"].predict_proba(np.array([n.belgilar for n in pk.namunalar]))
    S["n"] += len(P)
    return {"natija": [int(p[1] >= 0.5) for p in P]}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    lat = sorted(S["lat"]) or [0.0]
    return (f"# TYPE predictions_total counter\npredictions_total {S['n']}\n"
            f"# TYPE auth_errors_total counter\nauth_errors_total {S['xato']}\n"
            f"# TYPE request_latency_p95_seconds gauge\nrequest_latency_p95_seconds {lat[int(0.95 * (len(lat) - 1))]:.6f}\n"
            f"model_info{{version=\"{S['meta']['versiya']}\",sha=\"{S['meta']['sha256']}\"}} 1\n")


# --- Avtomatik test to'plami (CI'da pytest qiladigan ish) ---
K = {"x-api-key": API_KEY}
natija = []
def test(nom, shart):
    natija.append((nom, "PASS" if shart else "FAIL"))

with TestClient(app) as c:
    test("startupz 200", c.get("/startupz").json()["ok"])
    test("readyz 200", c.get("/readyz").status_code == 200)
    r = c.post("/v1/predict", json={"belgilar": Xte[0].tolist()}, headers=K)
    test("predict 200 + sinf", r.status_code == 200 and r.json()["sinf"] in ("xavfli", "xavfsiz"))
    test("x-request-id sarlavhasi", len(r.headers.get("x-request-id", "")) == 12)
    test("so'rov ID saqlanadi", c.get("/livez", headers={"x-request-id": "abc123"}).headers["x-request-id"] == "abc123")
    test("API kalitsiz -> 401", c.post("/v1/predict", json={"belgilar": Xte[0].tolist()}).status_code == 401)
    test("29 belgi -> 422", c.post("/v1/predict", json={"belgilar": [0.0] * 29}, headers=K).status_code == 422)
    test("batch 33 -> 413", c.post("/v1/predict/batch", json={"namunalar": [{"belgilar": [0.0] * 30}] * 33}, headers=K).status_code == 413)
    rb = c.post("/v1/predict/batch", json={"namunalar": [{"belgilar": x.tolist()} for x in Xte[:32]]}, headers=K)
    test("batch 32 aniqlik >= 0.9", rb.status_code == 200 and np.mean(np.array(rb.json()["natija"]) == yte[:32]) >= 0.9)
    test("/version sha256", c.get("/version").json()["sha256"] == S["meta"]["sha256"])
    t0 = time.perf_counter()
    for x in Xte[:100]:
        c.post("/v1/predict", json={"belgilar": x.tolist()}, headers=K)
    rps = 100 / (time.perf_counter() - t0)
    m = c.get("/metrics").text
    test("metrics: predictions_total", "predictions_total 133" in m)

print("\n=== tumor-api testlari ===")
for nom, h in natija:
    print(f"{h:4s}  {nom}")
print(f"Jami: {sum(h == 'PASS' for _, h in natija)}/{len(natija)} PASS;  ketma-ket ~{rps:.0f} so'rov/s (TestClient, 1 jarayon)")
print("\n--- /metrics ---\n" + m)
print("OpenAPI yo'llari:", ", ".join(sorted(app.openapi()["paths"])))
