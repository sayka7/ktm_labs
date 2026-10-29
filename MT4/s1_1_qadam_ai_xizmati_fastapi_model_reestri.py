"""MT4 — 1-qadam. AI xizmati: FastAPI, model reestri, API kalit, probe'lar va avtomatik testlar

Loyiha — «tumor-api»: sklearn breast_cancer (30 belgi) bo'yicha o'sma turini tasniflovchi xizmat. Ikki model versiyasi (v1 — logistik regressiya, v2 — tasodifiy o'rmon) o'qitilib, SHA-256 xeshi bilan reestrga (/tmp/models) yoziladi. Xizmat `lifespan` ichida modelni butunlik tekshiruvi va isitish bilan yuklaydi; /startupz, /livez, /readyz alohida, /v1/predict va /v1/predict/batch API kalit (x-api-key) talab qiladi, har javobga x-request-id va x-model-version sarlavhalari qo'shiladi, /metrics Prometheus matn formatida. Oxirida CI'dagi kabi 11 ta avtomatik test TestClient bilan bajariladi. MODEL_VERSION, MAX_BATCH ni o'zgartirib ko'ring.

Kutilgan natija: v1 aniqligi ~0,96, v2 ~0,94; 11/11 PASS; ketma-ket ~300-400 so'rov/s; /metrics da predictions_total 133 va model_info.
"""
# 1-qadam. AI xizmati: FastAPI, model reestri (v1/v2), API kalit, so'rov ID, batch, probe'lar va avtomatik testlar
# Loyiha: "tumor-api" — ko'krak bezi o'smasi belgilarini (sklearn breast_cancer, 30 belgi) tasniflovchi xizmat.
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

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
REESTR = "/tmp/models"
for ver, est in (("v1", make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))),
                 ("v2", RandomForestClassifier(n_estimators=60, random_state=0))):
    est.fit(Xtr, ytr); os.makedirs(f"{REESTR}/{ver}", exist_ok=True)
    # TODO 1: modelni pickle qilib model.pkl ga yozing; meta.json ga versiya, aniqlik va sha256[:16] ni saqlang

MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
API_KEY = os.getenv("API_KEY", "maxfiy-demo-kalit")
MAX_BATCH = int(os.getenv("MAX_BATCH", "32"))
S = {"model": None, "meta": None, "startup": False, "ready": False, "n": 0, "xato": 0, "lat": [], "boshlangan": time.time()}


@asynccontextmanager
async def lifespan(app):
    # TODO 2: meta va modelni o'qing, sha256 bilan butunligini tekshiring, isiting (warm-up), startup/ready ni yoqing
    raise NotImplementedError("TODO 2: lifespan")
    yield


app = FastAPI(title="tumor-api", version="1.0.0", lifespan=lifespan)

# TODO 3: middleware — x-request-id (kelganini saqlash yoki 12 belgili yangi) va x-model-version sarlavhalari, /v1/ kechikishi


class Namuna(BaseModel):
    belgilar: List[float] = Field(..., description="30 ta son belgisi")
    # TODO 4: field_validator — aynan 30 ta va faqat chekli (NaN/inf emas) qiymatlar


class Paket(BaseModel):
    namunalar: List[Namuna]


def kalit(x_api_key):
    # TODO 5: kalit noto'g'ri bo'lsa xato hisoblagichini oshirib, 401 qaytaring
    raise NotImplementedError("TODO 5")

# TODO 6: /startupz, /livez (uptime_s), /readyz (tayyor bo'lmasa 503), /version yo'llarini yozing
# TODO 7: /v1/predict (sinf: "xavfli"/"xavfsiz", ehtimol) va /v1/predict/batch (MAX_BATCH dan ko'p -> 413)
# TODO 8: /metrics — predictions_total, auth_errors_total, p95 kechikish va model_info Prometheus matn formatida

# --- Avtomatik test to'plami (o'zgartirmang) ---
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
