"""MT4 — 5-qadam. Versiyani chiqarish: RollingUpdate simulyatsiyasi va haqiqiy so'rovlar bilan canary

(a) Deployment kontrolleri 1 s qadamli modelda simulyatsiya qilinadi: 6 nusxa, har biri 50 so'rov/s, talab 260 so'rov/s, yangi Pod 30 s da tayyor. maxSurge/maxUnavailable kombinatsiyalari, blue/green va Recreate uchun yangilanish davomiyligi, minimal sig'im, yo'qolgan so'rovlar va maksimal Pod soni solishtiriladi. (b) Ikki FastAPI versiyasi TestClient orqali ishga tushadi va vaznli marshrutlovchi (Ingress/Service mesh o'rnida) trafikni 5 -> 25 -> 50 -> 100 % bosqichlarda bo'ladi. Har bosqichda kamida 60 canary namunasi yig'ilib, xato ulushi va p95 kechikish bazaviy versiya bilan solishtiriladi; mezon buzilsa avtomatik orqaga qaytariladi. Sog'lom, nuqsonli va sekin v2 sinab ko'riladi.

Kutilgan natija: maxUnavailable=0 da yo'qotish 0, maxUnavailable=1 da ~1800, Recreate da ~7800 so'rov yo'qoladi; canary: sog'lom v2 100 % gacha chiqadi, nuqsonli va sekin v2 5 % bosqichda to'xtatiladi.
"""
# 5-qadam. Yangi versiyani chiqarish: RollingUpdate simulyatsiyasi (maxSurge/maxUnavailable) va
# haqiqiy so'rovlar bilan canary — ikki FastAPI versiyasi, vaznli marshrutlovchi (Ingress o'rnida) va avtomatik tahlil
import warnings; warnings.filterwarnings("ignore")
import math, random, time
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel


def rolling(N=6, surge=1, unav=0, start_s=30, sig=50, talab=260, recreate=False, T=900):
    """1 s qadamli model. Har Pod sig' so'rov/s; yangi Pod start_s dan keyin tayyor.
    Qaytaradi: (davomiylik s, minimal sig'im, yo'qolgan so'rovlar, maksimal Pod soni)."""
    # TODO 1: har soniyada: start tugagan Pod'larni tayyorga o'tkazing; eski Pod'larni mavjudlik >= N-unav saqlanguncha o'chiring
    # TODO 2: jami Pod <= N+surge bo'lguncha yangi Pod yarating (Recreate: avval hammasi o'chadi, surge yo'q)
    # TODO 3: sig'im = tayyor Pod * sig; yo'qolgan += max(0, talab - sig'im); hammasi yangilanganda natijani qaytaring
    raise NotImplementedError("TODO 1-3: RollingUpdate modeli")


print("=== (a) RollingUpdate: 6 nusxa x 50 so'rov/s, talab 260 so'rov/s, Pod start 30 s ===")
for nom, kw in (("maxSurge=1, maxUnavailable=0", dict(surge=1, unav=0)), ("maxSurge=2, maxUnavailable=0", dict(surge=2, unav=0)),
                ("maxSurge=0, maxUnavailable=1", dict(surge=0, unav=1)), ("maxSurge=25%(2), maxUnav=25%(2)", dict(surge=2, unav=2)),
                ("maxSurge=6 (blue/green kabi)", dict(surge=6, unav=0)), ("Recreate", dict(recreate=True))):
    d, ms, tu, mp = rolling(**kw)
    print(f"{nom:34s} {d:12d} {ms:11d} {tu:17.0f} {mp:9d}")


def xizmat(versiya, xato_ulush, qoshimcha_ms):
    app = FastAPI(); rng = random.Random(sum(map(ord, versiya)))   # takrorlanuvchi urug'

    class Kirish(BaseModel):
        x: list[float]

    @app.post("/predict")
    def predict(k: Kirish):
        t0 = time.perf_counter()
        s = float(np.tanh(np.array(k.x) @ np.linspace(-1, 1, len(k.x))))
        while (time.perf_counter() - t0) * 1e3 < qoshimcha_ms: pass     # sun'iy qo'shimcha hisob
        if rng.random() < xato_ulush: raise HTTPException(500, "ichki xato")
        return {"versiya": versiya, "ball": s}
    return TestClient(app)


def canary(v1, v2, qadamlar=(5, 25, 50, 100), min_namuna=60, maks_sorov=1500, seed=1):
    """Qaytaradi: har bosqich bo'yicha jurnal DataFrame (vazn_%, xato_v1_%, xato_v2_%, p95_v1_ms, p95_v2_ms, qaror)."""
    rng = random.Random(seed); jurnal = []
    for w in qadamlar:
        # TODO 4: w % ehtimol bilan v2 ga, qolganini v1 ga yuboring; har versiyadan kamida min_namuna (status, ms) yig'ing
        # TODO 5: xato ulushi (status >= 500) va p95 kechikishni hisoblang (100 % da baza — oldingi bosqichdan)
        # TODO 6: xato > baza + 2 p.p. yoki p95 > max(1.5*baza, baza+2 ms) bo'lsa — "ORQAGA QAYTARISH" va to'xtash
        raise NotImplementedError("TODO 4-6: canary tahlili")
    return pd.DataFrame(jurnal)


v1 = xizmat("v1", 0.005, 0.0)
for nom, v2 in (("v2 sog'lom (xato 0,5 %, +0 ms)", xizmat("v2", 0.005, 0.0)),
                ("v2 nuqsonli (xato 8 %)", xizmat("v2b", 0.08, 0.0)),
                ("v2 sekin (+6 ms hisob)", xizmat("v2c", 0.005, 6.0))):
    j = canary(v1, v2)
    print(f"\n=== (b) Canary: {nom} ===")
    print(j.to_string(index=False))
print("\n=== Rolling va canary tugadi ===")
