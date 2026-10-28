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

# ---------------- (a) RollingUpdate simulyatsiyasi ----------------
def rolling(N=6, surge=1, unav=0, start_s=30, sig=50, talab=260, recreate=False, T=900):
    """1 s qadamli model. Har Pod sig' so'rov/s; yangi Pod start_s dan keyin tayyor. Deployment kontrolleri:
    yangi Pod'lar jami <= N+surge bo'lguncha yaratiladi; eski Pod'lar mavjudlik >= N-unav saqlangan holda o'chiriladi."""
    eski, yangi_tayyor, startda = N, 0, []
    tushdi = 0.0; min_sig = 10 ** 9; maks_pod = N; t = 0
    if recreate:
        eski = 0                                        # Recreate: hammasi birdan o'chadi
    while t < T:
        yangi_tayyor += sum(1 for r in startda if r == t); startda = [r for r in startda if r > t]
        if not recreate:
            while eski > 0 and eski - 1 + yangi_tayyor >= N - unav:
                eski -= 1
        while eski + yangi_tayyor + len(startda) < N + (0 if recreate else surge) and yangi_tayyor + len(startda) < N:
            startda.append(t + start_s)
        maks_pod = max(maks_pod, eski + yangi_tayyor + len(startda))
        sig_t = (eski + yangi_tayyor) * sig; min_sig = min(min_sig, sig_t)
        tushdi += max(0, talab - sig_t)
        if yangi_tayyor == N and eski == 0:
            return t, min_sig, tushdi, maks_pod
        t += 1
    return None, min_sig, tushdi, maks_pod


print("=== (a) RollingUpdate: 6 nusxa x 50 so'rov/s, talab 260 so'rov/s, Pod start 30 s ===")
print(f"{'strategiya':34s} {'davomiylik s':>12s} {'min sig`im':>11s} {'yo`qolgan so`rov':>17s} {'maks Pod':>9s}")
for nom, kw in (("maxSurge=1, maxUnavailable=0", dict(surge=1, unav=0)), ("maxSurge=2, maxUnavailable=0", dict(surge=2, unav=0)),
                ("maxSurge=0, maxUnavailable=1", dict(surge=0, unav=1)), ("maxSurge=25%(2), maxUnav=25%(2)", dict(surge=2, unav=2)),
                ("maxSurge=6 (blue/green kabi)", dict(surge=6, unav=0)), ("Recreate", dict(recreate=True))):
    d, ms, tu, mp = rolling(**kw)
    print(f"{nom:34s} {d:12d} {ms:11d} {tu:17.0f} {mp:9d}")
print("maxUnavailable>0 tezroq va arzonroq (kam qo'shimcha Pod), lekin sig'im talabdan pastga tushsa so'rovlar yo'qoladi.")

# ---------------- (b) Canary: haqiqiy so'rovlar bilan ----------------
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
    rng = random.Random(seed); jurnal = []
    for w in qadamlar:
        st = {"v1": [], "v2": []}; n = 0
        while (len(st["v2"]) < min_namuna or (w < 100 and len(st["v1"]) < min_namuna)) and n < maks_sorov:
            ver = "v2" if rng.random() * 100 < w else "v1"
            t0 = time.perf_counter()
            r = (v2 if ver == "v2" else v1).post("/predict", json={"x": [rng.random() for _ in range(16)]})
            st[ver].append((r.status_code, (time.perf_counter() - t0) * 1e3)); n += 1
        e = {v: (np.mean([s >= 500 for s, _ in st[v]]) if st[v] else 0.0) for v in st}
        p95 = {v: (np.percentile([l for _, l in st[v]], 95) if st[v] else 0.0) for v in st}
        # qaror: canary xato ulushi bazadan 2 p.p. dan ortiq yoki p95 1,5x (va kamida +2 ms) katta bo'lsa — orqaga qaytarish
        baza_e = e["v1"] if st["v1"] else jurnal[-1]["xato_v1_%"] / 100
        baza_p = p95["v1"] if st["v1"] else jurnal[-1]["p95_v1_ms"]
        yomon = e["v2"] > baza_e + 0.02 or p95["v2"] > max(1.5 * baza_p, baza_p + 2.0)
        jurnal.append({"vazn_%": w, "so'rov": n, "v2_namuna": len(st["v2"]), "xato_v1_%": 100 * baza_e, "xato_v2_%": 100 * e["v2"],
                       "p95_v1_ms": baza_p, "p95_v2_ms": p95["v2"], "qaror": "ORQAGA QAYTARISH" if yomon else "davom"})
        if yomon: break
    return pd.DataFrame(jurnal)


v1 = xizmat("v1", 0.005, 0.0)
for nom, v2 in (("v2 sog'lom (xato 0,5 %, +0 ms)", xizmat("v2", 0.005, 0.0)),
                ("v2 nuqsonli (xato 8 %)", xizmat("v2b", 0.08, 0.0)),
                ("v2 sekin (+6 ms hisob)", xizmat("v2c", 0.005, 6.0))):
    j = canary(v1, v2)
    print(f"\n=== (b) Canary: {nom} ===")
    print(j.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("Natija:", "v2 TO'LIQ CHIQARILDI (100 %)" if j.iloc[-1]["qaror"] == "davom" and j.iloc[-1]["vazn_%"] == 100
          else f"v2 {int(j.iloc[-1]['vazn_%'])} % bosqichda to'xtatildi, trafik v1 ga qaytarildi")
print("\nEslatma: 5 % bosqichda 60 ta canary namunasi uchun ~1200 so'rov kerak — kichik trafikda canary sekin qaror qiladi.")
print("\n=== Rolling va canary tugadi ===")
