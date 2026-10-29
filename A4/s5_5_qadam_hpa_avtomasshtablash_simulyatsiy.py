"""A4 — 5-qadam. HPA avtomasshtablash simulyatsiyasi (navbat modeli)

Minikube o'rniga diskret-vaqtli simulyatsiya: so'rov oqimi λ(t) (40 → 340 → 40 so'rov/s), har nusxa sig'imi μ = 1000/14 ms, CPU foizi = λ/(R·μ), HPA har 15 s da `ceil(R·cpu/nishon)` hisoblaydi, scaleUp/scaleDown stabilizatsiya oynalari va Pod start vaqti (20 s) hisobga olinadi, navbat va p95 kechikish Little qonuni bilan baholanadi. 7 ta konfiguratsiya (nishon 40/60/80 %, minReplicas 4, sekin start 60 s, maxReplicas 3, oynasiz) SLO buzilish vaqti va pod-soniya narxi bo'yicha solishtiriladi.

Kutilgan natija: Asosiy konfiguratsiyada reaksiya ~65 s, SLO ~200 s buziladi; minReplicas 4 SLO buzilishini 4x kamaytiradi, narx +23 %; maxReplicas 3 — eng yomon.
"""
# 5-qadam. HPA avtomasshtablash simulyatsiyasi: so'rov oqimi -> CPU foizi -> nusxalar soni -> navbat va kechikish
import math, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# --- Berilgan parametrlar (K8s manifestidan va 2-qadam o'lchovlaridan). O'zgartirib ko'ring!
XIZMAT_MS = 14.0            # bitta so'rovga xizmat vaqti, ms
MU = 1000.0 / XIZMAT_MS     # bitta nusxa sig'imi, so'rov/s
MIN_R, MAX_R = 2, 6
NISHON = 60                 # averageUtilization, %
UP_OYNA, DOWN_OYNA = 30, 180
HPA_DAVR = 15
START_S = 20
SLO_MS = 200
T = 900

def yuklama(t):
    """So'rov oqimi, so'rov/s: 40 -> (60 s ichida) 340 -> plato -> 40."""
    if t < 120: return 40
    if t < 180: return 40 + (t - 120) * 5
    if t < 480: return 340
    if t < 540: return 340 - (t - 480) * 5
    return 40

def simulyatsiya(min_r=MIN_R, max_r=MAX_R, nishon=NISHON, start_s=START_S, up_oyna=UP_OYNA, down_oyna=DOWN_OYNA):
    tayyor, kutilayotgan = min_r, []            # kutilayotgan: yangi Pod'lar tayyor bo'ladigan vaqtlar
    navbat = 0.0; tavsiyalar = []; log = []
    for t in range(T):
        # TODO 1: vaqti kelgan kutilayotgan Pod'larni tayyor ga qo'shing
        # TODO 2: lam, sig'im, navbat (1 s qadam), cpu foizi va p95 kechikish (Little qonuni + to'yinish qo'shimchasi)
        # TODO 3: har HPA_DAVR da istalgan = ceil(tayyor*cpu/nishon), [min_r, max_r] oralig'iga cheklang;
        #         scaleUp — up_oyna ichidagi tavsiyalarning eng kichigi, scaleDown — down_oyna ichidagi eng kattasi
        raise NotImplementedError("TODO 1-3: simulyatsiya qadami")
    df = pd.DataFrame(log)
    # TODO 4: max_nusxa, max_p95_ms, SLO_buzilgan_s, max_navbat, pod_soniya, reaksiya_s ko'rsatkichlarini qaytaring
    return df, {}

print("=== HPA simulyatsiyasi ===")
asosiy, x = simulyatsiya()
for k, v in x.items(): print(f"  {k}: {v}")

print("\n=== Konfiguratsiyalar taqqosi ===")
KONF = (("asosiy: min2 max6 nishon60 start20", {}), ("nishon 80 % (tejamkor)", dict(nishon=80)),
        ("nishon 40 % (zaxirali)", dict(nishon=40)), ("minReplicas 4 (oldindan zaxira)", dict(min_r=4)),
        ("sekin start 60 s (GPU Pod, TensorRT)", dict(start_s=60)), ("maxReplicas 3 (klaster kichik)", dict(max_r=3)),
        ("scaleUp oynasi 0 s (tez reaksiya)", dict(up_oyna=0)))
# TODO 5: har konfiguratsiya uchun simulyatsiya qilib jadval tuzing va pod_soniya'ni asosiyga bo'lib narx_nisbati ustunini qo'shing
