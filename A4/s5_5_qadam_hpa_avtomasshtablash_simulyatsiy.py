"""A4 — 5-qadam. HPA avtomasshtablash simulyatsiyasi (navbat modeli)

Minikube o'rniga diskret-vaqtli simulyatsiya: so'rov oqimi λ(t) (40 → 340 → 40 so'rov/s), har nusxa sig'imi μ = 1000/14 ms, CPU foizi = λ/(R·μ), HPA har 15 s da `ceil(R·cpu/nishon)` hisoblaydi, scaleUp/scaleDown stabilizatsiya oynalari va Pod start vaqti (20 s) hisobga olinadi, navbat va p95 kechikish Little qonuni bilan baholanadi. 7 ta konfiguratsiya (nishon 40/60/80 %, minReplicas 4, sekin start 60 s, maxReplicas 3, oynasiz) SLO buzilish vaqti va pod-soniya narxi bo'yicha solishtiriladi.

Kutilgan natija: Asosiy konfiguratsiyada reaksiya ~65 s, SLO ~200 s buziladi; minReplicas 4 SLO buzilishini 4x kamaytiradi, narx +23 %; maxReplicas 3 — eng yomon.
"""
# 5-qadam. HPA avtomasshtablash simulyatsiyasi: so'rov oqimi -> CPU foizi -> nusxalar soni -> navbat va kechikish
import math, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# --- Parametrlar (K8s manifestidan va 2-qadam o'lchovlaridan). O'zgartirib ko'ring!
XIZMAT_MS = 14.0            # bitta so'rovga xizmat vaqti (inferens + ishlov), ms  -> sig'im 1000/14 ~ 71 so'rov/s/nusxa
MU = 1000.0 / XIZMAT_MS
MIN_R, MAX_R = 2, 6         # HPA minReplicas / maxReplicas
NISHON = 60                 # averageUtilization, %
UP_OYNA, DOWN_OYNA = 30, 180   # stabilizationWindowSeconds
HPA_DAVR = 15               # HPA metrikani necha soniyada bir o'qiydi
START_S = 20                # yangi Pod tayyor bo'lgunicha (tasvir tortish + model yuklash + readiness)
SLO_MS = 200                # p95 kechikish talabi
T = 900                     # simulyatsiya davomiyligi, s

def yuklama(t):
    """So'rov oqimi, so'rov/s: tinch -> keskin o'sish -> plato -> yana pasayish (video kuzatuvda smena boshlanishi kabi)."""
    if t < 120: return 40
    if t < 180: return 40 + (t - 120) * 5          # 60 s ichida 40 -> 340
    if t < 480: return 340
    if t < 540: return 340 - (t - 480) * 5
    return 40

def simulyatsiya(min_r=MIN_R, max_r=MAX_R, nishon=NISHON, start_s=START_S, up_oyna=UP_OYNA, down_oyna=DOWN_OYNA):
    tayyor, kutilayotgan = min_r, []            # kutilayotgan: [tayyor bo'lish vaqti, ...]
    navbat = 0.0; tavsiyalar = []; log = []
    for t in range(T):
        kutilayotgan_yangi = [x for x in kutilayotgan if x > t]
        tayyor += len(kutilayotgan) - len(kutilayotgan_yangi); kutilayotgan = kutilayotgan_yangi
        lam = yuklama(t); sigim = tayyor * MU
        # Navbat dinamikasi (1 s qadam): kelgan - xizmat ko'rsatilgan
        navbat = max(0.0, navbat + lam - sigim)
        cpu = min(lam / sigim, 1.0) * 100 if sigim else 100     # foiz requests'ga nisbatan (requests ~= limits deb olamiz)
        kutish_ms = navbat / sigim * 1000 if sigim else 1e4     # Little: navbatdagi so'rov kutish vaqti = N / throughput
        p95_ms = XIZMAT_MS + kutish_ms + (XIZMAT_MS * 2 if cpu > 90 else 0)   # to'yinganda p95 keskin o'sadi (empirik qo'shimcha)
        # HPA boshqaruvchisi
        if t % HPA_DAVR == 0:
            istalgan = math.ceil(tayyor * cpu / nishon) if tayyor else min_r
            istalgan = min(max(istalgan, min_r), max_r)
            tavsiyalar.append((t, istalgan))
            jami = tayyor + len(kutilayotgan)
            if istalgan > jami:        # scaleUp: oxirgi up_oyna ichidagi tavsiyalarning ENG KICHIGI (keskin sakrashlardan himoya)
                xavfsiz = min(v for tt, v in tavsiyalar if tt >= t - up_oyna)
                if xavfsiz > jami:
                    kutilayotgan += [t + start_s] * (xavfsiz - jami)
            elif istalgan < jami:      # scaleDown: oxirgi down_oyna ichidagi ENG KATTASI (erta qisqartirishdan himoya)
                xavfsiz = max(v for tt, v in tavsiyalar if tt >= t - down_oyna)
                if xavfsiz < jami:
                    tayyor = max(min_r, xavfsiz)
        log.append(dict(t=t, lam=lam, tayyor=tayyor, cpu=cpu, navbat=navbat, p95_ms=p95_ms))
    df = pd.DataFrame(log)
    return df, dict(max_nusxa=int(df.tayyor.max()), max_p95_ms=round(df.p95_ms.max()), SLO_buzilgan_s=int((df.p95_ms > SLO_MS).sum()),
                    max_navbat=int(df.navbat.max()), pod_soniya=int(df.tayyor.sum()),
                    reaksiya_s=int(df[df.tayyor > min_r].t.min() - 120) if (df.tayyor > min_r).any() else None)

print("=== HPA simulyatsiyasi ===")
print(f"Nusxa sig'imi: {MU:.0f} so'rov/s; cho'qqi yuklama 340 so'rov/s -> kamida {math.ceil(340 / MU / (NISHON / 100))} nusxa kerak ({NISHON} % nishonda)")
asosiy, x = simulyatsiya()
print("\nAsosiy konfiguratsiya (min 2, max 6, nishon 60 %, start 20 s):")
for k, v in x.items(): print(f"  {k}: {v}")
print("\nVaqt bo'yicha (har 60 s):")
print(asosiy[asosiy.t % 60 == 0][["t", "lam", "tayyor", "cpu", "navbat", "p95_ms"]].round(0).to_string(index=False))

# Konfiguratsiyalarni solishtirish
print("\n=== Konfiguratsiyalar taqqosi ===")
rows = []
for nomi, kw in (("asosiy: min2 max6 nishon60 start20", {}),
                 ("nishon 80 % (tejamkor)", dict(nishon=80)),
                 ("nishon 40 % (zaxirali)", dict(nishon=40)),
                 ("minReplicas 4 (oldindan zaxira)", dict(min_r=4)),
                 ("sekin start 60 s (GPU Pod, TensorRT)", dict(start_s=60)),
                 ("maxReplicas 3 (klaster kichik)", dict(max_r=3)),
                 ("scaleUp oynasi 0 s (tez reaksiya)", dict(up_oyna=0))):
    _, r = simulyatsiya(**kw); rows.append(dict(konfiguratsiya=nomi, **r))
dk = pd.DataFrame(rows); dk["narx_nisbati"] = (dk.pod_soniya / dk.pod_soniya[0]).round(2)
print(dk.to_string(index=False))
print("\nXulosa: HPA reaktiv — yuklama o'sgach, metrika davri + stabilizatsiya + Pod start vaqti o'tgachgina sig'im paydo bo'ladi;")
print("  shu oraliqda navbat to'planib p95 SLO buziladi. Yechimlar: minReplicas zaxirasi (narx oshadi), pastroq nishon, tez start (kichik tasvir),")
print("  yoki so'rov/s bo'yicha prognozli masshtablash (KEDA). GPU Pod'larda start 30-60 s — muammo yanada keskin.")

fig, ax = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)
ax[0].plot(asosiy.t, asosiy.lam, label="so'rov/s"); ax[0].plot(asosiy.t, asosiy.tayyor * MU, "--", label="sig'im (nusxa x mu)"); ax[0].set_ylabel("so'rov/s"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].step(asosiy.t, asosiy.tayyor, where="post", color="tab:green", label="tayyor nusxalar"); ax[1].set_ylabel("nusxa"); ax[1].set_ylim(0, MAX_R + 1)
ax1b = ax[1].twinx(); ax1b.plot(asosiy.t, asosiy.cpu, color="tab:gray", alpha=.6, label="CPU %"); ax1b.axhline(NISHON, color="tab:gray", ls=":"); ax1b.set_ylabel("CPU %"); ax[1].legend(loc="upper left"); ax[1].grid(alpha=.3)
ax[2].plot(asosiy.t, asosiy.p95_ms, color="tab:red"); ax[2].axhline(SLO_MS, color="k", ls="--", label=f"SLO {SLO_MS} ms"); ax[2].set_yscale("log"); ax[2].set_ylabel("p95, ms"); ax[2].set_xlabel("vaqt, s"); ax[2].legend(); ax[2].grid(alpha=.3)
fig.suptitle("HPA simulyatsiyasi: yuklama, nusxalar, kechikish"); plt.tight_layout()
