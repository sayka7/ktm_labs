"""MT4 — 6-qadam. HPA (autoscaling/v2) + Cluster Autoscaler + Erlang-C: yuklama simulyatsiyasi

HPA'ning haqiqiy algoritmi modellashtiriladi: desired = ceil(current x metrika / nishon), 10 % tolerantlik, bir nechta metrikada maksimum, scaleUp siyosati (15 s da 100 % yoki 4 Pod) va scaleDown uchun 300 s stabilizatsiya oynasi. Pod'lar tugunlarga joylashadi (tugunda 3 Pod), joy yetmasa Pending bo'ladi va Cluster Autoscaler yangi tugun so'raydi (120 s), ortiqcha tugunni 10 daqiqadan keyin olib tashlaydi. Kechikish M/M/c (Erlang C) bilan baholanadi. 6 soatlik kunlik to'lqin va 10 daqiqalik keskin sakrash (+450 so'rov/s) ostida on-prem (4 tugun) va bulut konfiguratsiyalari SLO buzilish daqiqalari, yo'qolgan so'rovlar, Pod-soat va tugun-soat bo'yicha solishtiriladi.

Kutilgan natija: On-prem 4 tugunda sakrashda ~10 daqiqa SLO buzilishi va ~110 ming so'rov yo'qoladi; bulut CA bilan ~1,5-5 daqiqa; Erlang-C: 300 so'rov/s da 7 Pod p95 ~110 ms, 10 Pod ~64 ms.
"""
# 6-qadam. HPA (autoscaling/v2) algoritmi + Cluster Autoscaler + Erlang-C navbat modeli: yuklama simulyatsiyasi
# HPA formulasi: desired = ceil(current * joriy_metrika / nishon), tolerantlik 10 %, bir nechta metrika — maksimum.
# behavior: scaleUp — 15 s da max(100 %, 4 Pod); scaleDown — 300 s stabilizatsiya oynasi, 60 s da 10 %.
# Pod 1 yadro so'raydi, 1 so'rov = 20 ms CPU -> Pod sig'imi ~50 so'rov/s; tugun 4 yadro (3,5 ajratiladi) -> 3 Pod. Kechikish — M/M/c (Erlang C).
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DT = 15                         # HPA sinxronlash davri, s
T = 6 * 3600                    # 6 soat
XIZMAT = 0.020                  # s/so'rov (CPU vaqti)
MU = 1 / XIZMAT                 # 50 so'rov/s/Pod
REQ_CPU = 1.0                   # Pod requests.cpu = limits.cpu = 1 yadro (Guaranteed QoS)
SLO_MS = 200


def yuklama(t):
    """Kunlik to'lqin (80 -> 320 so'rov/s) + t=3 soatda 10 daqiqalik keskin sakrash (aksiya, +450)."""
    s = 80 + 240 * max(0.0, math.sin(math.pi * t / T)) ** 2
    if 3 * 3600 <= t < 3 * 3600 + 600: s += 450
    return s


def erlang_c(lam, c, mu):
    """M/M/c: navbatda kutish ehtimoli C va p95 javob vaqti (ms). Ortiqcha yuklamada None."""
    rho = lam / (c * mu)
    if rho >= 1: return None
    a = lam / mu; s = sum(a ** k / math.factorial(k) for k in range(c)); last = a ** c / math.factorial(c) / (1 - rho)
    C = last / (s + last)
    kut = 0.0 if C <= 0.05 else math.log(C / 0.05) / (c * mu - lam)
    return 1e3 * (kut + XIZMAT * 3.0)                 # xizmat vaqtining p95 ~ 3x o'rtacha (eksponensial)


def sim(nom, min_r=2, max_r=30, cpu_nishon=70, rps_nishon=None, tugun_max=4, tugun_boshl=4, tugun_start=120,
        pod_start=30, pod_per_tugun=3):
    tayyor = min_r; startda = []                       # (tayyor bo'lish vaqti)
    kutmoqda = 0                                       # Pending Pod'lar (tugunda joy yo'q)
    tugun = tugun_boshl; tugun_kelyapti = []
    tavsiya_tarix = []                                 # scaleDown stabilizatsiya oynasi uchun
    oxirgi_down = -999; ortiqcha_s = 0
    log = []; slo_buzildi = 0; tushdi = 0.0; pod_s = 0.0; tugun_s = 0.0
    for t in range(0, T, DT):
        lam = yuklama(t)
        tayyor += sum(1 for r in startda if r <= t); startda = [r for r in startda if r > t]
        tugun += sum(1 for r in tugun_kelyapti if r <= t); tugun_kelyapti = [r for r in tugun_kelyapti if r > t]
        # Pending Pod'larni joy bo'lsa joylash
        joy = tugun * pod_per_tugun - (tayyor + len(startda))
        joyla = min(joy, kutmoqda); kutmoqda -= joyla; startda += [t + pod_start] * joyla
        # HPA hisobi (faqat tayyor Pod'lar metrikasi)
        cpu_util = 100 * min(lam / tayyor * XIZMAT, 1.0) / REQ_CPU          # requests ga nisbatan %
        joriy = tayyor + len(startda) + kutmoqda
        tavs = []
        for metr, nishon in (("cpu", cpu_nishon), ("rps", rps_nishon)):
            if nishon is None: continue
            qiymat = cpu_util if metr == "cpu" else lam / tayyor
            nisbat = qiymat / nishon
            tavs.append(joriy if abs(nisbat - 1) <= 0.1 else math.ceil(tayyor * nisbat))
        desired = min(max(max(tavs), min_r), max_r)
        tavsiya_tarix.append((t, desired)); tavsiya_tarix = [(a, b) for a, b in tavsiya_tarix if t - a <= 300]
        if desired > joriy:
            desired = min(desired, joriy + max(joriy, 4))                     # scaleUp siyosati
            kutmoqda += desired - joriy
        else:
            stabil = max(b for _, b in tavsiya_tarix)                         # oyna ichidagi eng katta tavsiya
            if stabil < joriy and t - oxirgi_down >= 60:
                kamay = min(joriy - stabil, max(1, math.floor(0.1 * joriy)))
                ol = min(kamay, kutmoqda); kutmoqda -= ol; kamay -= ol
                tayyor -= kamay; oxirgi_down = t
        # Cluster Autoscaler: Pending bor bo'lsa yangi tugun so'raydi
        kerak = math.ceil(kutmoqda / pod_per_tugun) - len(tugun_kelyapti)
        if kerak > 0 and tugun + len(tugun_kelyapti) < tugun_max:
            tugun_kelyapti += [t + tugun_start] * min(kerak, tugun_max - tugun - len(tugun_kelyapti))
        # CA scale-down: tugun 10 daqiqa ortiqcha bo'lsa (Pod'lar kamroq tugunga sig'sa) — bittasi olib tashlanadi
        zarur = max(tugun_boshl, math.ceil((tayyor + len(startda) + kutmoqda) / pod_per_tugun))
        ortiqcha_s = ortiqcha_s + DT if tugun > zarur else 0
        if ortiqcha_s >= 600:
            tugun -= 1; ortiqcha_s = 0
        # SLO: kechikish va yo'qolgan so'rovlar
        p95 = erlang_c(lam, tayyor, MU)
        if p95 is None or p95 > SLO_MS:
            slo_buzildi += DT
        tushdi += max(0.0, lam - tayyor * MU * 0.98) * DT
        pod_s += (tayyor + len(startda)) * DT; tugun_s += tugun * DT
        log.append((t / 3600, lam, tayyor, kutmoqda, tugun))
    return {"konfiguratsiya": nom, "SLO_buzilish_daq": slo_buzildi / 60, "yo'qolgan_so'rov": tushdi,
            "Pod_soat": pod_s / 3600, "tugun_soat": tugun_s / 3600, "maks_Pod": max(l[2] for l in log)}, log


KONF = [
    ("asos: CPU 70 %, 4 tugun (on-prem)", {}),
    ("CPU 50 % (zaxira ko'p)", dict(cpu_nishon=50)),
    ("CPU 70 % + RPS 30/Pod", dict(rps_nishon=30)),
    ("bulut: CA 4 -> 10 tugun", dict(tugun_max=10)),
    ("bulut + RPS 30/Pod", dict(tugun_max=10, rps_nishon=30)),
    ("bulut + sekin tugun (300 s)", dict(tugun_max=10, tugun_start=300)),
    ("bulut + oldindan 6 tugun, min 8", dict(tugun_max=10, tugun_boshl=6, min_r=8)),
]
natija = []; loglar = {}
for nom, kw in KONF:
    r, lg = sim(nom, **kw); natija.append(r); loglar[nom] = lg
df = pd.DataFrame(natija)
print("=== HPA + Cluster Autoscaler simulyatsiyasi (6 soat, 15 s qadam) ===")
print("Yuklama: 80 -> 320 so'rov/s kunlik to'lqin + 3-soatda 10 daqiqa +450 so'rov/s; Pod sig'imi 50 so'rov/s; 1 tugun = 3 Pod")
print(df.to_string(index=False, float_format=lambda v: f"{v:.1f}"))

print("Kuzating: CPU metrikasi 100 % da to'yinadi (Pod o'z sig'imidan ko'p ishlay olmaydi), shuning uchun HPA bir qadamda")
print("faqat ~1,4x o'sadi; RPS metrikasi haqiqiy talabni ko'radi, lekin asosiy cheklov — tugun (120 s) va Pod (30 s) start vaqti.")
print("On-prem'da 4 tugun (12 Pod = 600 so'rov/s) sakrashni ko'tarmaydi; oldindan zaxira eng kam buzilish beradi, ammo qimmatroq.")
print("\n=== Erlang-C: 300 so'rov/s da Pod soniga qarab p95 (ms) ===")
for c in (7, 8, 9, 10, 12):
    p = erlang_c(300, c, MU)
    print(f"c={c:2d}: foydalanish {300 / (c * MU) * 100:5.1f} %, p95 = {'ortiqcha yuklama' if p is None else f'{p:6.1f} ms'}")

fig, ax = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
lg = np.array(loglar[KONF[0][0]]); lg2 = np.array(loglar[KONF[4][0]])
ax[0].plot(lg[:, 0], lg[:, 1], "k-"); ax[0].plot(lg[:, 0], lg[:, 2] * MU, "b--", label="sig'im (asos)")
ax[0].plot(lg2[:, 0], lg2[:, 2] * MU, "g:", label="sig'im (bulut CA)"); ax[0].set_ylabel("so'rov/s"); ax[0].legend()
ax[1].step(lg[:, 0], lg[:, 2], label="tayyor Pod (asos)"); ax[1].step(lg2[:, 0], lg2[:, 2], label="tayyor Pod (bulut)")
ax[1].step(lg2[:, 0], lg2[:, 4], label="tugunlar (bulut)"); ax[1].set_xlabel("vaqt, soat"); ax[1].legend(fontsize=8)
plt.tight_layout()
print("\n=== HPA simulyatsiyasi tugadi ===")
