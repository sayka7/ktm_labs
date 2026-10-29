"""MT4 — 6-qadam. HPA (autoscaling/v2) + Cluster Autoscaler + Erlang-C: yuklama simulyatsiyasi

HPA'ning haqiqiy algoritmi modellashtiriladi: desired = ceil(current x metrika / nishon), 10 % tolerantlik, bir nechta metrikada maksimum, scaleUp siyosati (15 s da 100 % yoki 4 Pod) va scaleDown uchun 300 s stabilizatsiya oynasi. Pod'lar tugunlarga joylashadi (tugunda 3 Pod), joy yetmasa Pending bo'ladi va Cluster Autoscaler yangi tugun so'raydi (120 s), ortiqcha tugunni 10 daqiqadan keyin olib tashlaydi. Kechikish M/M/c (Erlang C) bilan baholanadi. 6 soatlik kunlik to'lqin va 10 daqiqalik keskin sakrash (+450 so'rov/s) ostida on-prem (4 tugun) va bulut konfiguratsiyalari SLO buzilish daqiqalari, yo'qolgan so'rovlar, Pod-soat va tugun-soat bo'yicha solishtiriladi.

Kutilgan natija: On-prem 4 tugunda sakrashda ~10 daqiqa SLO buzilishi va ~110 ming so'rov yo'qoladi; bulut CA bilan ~1,5-5 daqiqa; Erlang-C: 300 so'rov/s da 7 Pod p95 ~110 ms, 10 Pod ~64 ms.
"""
# 6-qadam. HPA (autoscaling/v2) algoritmi + Cluster Autoscaler + Erlang-C navbat modeli: yuklama simulyatsiyasi
# Pod 1 yadro so'raydi, 1 so'rov = 20 ms CPU -> Pod sig'imi ~50 so'rov/s; tugun -> 3 Pod. Kechikish — M/M/c (Erlang C).
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
    """M/M/c: p95 javob vaqti (ms); rho >= 1 (ortiqcha yuklama) da None."""
    # TODO 1: rho, Erlang C (navbatda kutish ehtimoli) va p95 = kutish + 3*XIZMAT ni hisoblang
    raise NotImplementedError("TODO 1: Erlang-C")


def sim(nom, min_r=2, max_r=30, cpu_nishon=70, rps_nishon=None, tugun_max=4, tugun_boshl=4, tugun_start=120,
        pod_start=30, pod_per_tugun=3):
    """Qaytaradi: (natija lug'ati: konfiguratsiya, SLO_buzilish_daq, yo'qolgan_so'rov, Pod_soat, tugun_soat, maks_Pod; log)."""
    for t in range(0, T, DT):
        lam = yuklama(t)
        # TODO 2: start tugagan Pod va tugunlarni tayyorga o'tkazing; Pending Pod'larni bo'sh joyga joylang
        # TODO 3: HPA: desired = ceil(tayyor * metrika / nishon), 10 % tolerantlik, metrikalar bo'yicha maksimum, [min_r, max_r]
        # TODO 4: scaleUp: 15 s da max(100 %, 4 Pod); scaleDown: 300 s oynadagi eng katta tavsiya, 60 s da 10 %
        # TODO 5: Cluster Autoscaler: Pending bo'lsa tugun so'rash (tugun_start s), 10 daqiqa ortiqcha tugunni olib tashlash
        # TODO 6: SLO (Erlang-C p95 > 200 ms yoki ortiqcha yuklama), yo'qolgan so'rovlar, Pod-soat va tugun-soatni yig'ing
        raise NotImplementedError("TODO 2-6: HPA + CA simulyatsiyasi")


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
print("=== HPA + Cluster Autoscaler simulyatsiyasi (6 soat, 15 s qadam) ===")
print(pd.DataFrame(natija).to_string(index=False))

print("\n=== Erlang-C: 300 so'rov/s da Pod soniga qarab p95 (ms) ===")
# TODO 7: c = 7, 8, 9, 10, 12 uchun foydalanish va p95 ni chiqaring; sig'im va Pod soni grafiklarini chizing
print("\n=== HPA simulyatsiyasi tugadi ===")
