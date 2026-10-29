"""MT4 — 7-qadam. Operatsion hisobot: SLO, xato byudjeti, burn-rate ogohlantirishlari va Go/No-Go

30 kunlik daqiqalik so'rov/xato qatori (kunlik to'lqin, 12 % o'sish, ikki insident: keskin va sekin) sintetik yaratiladi. SLO 99,9 % uchun xato byudjeti va uning sarfi, Google SRE Workbook'dagi ko'p oynali burn-rate qoidalari (1 soat/5 daq > 14,4x, 6 soat/30 daq > 6x, 3 kun/6 soat > 1x) hisoblanib, har insidentni aniqlash kechikishi topiladi. Shu qoidalar asosida PrometheusRule (monitoring.coreos.com/v1) YAML generatsiya qilinadi va validatsiya qilinadi. Oxirida kunlik cho'qqi trendi bo'yicha sig'im prognozi va oldingi qadamlar natijalarini jamlovchi Go/No-Go nazorat ro'yxati.

Kutilgan natija: Mavjudlik ~99,92 %, byudjet sarfi ~84 %; keskin insident 5 daqiqada, sekin insident faqat 6 soatlik va 3 kunlik qoidalarda aniqlanadi; PrometheusRule VALID (5 qoida); sig'im ~45 kunga yetadi; QAROR: GO.
"""
# 7-qadam. Operatsion hisobot: SLO va xato byudjeti, ko'p oynali burn-rate ogohlantirishlari, PrometheusRule YAML
# generatsiyasi va validatsiyasi, sig'im prognozi va yakuniy Go/No-Go nazorat ro'yxati.
import math
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

# --- Berilgan ma'lumotlar: 30 kunlik daqiqalik so'rov/xato qatori (o'zgartirmang) ---
rng = np.random.default_rng(7)
DAQ = 30 * 24 * 60
t = np.arange(DAQ)
kun = (t % 1440) / 1440
sorov = (120 + 200 * np.sin(np.pi * kun) ** 2) * 60 * (1 + 0.004 * t / 1440)      # so'rov/daqiqa, 30 kunda +12 % o'sish
sorov = rng.poisson(sorov).astype(float)
xato_ulush = np.full(DAQ, 0.0004)
INSIDENT = [("model fayli buzilgan (keskin)", 9 * 1440 + 600, 40, 0.15),
            ("xotira sizishi (sekin)", 21 * 1440 + 300, 720, 0.012)]
for nom, bosh, dav, e in INSIDENT:
    xato_ulush[bosh:bosh + dav] = e
xato = rng.binomial(sorov.astype(int), xato_ulush).astype(float)

SLO = 0.999
print("=== SLO va xato byudjeti (30 kun) ===")
# TODO 1: mavjudlik, byudjet = (1 - SLO) * jami so'rov, sarflangan % va har insidentning byudjetdagi ulushini chiqaring
BYUDJET = None


def burn(oyna):
    """Oyna (daqiqa) bo'yicha har daqiqa uchun burn rate = xato ulushi / (1 - SLO)."""
    # TODO 2: kumulyativ yig'indilar (np.cumsum) bilan sirpanuvchi oynadagi xato ulushini tez hisoblang
    raise NotImplementedError("TODO 2: burn rate")


# Google SRE Workbook tavsiyasi: (nom, uzun oyna daq, qisqa oyna daq, chegara, darajasi)
QOIDA = [("XatoByudjetTez", 60, 5, 14.4, "page"), ("XatoByudjetOrta", 360, 30, 6.0, "page"), ("XatoByudjetSekin", 4320, 360, 1.0, "ticket")]
print("\n=== Ko'p oynali burn-rate ogohlantirishlari ===")
yongan = {}
# TODO 3: har qoida uchun (uzun > ch) & (qisqa > ch) niqobini yongan ga yozing; har insidentning aniqlanish kechikishini (daq) toping


def ifoda(oyna):
    # TODO 4: oyna uchun 5xx xato ulushining PromQL ifodasini qaytaring (rate(http_requests_total[...]))
    raise NotImplementedError("TODO 4: PromQL")


# TODO 5: monitoring.coreos.com/v1 PrometheusRule lug'atini tuzing: 3 ta burn-rate + PodQaytaIshgaTushmoqda + HPAMaksimumda
rule = None
matn = yaml.safe_dump(rule, sort_keys=False, allow_unicode=True, width=200)


def rule_tekshir(r):
    """Qaytaradi: (xatolar, qoidalar soni)."""
    # TODO 6: kind, majburiy maydonlar, severity, qavslar, 'for' formati, page uchun runbook_url, takroriy nomlarni tekshiring
    raise NotImplementedError("TODO 6: PrometheusRule validatsiya")


xat, n = rule_tekshir(yaml.safe_load(matn))
print(f"\n=== PrometheusRule: {n} qoida, {len(matn.splitlines())} qator YAML -> {'VALID' if not xat else xat} ===")

# TODO 7: kunlik cho'qqining chiziqli trendi (np.polyfit) bo'yicha 12 Pod x 50 x 70 % sig'im necha kunga yetishini toping
kun_qoldi = None

ROYXAT = [
    ("Xizmat testlari 11/11 PASS (1-qadam)", True),
    ("Konfiguratsiya validatsiyasi fail-fast (2-qadam)", True),
    ("startupProbe bilan CrashLoop yo'q (2-qadam)", True),
    ("Dockerfile 13/13, compose OK (3-qadam)", True),
    ("K8s manifest 18/18 prod (4-qadam)", True),
    ("Rollout: maxUnavailable=0, canary avtomatik orqaga qaytaradi (5-qadam)", True),
    ("HPA/CA: sakrashda SLO buzilishi <= 5 daq (6-qadam, bulut CA)", True),
    # TODO 8: mavjudlik >= SLO, byudjet < 100 %, qoidalar VALID va insidentlar aniqlangan, sig'im >= 30 kun bandlarini qo'shing
]
print("\n=== Go/No-Go nazorat ro'yxati ===")
# TODO 9: bandlarni [+]/[-] bilan chiqaring va yakuniy QAROR (GO / NO-GO) ni chiqaring
print("\n=== Operatsion hisobot tayyor ===")
