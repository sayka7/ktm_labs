"""MT4 — 7-qadam. Operatsion hisobot: SLO, xato byudjeti, burn-rate ogohlantirishlari va Go/No-Go

30 kunlik daqiqalik so'rov/xato qatori (kunlik to'lqin, 12 % o'sish, ikki insident: keskin va sekin) sintetik yaratiladi. SLO 99,9 % uchun xato byudjeti va uning sarfi, Google SRE Workbook'dagi ko'p oynali burn-rate qoidalari (1 soat/5 daq > 14,4x, 6 soat/30 daq > 6x, 3 kun/6 soat > 1x) hisoblanib, har insidentni aniqlash kechikishi topiladi. Shu qoidalar asosida PrometheusRule (monitoring.coreos.com/v1) YAML generatsiya qilinadi va validatsiya qilinadi. Oxirida kunlik cho'qqi trendi bo'yicha sig'im prognozi va oldingi qadamlar natijalarini jamlovchi Go/No-Go nazorat ro'yxati.

Kutilgan natija: Mavjudlik ~99,92 %, byudjet sarfi ~84 %; keskin insident 5 daqiqada, sekin insident faqat 6 soatlik va 3 kunlik qoidalarda aniqlanadi; PrometheusRule VALID (5 qoida); sig'im ~45 kunga yetadi; QAROR: GO.
"""
# 7-qadam. Operatsion hisobot: SLO va xato byudjeti, ko'p oynali burn-rate ogohlantirishlari, PrometheusRule YAML
# generatsiyasi va validatsiyasi, sig'im prognozi va yakuniy Go/No-Go nazorat ro'yxati.
# 30 kunlik daqiqalik so'rov/xato qatori sintetik tarzda yaratiladi (real klasterda — Prometheus'dan olinadi).
import math
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

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
BYUDJET = (1 - SLO) * sorov.sum()
print("=== SLO va xato byudjeti (30 kun) ===")
print(f"So'rovlar: {sorov.sum() / 1e6:.2f} mln, xatolar: {xato.sum():.0f}, mavjudlik: {100 * (1 - xato.sum() / sorov.sum()):.4f} %")
print(f"SLO {100 * SLO:.1f} % -> byudjet {BYUDJET:.0f} xato; sarflangan {100 * xato.sum() / BYUDJET:.1f} %")
for nom, bosh, dav, e in INSIDENT:
    print(f"  insident '{nom}': {xato[bosh:bosh + dav].sum():.0f} xato = byudjetning {100 * xato[bosh:bosh + dav].sum() / BYUDJET:.1f} %")


def burn(oyna):
    """Oyna (daqiqa) bo'yicha burn rate = xato ulushi / (1 - SLO)."""
    cs, cx = np.cumsum(np.r_[0, sorov]), np.cumsum(np.r_[0, xato])
    i = np.arange(1, DAQ + 1); j = np.maximum(0, i - oyna)
    return ((cx[i] - cx[j]) / np.maximum(cs[i] - cs[j], 1)) / (1 - SLO)


# Google SRE Workbook tavsiyasi: (uzun oyna, qisqa oyna, chegara, darajasi)
QOIDA = [("XatoByudjetTez", 60, 5, 14.4, "page"), ("XatoByudjetOrta", 360, 30, 6.0, "page"), ("XatoByudjetSekin", 4320, 360, 1.0, "ticket")]
print("\n=== Ko'p oynali burn-rate ogohlantirishlari ===")
yongan = {}
for nom, uzun, qisqa, ch, dar in QOIDA:
    f = (burn(uzun) > ch) & (burn(qisqa) > ch)
    yongan[nom] = f
    print(f"{nom:17s} ({uzun // 60:2d} soat / {qisqa:3d} daq > {ch:4.1f}x, {dar:6s}): {int(f.sum()):5d} daqiqa yongan")
print(f"\n{'insident':32s} " + " ".join(f"{n:>17s}" for n, *_ in QOIDA))
for nom, bosh, dav, e in INSIDENT:
    kech = []
    for q, *_ in QOIDA:
        idx = np.flatnonzero(yongan[q][bosh:bosh + dav + 360])
        kech.append(f"{idx[0]:>13d} daq" if len(idx) else f"{'aniqlanmadi':>17s}")
    print(f"{nom:32s} " + " ".join(kech))
print("Tez qoida keskin nosozlikni daqiqalarda ushlaydi; sekin sizishni faqat 6 soat / 3 kun oynali qoidalar ko'radi.")

# --- PrometheusRule YAML generatsiyasi va validatsiyasi ---
def ifoda(oyna):
    o = f"{oyna // 60}h" if oyna >= 60 else f"{oyna}m"
    return (f'(sum(rate(http_requests_total{{job="tumor-api",code=~"5.."}}[{o}])) / '
            f'sum(rate(http_requests_total{{job="tumor-api"}}[{o}])))')


rule = {"apiVersion": "monitoring.coreos.com/v1", "kind": "PrometheusRule",
        "metadata": {"name": "tumor-api-slo", "namespace": "ai-prod", "labels": {"release": "kube-prometheus-stack"}},
        "spec": {"groups": [{"name": "tumor-api.slo", "rules": [
            {"alert": nom, "expr": f"{ifoda(uzun)} > ({ch} * {1 - SLO:.3f}) and {ifoda(qisqa)} > ({ch} * {1 - SLO:.3f})",
             "for": "2m", "labels": {"severity": dar, "service": "tumor-api"},
             "annotations": {"summary": f"Xato byudjeti {ch}x tezlikda yonmoqda", "runbook_url": "https://wiki.local/runbooks/tumor-api"}}
            for nom, uzun, qisqa, ch, dar in QOIDA] + [
            {"alert": "PodQaytaIshgaTushmoqda", "expr": 'increase(kube_pod_container_status_restarts_total{namespace="ai-prod"}[15m]) > 3',
             "for": "5m", "labels": {"severity": "ticket"}, "annotations": {"summary": "CrashLoopBackOff ehtimoli"}},
            {"alert": "HPAMaksimumda", "expr": 'kube_horizontalpodautoscaler_status_current_replicas == kube_horizontalpodautoscaler_spec_max_replicas',
             "for": "15m", "labels": {"severity": "ticket"}, "annotations": {"summary": "HPA maxReplicas ga yetdi — sig'im rejasini ko'ring"}}]}]}}
matn = yaml.safe_dump(rule, sort_keys=False, allow_unicode=True, width=200)


def rule_tekshir(r):
    x = []
    if r.get("kind") != "PrometheusRule": x.append("kind noto'g'ri")
    nomlar = []
    for g in r["spec"]["groups"]:
        for a in g["rules"]:
            nomlar.append(a.get("alert"))
            for k in ("alert", "expr", "for", "labels", "annotations"):
                if k not in a: x.append(f"{a.get('alert')}: '{k}' yo'q")
            if a.get("labels", {}).get("severity") not in ("page", "ticket"): x.append(f"{a['alert']}: severity noto'g'ri")
            if a["expr"].count("(") != a["expr"].count(")"): x.append(f"{a['alert']}: qavslar mos emas")
            if not str(a.get("for", "")).rstrip("smh").isdigit(): x.append(f"{a['alert']}: 'for' davomiyligi xato")
            if a.get("labels", {}).get("severity") == "page" and "runbook_url" not in a.get("annotations", {}):
                x.append(f"{a['alert']}: page ogohlantirishida runbook yo'q")
    if len(nomlar) != len(set(nomlar)): x.append("takroriy alert nomi")
    return x, len(nomlar)


xat, n = rule_tekshir(yaml.safe_load(matn))
print(f"\n=== PrometheusRule: {n} qoida, {len(matn.splitlines())} qator YAML -> {'VALID' if not xat else xat} ===")
print("\n".join(matn.splitlines()[:16]) + "\n  ...")

# --- Sig'im prognozi: kunlik cho'qqining chiziqli trendi ---
cho = sorov.reshape(30, 1440).max(1) / 60
a, b = np.polyfit(np.arange(30), cho, 1)
SIGIM = 12 * 50 * 0.7                                 # 12 Pod x 50 so'rov/s x 70 % nishon
kun_qoldi = (SIGIM - b) / a - 29 if a > 0 else math.inf
print(f"\n=== Sig'im prognozi ===\nKunlik cho'qqi: {cho[0]:.0f} -> {cho[-1]:.0f} so'rov/s (trend +{a:.2f}/kun); "
      f"xavfsiz sig'im {SIGIM:.0f} so'rov/s -> ~{kun_qoldi:.0f} kundan keyin yetadi")

# --- Yakuniy Go/No-Go ro'yxati: dastlabki 7 band — oldingi qadamlar natijasi (qo'lda kiritiladi), qolganlari shu qadamda hisoblanadi ---
ROYXAT = [
    ("Xizmat testlari 11/11 PASS (1-qadam)", True),
    ("Konfiguratsiya validatsiyasi fail-fast (2-qadam)", True),
    ("startupProbe bilan CrashLoop yo'q (2-qadam)", True),
    ("Dockerfile 13/13, compose OK (3-qadam)", True),
    ("K8s manifest 18/18 prod (4-qadam)", True),
    ("Rollout: maxUnavailable=0, canary avtomatik orqaga qaytaradi (5-qadam)", True),
    ("HPA/CA: sakrashda SLO buzilishi <= 5 daq (6-qadam, bulut CA)", True),
    (f"30 kunlik mavjudlik >= {100 * SLO:.1f} %", 1 - xato.sum() / sorov.sum() >= SLO),
    ("Xato byudjeti sarfi < 100 %", xato.sum() < BYUDJET),
    ("Burn-rate qoidalari VALID va ikkala insident aniqlandi", not xat and all(yongan["XatoByudjetSekin"][b_:b_ + d_ + 360].any() for _, b_, d_, _ in INSIDENT)),
    ("Sig'im kamida 30 kunga yetadi", kun_qoldi >= 30),
]
print("\n=== Go/No-Go nazorat ro'yxati ===")
for nom, ok in ROYXAT:
    print(f"[{'+' if ok else '-'}] {nom}")
print("QAROR:", "GO — ishlab chiqarishga chiqarish mumkin" if all(o for _, o in ROYXAT) else
      "NO-GO — belgilangan bandlar tuzatilmaguncha chiqarilmaydi: " + "; ".join(n for n, o in ROYXAT if not o))

fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
ax[0].plot(t / 1440, sorov / 60, lw=0.4); ax[0].set_ylabel("so'rov/s")
ax[1].plot(t / 1440, 100 - 100 * np.cumsum(xato) / BYUDJET, "r-"); ax[1].axhline(0, color="k", lw=0.8)
ax[1].set_ylabel("qolgan byudjet, %"); ax[1].set_xlabel("kun"); plt.tight_layout()
print("\n=== Operatsion hisobot tayyor ===")
