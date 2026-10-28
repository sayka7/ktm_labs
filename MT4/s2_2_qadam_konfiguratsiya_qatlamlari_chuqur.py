"""MT4 — 2-qadam. Konfiguratsiya qatlamlari, chuqur readiness, probe'lar va to'g'ri to'xtash

(a) Sozlamalar uch qatlamdan yig'iladi: standart < ConfigMap (YAML) < muhit o'zgaruvchilari (TUMOR_*); har kalit sxema bo'yicha tur va diapazonga tekshiriladi, sir maskalanadi, xato bo'lsa xizmat ishga tushmaydi (fail fast); konfiguratsiya xeshi Deployment annotatsiyasi uchun hisoblanadi. (b) /readyz model fayli, uning xeshi, disk, xotira, o'z-o'zini sinash va drain holatini tekshiradi; /livez esa bog'liqliklarni tekshirmaydi. (c) Kubelet probe'lari simulyatsiyasi: model 45 s yuklanganda noto'g'ri liveness yoki qisqa startupProbe CrashLoopBackOff'ga olib kelishi ko'rsatiladi. (d) preStop va terminationGracePeriod ning yo'qolgan so'rovlarga ta'siri.

Kutilgan natija: To'g'ri konfiguratsiyada OK va config-hash, xatoda 3 ta xato; buzilgan faylda readyz 503, livez 200; 2 ta XATO konfiguratsiyada 6 restart va hech qachon tayyor emas, to'g'rilarida 45 s; preStop=0 da ~300 so'rov yo'qoladi.
"""
# 2-qadam. Konfiguratsiya qatlamlari va sog'liq tekshiruvlari: sozlamalar validatsiyasi, chuqur readiness,
# kubelet probe'lari simulyatsiyasi (CrashLoopBackOff) va to'g'ri to'xtash (graceful shutdown)
import warnings; warnings.filterwarnings("ignore")
import hashlib, json, os, shutil, time
import psutil, yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ---------------- (a) Konfiguratsiya: standart < ConfigMap (YAML) < muhit o'zgaruvchilari ----------------
STANDART = {"model_version": "v1", "max_batch": 32, "log_level": "INFO", "workers": 2, "timeout_s": 5.0, "api_key": ""}
SXEMA = {  # kalit: (tur, tekshiruv, izoh)
    "model_version": (str, lambda v: v in ("v1", "v2", "v3"), "v1|v2|v3"),
    "max_batch": (int, lambda v: 1 <= v <= 256, "1..256"),
    "log_level": (str, lambda v: v in ("DEBUG", "INFO", "WARNING", "ERROR"), "log darajasi"),
    "workers": (int, lambda v: 1 <= v <= 8, "1..8 (CPU limitiga mos)"),
    "timeout_s": (float, lambda v: 0.1 <= v <= 60, "0,1..60 s"),
    "api_key": (str, lambda v: len(v) >= 16, ">= 16 belgi (Secret)"),
}
CONFIGMAP = """
model_version: v2
max_batch: 64
log_level: INFO
workers: 2
"""


def yukla(configmap_text, env):
    cfg = dict(STANDART); manba = {k: "standart" for k in cfg}
    for k, v in (yaml.safe_load(configmap_text) or {}).items():
        cfg[k] = v; manba[k] = "ConfigMap"
    for k in cfg:
        e = env.get("TUMOR_" + k.upper())
        if e is not None:
            cfg[k] = e; manba[k] = "env"
    xatolar = []
    for k, (tur, shart, izoh) in SXEMA.items():
        try:
            cfg[k] = tur(cfg[k])
            if not shart(cfg[k]): xatolar.append(f"{k}={cfg[k]!r} — ruxsat: {izoh}")
        except (TypeError, ValueError):
            xatolar.append(f"{k}={cfg[k]!r} — tur {tur.__name__} emas")
    return cfg, manba, xatolar


def maskala(cfg):
    return {k: ("***" + v[-3:] if k == "api_key" and v else v) for k, v in cfg.items()}


print("=== (a) Konfiguratsiya qatlamlari ===")
for nom, env in (("to'g'ri", {"TUMOR_API_KEY": "s3cr3t-kalit-2026-xyz", "TUMOR_MAX_BATCH": "128"}),
                 ("xato", {"TUMOR_API_KEY": "qisqa", "TUMOR_WORKERS": "ikki", "TUMOR_TIMEOUT_S": "120"})):
    cfg, manba, xato = yukla(CONFIGMAP, env)
    print(f"\n[{nom}] yakuniy sozlama: {json.dumps(maskala(cfg), ensure_ascii=False)}")
    print("  manbalar: " + ", ".join(f"{k}<-{m}" for k, m in manba.items()))
    if xato:
        print("  XATO — xizmat ishga TUSHMAYDI (fail fast):"); [print("   -", x) for x in xato]
    else:
        h = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]
        print(f"  OK. config-hash = {h}  -> Deployment annotatsiyasi: ConfigMap o'zgarsa Pod'lar qayta yaratiladi")

# ---------------- (b) Chuqur readiness: bog'liqliklar tekshiruvi ----------------
os.makedirs("/tmp/models/v2", exist_ok=True)
open("/tmp/models/v2/model.pkl", "wb").write(b"demo-model-bytes")
KUTILGAN = hashlib.sha256(b"demo-model-bytes").hexdigest()
app = FastAPI()
HOLAT = {"draining": False}


def tekshiruvlar():
    t = {}
    p = "/tmp/models/v2/model.pkl"
    t["model_fayl"] = os.path.exists(p)
    t["model_sha"] = t["model_fayl"] and hashlib.sha256(open(p, "rb").read()).hexdigest() == KUTILGAN
    t["disk_bosh_MB>50"] = shutil.disk_usage("/tmp").free / 2 ** 20 > 50
    t["xotira_RSS_MB<1200"] = psutil.Process().memory_info().rss / 2 ** 20 < 1200
    t0 = time.perf_counter(); sum(i * i for i in range(20000)); t["selftest_ms<50"] = (time.perf_counter() - t0) * 1e3 < 50
    t["draining_emas"] = not HOLAT["draining"]
    return t


@app.get("/livez")
def livez():                      # liveness — faqat jarayon tirikmi (bog'liqliklarni TEKSHIRMAYDI!)
    return {"ok": True}


@app.get("/readyz")
def readyz():
    t = tekshiruvlar(); ok = all(t.values())
    from fastapi.responses import JSONResponse
    return JSONResponse({"ok": ok, "tekshiruvlar": t}, status_code=200 if ok else 503)


print("\n=== (b) Chuqur readiness ===")
with TestClient(app) as c:
    r = c.get("/readyz"); print("normal:      ", r.status_code, r.json()["tekshiruvlar"])
    open("/tmp/models/v2/model.pkl", "wb").write(b"buzilgan")
    r = c.get("/readyz"); print("buzilgan fayl:", r.status_code, {k: v for k, v in r.json()["tekshiruvlar"].items() if not v})
    print("livez buzilganda ham:", c.get("/livez").status_code, "(to'g'ri: qayta ishga tushirish muammoni hal qilmaydi)")
    open("/tmp/models/v2/model.pkl", "wb").write(b"demo-model-bytes")
    HOLAT["draining"] = True
    print("SIGTERM (drain):", c.get("/readyz").status_code)

# ---------------- (c) Kubelet probe simulyatsiyasi ----------------
def kubelet(startup_s, probe, T=600):
    """Konteyner startup_s soniyada tayyor bo'ladi. Probe'lar har 'period' soniyada tekshiriladi.
    liveness/startup ketma-ket failureThreshold marta yiqilsa -> restart; kutish (backoff) 10, 20, 40 ... 300 s."""
    t, restart, backoff, voqealar = 0, 0, 10, []
    while t < T:
        bosh = t
        if "startup" in probe:                          # startupProbe o'tguncha liveness ishlamaydi
            per, thr = probe["startup"]
            yiqildi_t = per * thr if startup_s > per * thr else None
            nom = "startupProbe"
        else:
            per, thr = probe["liv"]; d = probe["liv_delay"]
            erta = [d + per * i for i in range(thr)]    # ilova tayyor bo'lguncha bo'ladigan birinchi thr ta tekshiruv
            yiqildi_t = erta[-1] if erta[-1] < startup_s else None
            nom = f"liveness {thr}x"
        if yiqildi_t is not None:
            t = bosh + yiqildi_t; restart += 1
            voqealar.append(f"t={t:4.0f}s {nom} yiqildi -> restart #{restart} (CrashLoopBackOff, kutish {backoff}s)")
            t += backoff; backoff = min(backoff * 2, 300); continue
        rper = probe["rdy"]
        tayyor = bosh + rper * -(-startup_s // rper)     # birinchi muvaffaqiyatli readiness tekshiruvi
        voqealar.append(f"t={tayyor:4.0f}s readiness OK -> Service trafikni yubora boshladi")
        return tayyor, restart, voqealar
    return None, restart, voqealar


KONF = {  # liv: (period, failureThreshold), rdy: readiness period
    "XATO: liveness 10s, startupProbe yo'q": {"liv_delay": 10, "liv": (5, 3), "rdy": 5},
    "XATO: startupProbe 2s x 15 (30 s)":    {"startup": (2, 15), "liv": (10, 3), "rdy": 5},
    "TO'G'RI: startupProbe 2s x 30":         {"startup": (2, 30), "liv": (10, 3), "rdy": 5},
    "TO'G'RI: liveness delay 60s":           {"liv_delay": 60, "liv": (10, 3), "rdy": 5},
}
print("\n=== (c) Kubelet probe simulyatsiyasi: model yuklanishi 45 s ===")
for nom, pr in KONF.items():
    tayyor, rs, v = kubelet(45, pr)
    print(f"\n{nom}: tayyor = {tayyor if tayyor is not None else 'HECH QACHON (600 s ichida)'}, restartlar = {rs}")
    for x in v[:5]: print("   " + x)

# ---------------- (d) Graceful shutdown: preStop + terminationGracePeriod ----------------
print("\n=== (d) To'xtashda yo'qolgan so'rovlar (100 so'rov/s, so'rov 0,4 s gacha) ===")
for prestop, grace in ((0, 30), (5, 30), (5, 3), (10, 30)):
    # endpoint yangilanishi (kube-proxy/ingress) ~3 s kechikadi: shu vaqt ichida kelgan so'rovlar preStop bo'lmasa yo'qoladi
    kechikish = 3.0
    yoqol_kirish = 100 * max(0.0, kechikish - prestop)
    yoqol_jarayon = 100 * 0.4 if grace < prestop + 0.4 else 0
    print(f"preStop={prestop:2d}s grace={grace:2d}s -> yo'qolgan ~{yoqol_kirish + yoqol_jarayon:5.0f} so'rov")
print("\n=== Konfiguratsiya va sog'liq tekshiruvlari tugadi ===")
