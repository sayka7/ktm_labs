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
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

# --- Berilgan ma'lumotlar (o'zgartirmang) ---
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
    """Qaytaradi: (cfg, manba, xatolar)."""
    # TODO 1: standart -> ConfigMap (yaml.safe_load) -> TUMOR_<KALIT> muhit o'zgaruvchilari tartibida birlashtiring, manbani yozing
    # TODO 2: SXEMA bo'yicha har kalitni turga o'tkazing va diapazonini tekshiring; xatolarni ro'yxatga yig'ing
    raise NotImplementedError("TODO 1-2: konfiguratsiya yuklash")


def maskala(cfg):
    # TODO 3: api_key ni '***' + oxirgi 3 belgi ko'rinishida yashiring
    raise NotImplementedError("TODO 3")


print("=== (a) Konfiguratsiya qatlamlari ===")
for nom, env in (("to'g'ri", {"TUMOR_API_KEY": "s3cr3t-kalit-2026-xyz", "TUMOR_MAX_BATCH": "128"}),
                 ("xato", {"TUMOR_API_KEY": "qisqa", "TUMOR_WORKERS": "ikki", "TUMOR_TIMEOUT_S": "120"})):
    cfg, manba, xato = yukla(CONFIGMAP, env)
    # TODO 4: maskalangan sozlama va manbalarni chiqaring; xato bo'lsa "fail fast", bo'lmasa config-hash (sha256, sort_keys)

os.makedirs("/tmp/models/v2", exist_ok=True)
open("/tmp/models/v2/model.pkl", "wb").write(b"demo-model-bytes")
KUTILGAN = hashlib.sha256(b"demo-model-bytes").hexdigest()
app = FastAPI()
HOLAT = {"draining": False}


def tekshiruvlar():
    # TODO 5: model fayli, uning sha256, disk (>50 MB), RSS xotira (<1200 MB), self-test (<50 ms) va drain holatini tekshiring
    raise NotImplementedError("TODO 5: chuqur readiness tekshiruvlari")

# TODO 6: /livez (faqat jarayon tirikmi) va /readyz (hammasi True bo'lsa 200, aks holda 503 + tekshiruvlar) yo'llari

print("\n=== (b) Chuqur readiness ===")
# TODO 7: TestClient bilan normal, buzilgan fayl, livez va drain holatlarini sinang


def kubelet(startup_s, probe, T=600):
    """Konteyner startup_s soniyada tayyor bo'ladi. liveness/startup ketma-ket failureThreshold marta yiqilsa -> restart;
    kutish (backoff) 10, 20, 40 ... 300 s. Qaytaradi: (tayyor_vaqt yoki None, restartlar, voqealar)."""
    # TODO 8: startupProbe (period*threshold) yoki liveness (delay + period*i) ilova tayyor bo'lgunga qadar yiqiladimi — modellang
    raise NotImplementedError("TODO 8: kubelet simulyatsiyasi")


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

print("\n=== (d) To'xtashda yo'qolgan so'rovlar (100 so'rov/s, so'rov 0,4 s gacha) ===")
# TODO 9: preStop/grace juftliklari (0,30), (5,30), (5,3), (10,30) uchun yo'qolgan so'rovlarni baholang (endpoint kechikishi 3 s)
print("\n=== Konfiguratsiya va sog'liq tekshiruvlari tugadi ===")
