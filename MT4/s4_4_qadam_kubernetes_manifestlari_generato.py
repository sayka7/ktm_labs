"""MT4 — 4-qadam. Kubernetes manifestlari: generator, dev/prod overlay va 18 qoidali tekshiruv

Python generatori bitta parametrlar lug'atidan 9 ta obyekt yaratadi: Namespace, ConfigMap, Secret, Deployment (startup/readiness/liveness probe, preStop, securityContext, topologySpread, checksum/config annotatsiyasi, NAS'dagi PVC), Service, HPA (autoscaling/v2), PodDisruptionBudget, Ingress (TLS) va NetworkPolicy. Kustomize'ga o'xshash chuqur birlashtirish bilan dev va prod overlay'lari olinadi, natija multi-hujjat YAML'ga yozilib qayta o'qiladi. Tekshiruvchi obyekt qoidalari bilan birga obyektlararo moslikni ham ko'radi: Service selector, targetPort, Ingress backend, HPA target, PDB, configMapKeyRef kalitlari. Ataylab buzilgan manifestda tipik xatolar ushlanadi.

Kutilgan natija: dev 8 obyekt (1 nusxada PDB yo'q), prod 9 obyekt; dev va prod 18/18 PASS, buzilgan 11/18; Ingress YAML ko'rsatiladi.
"""
# 4-qadam. Kubernetes manifestlari: Python generatori (dev/prod overlay), pyyaml bilan multi-hujjat YAML,
# obyekt qoidalari va obyektlararo moslik tekshiruvi (Service selector, Ingress backend, HPA target, ConfigMap kalitlari)
import copy, hashlib, json
import yaml

ASOS = {"nom": "tumor-api", "ns": "ai-prod", "obraz": "registry.local/tumor-api:1.0.0", "port": 8000,
        "replicas": 3, "cpu_req": "500m", "cpu_lim": "1", "mem_req": "512Mi", "mem_lim": "1Gi",
        "hpa": (3, 10, 65), "host": "tumor.ai.local", "config": {"MODEL_VERSION": "v1", "MAX_BATCH": "32", "LOG_LEVEL": "INFO"}}
OVERLAY = {"dev": {"ns": "ai-dev", "replicas": 1, "hpa": (1, 2, 80), "host": "tumor-dev.ai.local", "config": {"LOG_LEVEL": "DEBUG"}},
           "prod": {}}


def birlashtir(a, b):
    """Kustomize'ga o'xshash chuqur birlashtirish (deep merge)."""
    # TODO 1: a ning nusxasiga b ni rekursiv qo'shing (ikkalasi dict bo'lsa — ichkariga, aks holda almashtirish)
    raise NotImplementedError("TODO 1: deep merge")


def generatsiya(p):
    """Qaytaradi: Namespace, ConfigMap, Secret, Deployment, Service, HPA, PDB, Ingress, NetworkPolicy lug'atlari ro'yxati."""
    # TODO 2: umumiy label'lar va config xeshi (checksum/config annotatsiyasi uchun)
    # TODO 3: Deployment — RollingUpdate, securityContext, topologySpread, envFrom/secretKeyRef/configMapKeyRef,
    #         resources, startup/readiness/liveness probe'lar, preStop, PVC (models-nas) ni yozing
    # TODO 4: Service (targetPort "http"), HPA autoscaling/v2, PDB, Ingress (TLS), NetworkPolicy obyektlarini qo'shing
    # TODO 5: replicas < 2 bo'lsa PDB ni chiqarib tashlang
    raise NotImplementedError("TODO 2-5: manifest generatori")


def mcpu(v):
    return int(v[:-1]) if str(v).endswith("m") else int(float(v) * 1000)


def mib(v):
    return int(v[:-2]) * (1024 if v.endswith("Gi") else 1)


QOIDALAR = ["K01 obraz tegi qotirilgan (latest emas)", "K02 requests/limits bor", "K03 requests <= limits",
            "K04 startup/readiness/liveness probe", "K05 readiness != liveness yo'li", "K06 runAsNonRoot + readOnlyRootFS",
            "K07 sir Secret orqali (ochiq env emas)", "K08 selector == template labels", "K09 Service selector Pod'larga mos",
            "K10 Service targetPort konteynerda bor", "K11 Ingress backend Service mavjud", "K12 Ingress TLS bor",
            "K13 HPA target Deployment mavjud", "K14 HPA min <= replicas <= max", "K15 PDB minAvailable < replicas (drain bloklanmaydi)",
            "K16 configMapKeyRef kalitlari mavjud", "K17 checksum/config annotatsiyasi", "K18 hamma obyekt bitta namespace'da"]


def tekshir(hujjatlar):
    """Qaytaradi: {qoida: bool} — QOIDALAR dagi 18 ta qoida uchun."""
    # TODO 6: (kind, name) -> obyekt lug'atini tuzing, Deployment va konteynerni toping; K01–K08 obyekt qoidalarini yozing
    # TODO 7: K09–K18 obyektlararo moslik: Service/Ingress/HPA/PDB/ConfigMap havolalari haqiqatan mavjudmi
    raise NotImplementedError("TODO 6-7: manifest tekshiruvchisi")


natija = {}
for muhit, ov in OVERLAY.items():
    p = birlashtir(ASOS, ov)
    docs = generatsiya(p)
    # TODO 8: safe_dump_all -> safe_load_all aylanma tekshiruvi; natija[muhit] = tekshir(...); prod uchun Ingress YAML ni chiqaring
    PROD = docs

buz = copy.deepcopy(PROD)
d = next(h for h in buz if h["kind"] == "Deployment"); cc = d["spec"]["template"]["spec"]["containers"][0]
cc["image"] = "registry.local/tumor-api:latest"; cc["resources"]["requests"]["cpu"] = "2"
cc["env"].append({"name": "API_KEY_OCHIQ", "value": "123"}); cc["env"][1]["valueFrom"]["configMapKeyRef"]["key"] = "MODEL_VER"
next(h for h in buz if h["kind"] == "Service")["spec"]["selector"] = {"app": "tumor"}
next(h for h in buz if h["kind"] == "Ingress")["spec"]["rules"][0]["http"]["paths"][0]["backend"]["service"]["port"] = {"number": 8080}
next(h for h in buz if h["kind"] == "PodDisruptionBudget")["spec"]["minAvailable"] = 3
natija["buzilgan"] = tekshir(buz)

print("=== Manifest qoidalari (dev / prod / buzilgan) ===")
for m, q in natija.items():
    print(f"{m:9s}: {sum(map(bool, q.values()))}/{len(q)} PASS")
print("\n=== K8s manifestlari tekshiruvi tugadi ===")
