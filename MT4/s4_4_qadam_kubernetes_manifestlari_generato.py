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
    r = copy.deepcopy(a)
    for k, v in b.items():
        r[k] = birlashtir(r[k], v) if isinstance(v, dict) and isinstance(r.get(k), dict) else v
    return r


def generatsiya(p):
    L = {"app": p["nom"], "app.kubernetes.io/version": p["obraz"].split(":")[-1]}
    cfg_hash = hashlib.sha256(json.dumps(p["config"], sort_keys=True).encode()).hexdigest()[:10]
    meta = lambda nom, **kw: {"name": nom, "namespace": p["ns"], "labels": dict(L), **kw}
    hpa_min, hpa_max, nishon = p["hpa"]
    docs = [
        {"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": p["ns"]}},
        {"apiVersion": "v1", "kind": "ConfigMap", "metadata": meta(p["nom"] + "-config"), "data": p["config"]},
        {"apiVersion": "v1", "kind": "Secret", "metadata": meta(p["nom"] + "-secret"), "type": "Opaque",
         "stringData": {"API_KEY": "<<CI/CD da Vault'dan qo'yiladi>>"}},
        {"apiVersion": "apps/v1", "kind": "Deployment", "metadata": meta(p["nom"]),
         "spec": {"replicas": p["replicas"], "revisionHistoryLimit": 5,
                  "strategy": {"type": "RollingUpdate", "rollingUpdate": {"maxSurge": 1, "maxUnavailable": 0}},
                  "selector": {"matchLabels": {"app": p["nom"]}},
                  "template": {"metadata": {"labels": dict(L), "annotations": {"checksum/config": cfg_hash}},
                               "spec": {"securityContext": {"runAsNonRoot": True, "runAsUser": 10001, "seccompProfile": {"type": "RuntimeDefault"}},
                                        "terminationGracePeriodSeconds": 30,
                                        "topologySpreadConstraints": [{"maxSkew": 1, "topologyKey": "kubernetes.io/hostname",
                                                                       "whenUnsatisfiable": "ScheduleAnyway", "labelSelector": {"matchLabels": {"app": p["nom"]}}}],
                                        "containers": [{
                                            "name": "api", "image": p["obraz"], "imagePullPolicy": "IfNotPresent",
                                            "ports": [{"name": "http", "containerPort": p["port"]}],
                                            "envFrom": [{"configMapRef": {"name": p["nom"] + "-config"}}],
                                            "env": [{"name": "API_KEY", "valueFrom": {"secretKeyRef": {"name": p["nom"] + "-secret", "key": "API_KEY"}}},
                                                    {"name": "MODEL_VERSION", "valueFrom": {"configMapKeyRef": {"name": p["nom"] + "-config", "key": "MODEL_VERSION"}}}],
                                            "resources": {"requests": {"cpu": p["cpu_req"], "memory": p["mem_req"]}, "limits": {"cpu": p["cpu_lim"], "memory": p["mem_lim"]}},
                                            "startupProbe": {"httpGet": {"path": "/startupz", "port": "http"}, "periodSeconds": 2, "failureThreshold": 30},
                                            "readinessProbe": {"httpGet": {"path": "/readyz", "port": "http"}, "periodSeconds": 5},
                                            "livenessProbe": {"httpGet": {"path": "/livez", "port": "http"}, "periodSeconds": 10, "failureThreshold": 3},
                                            "lifecycle": {"preStop": {"exec": {"command": ["sleep", "5"]}}},
                                            "securityContext": {"readOnlyRootFilesystem": True, "allowPrivilegeEscalation": False, "capabilities": {"drop": ["ALL"]}},
                                            "volumeMounts": [{"name": "models", "mountPath": "/models", "readOnly": True}]}],
                                        "volumes": [{"name": "models", "persistentVolumeClaim": {"claimName": "models-nas", "readOnly": True}}]}}}},
        {"apiVersion": "v1", "kind": "Service", "metadata": meta(p["nom"]),
         "spec": {"selector": {"app": p["nom"]}, "ports": [{"name": "http", "port": 80, "targetPort": "http"}]}},
        {"apiVersion": "autoscaling/v2", "kind": "HorizontalPodAutoscaler", "metadata": meta(p["nom"]),
         "spec": {"scaleTargetRef": {"apiVersion": "apps/v1", "kind": "Deployment", "name": p["nom"]},
                  "minReplicas": hpa_min, "maxReplicas": hpa_max,
                  "metrics": [{"type": "Resource", "resource": {"name": "cpu", "target": {"type": "Utilization", "averageUtilization": nishon}}}],
                  "behavior": {"scaleDown": {"stabilizationWindowSeconds": 300}}}},
        {"apiVersion": "policy/v1", "kind": "PodDisruptionBudget", "metadata": meta(p["nom"]),
         "spec": {"minAvailable": max(1, p["replicas"] - 1), "selector": {"matchLabels": {"app": p["nom"]}}}},
        {"apiVersion": "networking.k8s.io/v1", "kind": "Ingress",
         "metadata": meta(p["nom"], annotations={"nginx.ingress.kubernetes.io/proxy-body-size": "1m"}),
         "spec": {"ingressClassName": "nginx", "tls": [{"hosts": [p["host"]], "secretName": p["nom"] + "-tls"}],
                  "rules": [{"host": p["host"], "http": {"paths": [{"path": "/", "pathType": "Prefix",
                             "backend": {"service": {"name": p["nom"], "port": {"name": "http"}}}}]}}]}},
        {"apiVersion": "networking.k8s.io/v1", "kind": "NetworkPolicy", "metadata": meta(p["nom"] + "-faqat-ingress"),
         "spec": {"podSelector": {"matchLabels": {"app": p["nom"]}}, "policyTypes": ["Ingress"],
                  "ingress": [{"from": [{"namespaceSelector": {"matchLabels": {"kubernetes.io/metadata.name": "ingress-nginx"}}}],
                               "ports": [{"port": p["port"]}]}]}},
    ]
    # 1 nusxali muhitda PDB tugun drain'ini butunlay bloklaydi — shuning uchun faqat replicas >= 2 da yaratiladi
    return [h for h in docs if h["kind"] != "PodDisruptionBudget" or p["replicas"] >= 2]


def mcpu(v):
    return int(v[:-1]) if str(v).endswith("m") else int(float(v) * 1000)


def mib(v):
    return int(v[:-2]) * (1024 if v.endswith("Gi") else 1)


def tekshir(hujjatlar):
    q = {}; ob = {(h["kind"], h["metadata"]["name"]): h for h in hujjatlar}
    dep = next(h for h in hujjatlar if h["kind"] == "Deployment"); t = dep["spec"]["template"]; c = t["spec"]["containers"][0]
    q["K01 obraz tegi qotirilgan (latest emas)"] = ":" in c["image"] and not c["image"].endswith(":latest")
    q["K02 requests/limits bor"] = all(k in c.get("resources", {}).get(x, {}) for x in ("requests", "limits") for k in ("cpu", "memory"))
    q["K03 requests <= limits"] = mcpu(c["resources"]["requests"]["cpu"]) <= mcpu(c["resources"]["limits"]["cpu"]) and \
        mib(c["resources"]["requests"]["memory"]) <= mib(c["resources"]["limits"]["memory"])
    q["K04 startup/readiness/liveness probe"] = all(k in c for k in ("startupProbe", "readinessProbe", "livenessProbe"))
    q["K05 readiness != liveness yo'li"] = c["readinessProbe"]["httpGet"]["path"] != c["livenessProbe"]["httpGet"]["path"]
    q["K06 runAsNonRoot + readOnlyRootFS"] = t["spec"].get("securityContext", {}).get("runAsNonRoot") and c.get("securityContext", {}).get("readOnlyRootFilesystem")
    q["K07 sir Secret orqali (ochiq env emas)"] = not any("KEY" in e["name"] and "value" in e for e in c.get("env", []))
    q["K08 selector == template labels"] = all(t["metadata"]["labels"].get(k) == v for k, v in dep["spec"]["selector"]["matchLabels"].items())
    svc = ob.get(("Service", dep["metadata"]["name"]))
    q["K09 Service selector Pod'larga mos"] = bool(svc) and all(t["metadata"]["labels"].get(k) == v for k, v in svc["spec"]["selector"].items())
    portnom = {p.get("name") for p in c["ports"]} | {p["containerPort"] for p in c["ports"]}
    q["K10 Service targetPort konteynerda bor"] = bool(svc) and all(p["targetPort"] in portnom for p in svc["spec"]["ports"])
    ing = next((h for h in hujjatlar if h["kind"] == "Ingress"), None)
    be = [pp["backend"]["service"] for r in ing["spec"]["rules"] for pp in r["http"]["paths"]] if ing else []
    q["K11 Ingress backend Service mavjud"] = bool(be) and all(("Service", b["name"]) in ob and
        any(b["port"].get("name") == sp.get("name") or b["port"].get("number") == sp["port"] for sp in ob[("Service", b["name"])]["spec"]["ports"]) for b in be)
    q["K12 Ingress TLS bor"] = bool(ing and ing["spec"].get("tls"))
    hpa = next((h for h in hujjatlar if h["kind"] == "HorizontalPodAutoscaler"), None)
    q["K13 HPA target Deployment mavjud"] = bool(hpa) and ("Deployment", hpa["spec"]["scaleTargetRef"]["name"]) in ob
    q["K14 HPA min <= replicas <= max"] = bool(hpa) and hpa["spec"]["minReplicas"] <= dep["spec"]["replicas"] <= hpa["spec"]["maxReplicas"]
    pdb = next((h for h in hujjatlar if h["kind"] == "PodDisruptionBudget"), None)
    q["K15 PDB minAvailable < replicas (drain bloklanmaydi)"] = pdb is None or pdb["spec"]["minAvailable"] < dep["spec"]["replicas"]
    cm_kalit = {(h["metadata"]["name"], k) for h in hujjatlar if h["kind"] == "ConfigMap" for k in h.get("data", {})}
    ref = [e["valueFrom"]["configMapKeyRef"] for e in c.get("env", []) if "configMapKeyRef" in e.get("valueFrom", {})]
    q["K16 configMapKeyRef kalitlari mavjud"] = all((r["name"], r["key"]) in cm_kalit for r in ref)
    q["K17 checksum/config annotatsiyasi"] = "checksum/config" in t["metadata"].get("annotations", {})
    q["K18 hamma obyekt bitta namespace'da"] = len({h["metadata"].get("namespace") for h in hujjatlar if h["kind"] != "Namespace"}) == 1
    return q


natija = {}
for muhit, ov in OVERLAY.items():
    p = birlashtir(ASOS, ov)
    docs = generatsiya(p)
    matn = yaml.safe_dump_all(docs, sort_keys=False, allow_unicode=True)
    qayta = list(yaml.safe_load_all(matn))                 # aylanma tekshiruv: YAML -> Python
    natija[muhit] = tekshir(qayta)
    print(f"[{muhit}] {len(qayta)} obyekt: " + ", ".join(h["kind"] for h in qayta) + f"; YAML {len(matn.splitlines())} qator")
    if muhit == "prod":
        PROD = qayta
        print("\n--- Ingress (prod) ---\n" + yaml.safe_dump(next(h for h in qayta if h["kind"] == "Ingress"), sort_keys=False))

# --- Ataylab buzilgan manifest: tipik xatolar ---
buz = copy.deepcopy(PROD)
d = next(h for h in buz if h["kind"] == "Deployment"); cc = d["spec"]["template"]["spec"]["containers"][0]
cc["image"] = "registry.local/tumor-api:latest"; cc["resources"]["requests"]["cpu"] = "2"
cc["env"].append({"name": "API_KEY_OCHIQ", "value": "123"}); cc["env"][1]["valueFrom"]["configMapKeyRef"]["key"] = "MODEL_VER"
next(h for h in buz if h["kind"] == "Service")["spec"]["selector"] = {"app": "tumor"}
next(h for h in buz if h["kind"] == "Ingress")["spec"]["rules"][0]["http"]["paths"][0]["backend"]["service"]["port"] = {"number": 8080}
next(h for h in buz if h["kind"] == "PodDisruptionBudget")["spec"]["minAvailable"] = 3
natija["buzilgan"] = tekshir(buz)

print("=== Manifest qoidalari (dev / prod / buzilgan) ===")
for k in natija["prod"]:
    print(f"{k:52s} " + "  ".join(f"{'PASS' if natija[m][k] else 'FAIL':5s}" for m in natija))
for m, q in natija.items():
    print(f"{m:9s}: {sum(map(bool, q.values()))}/{len(q)} PASS")
print("\n=== K8s manifestlari tekshiruvi tugadi ===")
