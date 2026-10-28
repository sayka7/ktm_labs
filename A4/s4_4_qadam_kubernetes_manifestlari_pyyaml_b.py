"""A4 — 4-qadam. Kubernetes manifestlari: pyyaml bilan validatsiya va 18 qoida

Deployment/ConfigMap/Service/HPA YAML'i `yaml.safe_load_all` bilan o'qilib 18 qoida bo'yicha tekshiriladi: replicas ≥ 2, selector/label mosligi, requests/limits, readiness va liveness farqi va vaqtlari, runAsNonRoot, RollingUpdate maxUnavailable 0, Service port/selector mosligi, HPA min/max/behavior/target, Prometheus annotatsiyalari. Tipik xatolar bilan yozilgan ikkinchi manifest 2/18 oladi. HPA chegarasi millicore'da hisoblanadi.

Kutilgan natija: To'g'ri manifest 18/18 PASS, xatoli 2/18; requests 1500m × 60 % = 900m chegara; 2 nusxa 95 % da → 4 nusxa.
"""
# 4-qadam. Kubernetes manifestlari: Deployment / ConfigMap / Service / HPA — pyyaml bilan tahlil va qoidalar tekshiruvi
import yaml, pandas as pd

MANIFEST = """
apiVersion: apps/v1
kind: Deployment
metadata: { name: digits-infer, labels: { app: digits-infer } }
spec:
  replicas: 2
  strategy: { type: RollingUpdate, rollingUpdate: { maxUnavailable: 0, maxSurge: 1 } }
  selector: { matchLabels: { app: digits-infer } }
  template:
    metadata:
      labels: { app: digits-infer }
      annotations: { prometheus.io/scrape: "true", prometheus.io/port: "8000", prometheus.io/path: "/metrics" }
    spec:
      securityContext: { runAsNonRoot: true, runAsUser: 10001 }
      containers:
        - name: infer
          image: digits-infer:1.0
          imagePullPolicy: IfNotPresent
          ports: [{ containerPort: 8000 }]
          env:
            - { name: MODEL_PATH, valueFrom: { configMapKeyRef: { name: digits-config, key: model_path } } }
            - { name: ORT_THREADS, value: "2" }
          resources:
            requests: { cpu: "1500m", memory: "600Mi" }
            limits:   { cpu: "2",     memory: "1Gi" }
          readinessProbe: { httpGet: { path: /readyz, port: 8000 }, initialDelaySeconds: 5, periodSeconds: 5, failureThreshold: 12 }
          livenessProbe:  { httpGet: { path: /healthz, port: 8000 }, initialDelaySeconds: 30, periodSeconds: 15 }
          volumeMounts: [{ name: models, mountPath: /models, readOnly: true }]
      volumes:
        - name: models
          hostPath: { path: /data/models, type: Directory }
---
apiVersion: v1
kind: ConfigMap
metadata: { name: digits-config }
data: { model_path: /models/digits_int8.onnx }
---
apiVersion: v1
kind: Service
metadata: { name: digits-infer }
spec:
  type: NodePort
  selector: { app: digits-infer }
  ports: [{ port: 80, targetPort: 8000, nodePort: 30080 }]
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata: { name: digits-infer }
spec:
  scaleTargetRef: { apiVersion: apps/v1, kind: Deployment, name: digits-infer }
  minReplicas: 2
  maxReplicas: 6
  metrics:
    - type: Resource
      resource: { name: cpu, target: { type: Utilization, averageUtilization: 60 } }
  behavior:
    scaleUp:   { stabilizationWindowSeconds: 30 }
    scaleDown: { stabilizationWindowSeconds: 180 }
"""

# "Yangi boshlovchi" manifesti — tipik xatolar bilan
XATOLI = """
apiVersion: apps/v1
kind: Deployment
metadata: { name: digits-infer }
spec:
  replicas: 1
  selector: { matchLabels: { app: digits-infer } }
  template:
    metadata: { labels: { app: digits } }
    spec:
      containers:
        - name: infer
          image: digits-infer:latest
          ports: [{ containerPort: 8000 }]
          livenessProbe: { httpGet: { path: /healthz, port: 8000 }, initialDelaySeconds: 2, periodSeconds: 5 }
---
apiVersion: v1
kind: Service
metadata: { name: digits-infer }
spec:
  selector: { app: digits-infer }
  ports: [{ port: 80, targetPort: 9000 }]
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata: { name: digits-infer }
spec:
  scaleTargetRef: { apiVersion: apps/v1, kind: Deployment, name: digits-infer }
  minReplicas: 1
  maxReplicas: 3
  metrics: [{ type: Resource, resource: { name: cpu, target: { type: Utilization, averageUtilization: 60 } } }]
"""

def cpu_m(v):
    """CPU qiymatini millicore'ga: '1500m' -> 1500, '2' -> 2000."""
    v = str(v); return int(v[:-1]) if v.endswith("m") else int(float(v) * 1000)

def tekshir(text):
    obj = {d["kind"]: d for d in yaml.safe_load_all(text) if d}
    q = {}
    dep = obj.get("Deployment", {}); tpl = dep.get("spec", {}).get("template", {}); pod = tpl.get("spec", {})
    k = pod.get("containers", [{}])[0]; res = k.get("resources", {})
    q["Deployment, Service, HPA, ConfigMap mavjud"] = all(x in obj for x in ("Deployment", "Service", "HorizontalPodAutoscaler", "ConfigMap"))
    q["replicas >= 2 (bitta nuqta nosozligi yo'q)"] = dep.get("spec", {}).get("replicas", 0) >= 2
    q["selector == template labels"] = dep.get("spec", {}).get("selector", {}).get("matchLabels") == tpl.get("metadata", {}).get("labels")
    q["image tegi 'latest' emas"] = ":" in k.get("image", "") and not k["image"].endswith(":latest")
    q["resources.requests (CPU va xotira) — HPA uchun shart"] = all(x in res.get("requests", {}) for x in ("cpu", "memory"))
    q["resources.limits (CPU va xotira)"] = all(x in res.get("limits", {}) for x in ("cpu", "memory"))
    q["limits >= requests"] = bool(res.get("limits")) and cpu_m(res["limits"]["cpu"]) >= cpu_m(res["requests"]["cpu"])
    q["readinessProbe bor"] = "readinessProbe" in k
    q["livenessProbe bor va readiness'dan kechroq boshlanadi"] = "livenessProbe" in k and \
        k["livenessProbe"].get("initialDelaySeconds", 0) >= k.get("readinessProbe", {}).get("initialDelaySeconds", 0) + 10
    q["readiness va liveness yo'llari har xil"] = k.get("readinessProbe", {}).get("httpGet", {}).get("path") != k.get("livenessProbe", {}).get("httpGet", {}).get("path")
    q["runAsNonRoot"] = pod.get("securityContext", {}).get("runAsNonRoot") is True
    q["RollingUpdate maxUnavailable: 0 (xizmat to'xtamaydi)"] = dep.get("spec", {}).get("strategy", {}).get("rollingUpdate", {}).get("maxUnavailable") == 0
    svc = obj.get("Service", {}).get("spec", {})
    q["Service targetPort == containerPort"] = svc.get("ports", [{}])[0].get("targetPort") == k.get("ports", [{}])[0].get("containerPort")
    q["Service selector Pod labellariga mos"] = svc.get("selector") == tpl.get("metadata", {}).get("labels")
    hpa = obj.get("HorizontalPodAutoscaler", {}).get("spec", {})
    q["HPA minReplicas >= 2 va max > min"] = hpa.get("minReplicas", 0) >= 2 and hpa.get("maxReplicas", 0) > hpa.get("minReplicas", 0)
    q["HPA behavior (stabilizatsiya oynalari) bor"] = "behavior" in hpa
    q["HPA nishoni Deployment nomiga mos"] = hpa.get("scaleTargetRef", {}).get("name") == dep.get("metadata", {}).get("name")
    q["Prometheus scrape annotatsiyalari"] = tpl.get("metadata", {}).get("annotations", {}).get("prometheus.io/scrape") == "true"
    return q, obj

print("=== Kubernetes manifest tekshiruvi ===")
jami = {}
for nomi, text in (("to'g'ri", MANIFEST), ("xatoli", XATOLI)):
    q, obj = tekshir(text); jami[nomi] = q
    print(f"\n[{nomi.upper()}] ob'ektlar: {list(obj)}   ball {sum(q.values())}/{len(q)}")
    for k, v in q.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")

print("\nQiyosiy jadval:")
jad = pd.DataFrame({"qoida": list(jami["to'g'ri"]), "to'g'ri": list(jami["to'g'ri"].values()), "xatoli": list(jami["xatoli"].values())})
print(jad.replace({True: "+", False: "-"}).to_string(index=False))

# HPA hisobi: foiz requests'ga nisbatan — chegara qancha millicore?
obj = {d["kind"]: d for d in yaml.safe_load_all(MANIFEST)}
req = cpu_m(obj["Deployment"]["spec"]["template"]["spec"]["containers"][0]["resources"]["requests"]["cpu"])
hpa = obj["HorizontalPodAutoscaler"]["spec"]; target = hpa["metrics"][0]["resource"]["target"]["averageUtilization"]
print(f"\n=== HPA hisobi ===\nrequests.cpu = {req}m, nishon {target} % -> masshtablash chegarasi har Pod'da {req * target // 100}m CPU.")
print(f"Formula: kerakli_nusxa = ceil(joriy_nusxa x joriy_foiz / {target}). Masalan 2 nusxa 95 % da -> ceil(2 x 95/60) = {-(-2 * 95 // 60)} nusxa.")
print(f"Klaster sig'imi: maxReplicas {hpa['maxReplicas']} x limits 2 CPU = {hpa['maxReplicas'] * 2} vCPU kerak bo'lishi mumkin — tugunlar yetarlimi, tekshiring.")
print("\nReal labda: kubectl apply -f k8s/ && kubectl rollout status deployment/digits-infer && kubectl get hpa -w")
