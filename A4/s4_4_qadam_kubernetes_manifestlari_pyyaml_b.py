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
    # TODO 2: 'm' qo'shimchasini hisobga olib millicore'ga o'tkazing
    raise NotImplementedError("TODO 2: cpu_m")

def tekshir(text):
    # TODO 1: yaml.safe_load_all bilan ob'ektlarni o'qing va {kind: ob'ekt} lug'atini tuzing
    # TODO 3: selector/label va Service selector/targetPort mosligi
    # TODO 4: probe'lar (bor, liveness kechroq, yo'llar har xil)
    # TODO 5: replicas, image tegi, requests/limits, runAsNonRoot, maxUnavailable 0, HPA min/max/behavior/nishon, Prometheus — jami 18 qoida
    raise NotImplementedError("TODO 1-5: manifest qoidalari")

print("=== Kubernetes manifest tekshiruvi ===")
for nomi, text in (("to'g'ri", MANIFEST), ("xatoli", XATOLI)):
    q, obj = tekshir(text)
    print(f"\n[{nomi.upper()}] ob'ektlar: {list(obj)}   ball {sum(q.values())}/{len(q)}")
    for k, v in q.items(): print(f"  {'PASS' if v else 'FAIL'}  {k}")

print("\n=== HPA hisobi ===")
# TODO 6: requests.cpu (millicore) x averageUtilization / 100 = chegara; 2 nusxa 95 % da kerakli nusxalar soni
