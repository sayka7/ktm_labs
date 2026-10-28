"""MT4 — 3-qadam. Dockerfile va docker-compose: generatsiya, validatsiya va qatlam keshi

Loyiha parametrlaridan ko'p bosqichli (multi-stage) Dockerfile generatsiya qilinadi: builder bosqichida wheel'lar, yakuniy obrazda faqat runtime, root bo'lmagan foydalanuvchi, HEALTHCHECK va exec-shakldagi CMD. 13 qoidali tahlilchi uni ataylab yomon yozilgan Dockerfile bilan solishtiradi. Qatlam keshi xeshlar zanjiri bilan simulyatsiya qilinadi: faqat kod o'zgarganda nechta qatlam qayta qurilishi va yomon tartibda pip ham qayta ishlashi ko'rsatiladi. So'ng API + Redis + Prometheus uchun docker-compose.yml pyyaml bilan yaratiladi va tekshiriladi: obraz teglari, port to'qnashuvi, depends_on + healthcheck, hajm va tarmoqlar e'loni, sirlar, ishga tushish tartibi (topologik saralash) va sikl.

Kutilgan natija: Yaxshi Dockerfile 13/13, yomoni 0/13; kod o'zgarganda 1 qatlam (~3 s), requirements o'zgarganda 5 qatlam (~200 s, taxminiy); compose OK, tartib redis -> api -> prometheus; buzilgan compose'da 4 xato.
"""
# 3-qadam. Dockerfile va docker-compose: shablondan generatsiya, qoidalar bo'yicha validatsiya va qatlam keshi simulyatsiyasi
# Docker sandbox'da yo'q — fayllar matn sifatida yaratiladi va Python'da tahlil qilinadi (hadolint/compose config o'rniga).
import hashlib, re
import yaml

LOYIHA = {"nom": "tumor-api", "python": "3.12-slim-bookworm", "port": 8000, "uid": 10001,
          "paketlar": ["fastapi==0.115.6", "uvicorn==0.34.0", "scikit-learn==1.6.1", "numpy==2.2.1"]}


def dockerfile(p):
    """Ko'p bosqichli (multi-stage) Dockerfile: builder'da g'ildiraklar (wheels), yakuniy obrazda faqat runtime."""
    return f"""# syntax=docker/dockerfile:1
FROM python:{p['python']} AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

FROM python:{p['python']}
LABEL org.opencontainers.image.title="{p['nom']}" org.opencontainers.image.version="1.0.0"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MODEL_DIR=/models
RUN groupadd -g {p['uid']} app && useradd -u {p['uid']} -g app -s /usr/sbin/nologin app
WORKDIR /app
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels
COPY --chown=app:app app/ ./app/
USER app
EXPOSE {p['port']}
HEALTHCHECK --interval=15s --timeout=3s --start-period=40s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:{p['port']}/readyz').status==200 else 1)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "{p['port']}", "--workers", "2"]
"""


YOMON = """FROM python:latest
ENV API_KEY=maxfiy123
COPY . .
RUN pip install fastapi uvicorn scikit-learn
RUN apt-get update
RUN apt-get install -y curl
EXPOSE 8080
CMD uvicorn app.main:app --port 8000
"""


def tahlil(txt, port):
    qatorlar = [l.strip() for l in txt.splitlines() if l.strip() and not l.strip().startswith("#")]
    ins = [l.split()[0].upper() for l in qatorlar]
    froms = [l for l in qatorlar if l.upper().startswith("FROM")]
    q = {}
    q["D01 ko'p bosqichli (>=2 FROM)"] = len(froms) >= 2
    q["D02 bazaviy obraz teg bilan, ':latest' emas"] = all(":" in f.split()[1] and not f.split()[1].endswith(":latest") for f in froms)
    q["D03 slim/distroless baza"] = any(k in froms[-1] for k in ("slim", "distroless", "alpine"))
    q["D04 USER (root emas)"] = "USER" in ins and not any(re.match(r"USER\s+(root|0)\b", l) for l in qatorlar)
    q["D05 ENV da sir yo'q"] = not any(re.search(r"ENV.*(KEY|SECRET|PASSWORD|TOKEN)=", l, re.I) for l in qatorlar)
    q["D06 'COPY . .' yo'q"] = not any(re.match(r"COPY\s+\.\s+\.", l) for l in qatorlar)
    q["D07 requirements koddan oldin (kesh)"] = any("requirements" in l for l in qatorlar) and \
        (min([i for i, l in enumerate(qatorlar) if "requirements" in l] or [99]) < max(i for i, l in enumerate(qatorlar) if l.startswith("COPY")))
    q["D08 pip --no-cache-dir"] = all("--no-cache-dir" in l for l in qatorlar if "pip install" in l)
    q["D09 versiyalar qotirilgan (==)"] = "requirements" in txt or all("==" in l for l in qatorlar if "pip install" in l)
    q["D10 HEALTHCHECK bor"] = "HEALTHCHECK" in ins
    q["D11 EXPOSE = ilova porti"] = any(l == f"EXPOSE {port}" for l in qatorlar)
    q["D12 CMD exec shaklida (JSON)"] = any(l.startswith("CMD [") for l in qatorlar)
    q["D13 apt-get update alohida RUN emas"] = not any(l.strip() == "RUN apt-get update" for l in qatorlar)
    return q


YAXSHI = dockerfile(LOYIHA)
print("=== Generatsiya qilingan Dockerfile ===\n" + YAXSHI)
print("=== Dockerfile qoidalari: yaxshi vs yomon ===")
qy, qn = tahlil(YAXSHI, 8000), tahlil(YOMON, 8000)
for k in qy:
    print(f"{k:42s} {'PASS' if qy[k] else 'FAIL':5s} | {'PASS' if qn[k] else 'FAIL'}")
print(f"Ball: yaxshi {sum(qy.values())}/{len(qy)}, yomon {sum(qn.values())}/{len(qn)}")

# --- Qatlam keshi simulyatsiyasi: har ko'rsatma xeshi = oldingi xesh + ko'rsatma + tegishli fayllar xeshi ---
def qurish(txt, fayllar, kesh):
    """Har FROM yangi zanjir boshlaydi; COPY xeshiga fayllar mazmuni, COPY --from ga esa o'sha bosqich xeshi qo'shiladi."""
    h = "0"; bosqich = {}; joriy = None; qayta = []
    for l in [l for l in txt.splitlines() if l.strip() and not l.startswith("#")]:
        soz = l.split()
        if soz[0] == "FROM":
            if joriy: bosqich[joriy] = h
            h = hashlib.sha1(l.encode()).hexdigest(); joriy = soz[-1] if "AS" in soz else "yakuniy"; continue
        manba = ""
        if soz[0] == "COPY":
            argl = [a for a in soz[1:] if not a.startswith("--")]
            fr = next((a.split("=")[1] for a in soz[1:] if a.startswith("--from=")), None)
            if fr:
                manba = bosqich.get(fr, "")
            else:
                src = argl[0].rstrip("/").lstrip("./")
                manba = "".join(v for k, v in sorted(fayllar.items()) if src == "" or k.startswith(src))
        h = hashlib.sha1((h + l + manba).encode()).hexdigest()
        if h not in kesh and soz[0] in ("RUN", "COPY"):
            qayta.append(soz[0] + " " + " ".join(a for a in soz[1:] if not a.startswith("--chown"))[:30])
        kesh.add(h)
    return qayta


VAQT = {"RUN pip": 95, "RUN groupadd": 2}            # taxminiy qatlam qurish vaqti, s
fayllar = {"requirements.txt": "fastapi==0.115.6...", "app/main.py": "v1-kod", "app/model.py": "m"}
kesh = set(); qurish(YAXSHI, fayllar, kesh)
print("\n=== Qatlam keshi (multi-stage, requirements alohida) ===")
for nom, ozgarish in (("faqat kod o'zgardi (app/main.py)", {"app/main.py": "v2-kod"}),
                      ("requirements o'zgardi", {"requirements.txt": "fastapi==0.116.0..."})):
    f2 = {**fayllar, **ozgarish}
    q = qurish(YAXSHI, f2, set(kesh))
    t = sum(next((v for k, v in VAQT.items() if x.startswith(k)), 3) for x in q)
    print(f"{nom:36s}: qayta quriladi {len(q)} qatlam ~{t:4d} s -> {q}")
kesh_y = set(); qurish(YOMON, fayllar, kesh_y)
q = qurish(YOMON, {**fayllar, "app/main.py": "v2-kod"}, set(kesh_y))
print(f"{'YOMON Dockerfile, faqat kod o`zgardi':36s}: qayta quriladi {len(q)} qatlam -> pip ham qayta (COPY . . birinchi)")

# --- docker-compose: API + Redis kesh + Prometheus; generatsiya va validatsiya ---
COMPOSE = {
    "services": {
        "api": {"build": ".", "image": "registry.local/tumor-api:1.0.0", "ports": ["8000:8000"],
                "environment": {"MODEL_VERSION": "v1", "REDIS_URL": "redis://redis:6379/0"},
                "env_file": [".env"], "volumes": ["models:/models:ro"], "read_only": True,
                "depends_on": {"redis": {"condition": "service_healthy"}},
                "healthcheck": {"test": ["CMD", "python", "-c", "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/readyz')"],
                                "interval": "15s", "timeout": "3s", "retries": 3, "start_period": "40s"},
                "deploy": {"resources": {"limits": {"cpus": "2", "memory": "1g"}}}, "restart": "unless-stopped",
                "networks": ["ichki"]},
        "redis": {"image": "redis:7.4-alpine", "healthcheck": {"test": ["CMD", "redis-cli", "ping"], "interval": "5s", "retries": 5},
                  "networks": ["ichki"], "restart": "unless-stopped"},
        "prometheus": {"image": "prom/prometheus:v3.1.0", "ports": ["9090:9090"], "volumes": ["./prometheus.yml:/etc/prometheus/prometheus.yml:ro"],
                       "depends_on": {"api": {"condition": "service_healthy"}}, "networks": ["ichki"]},
    },
    "volumes": {"models": {}},
    "networks": {"ichki": {}},
}
matn = yaml.safe_dump(COMPOSE, sort_keys=False, allow_unicode=True)
print("\n=== docker-compose.yml (generatsiya) — birinchi 12 qator ===\n" + "\n".join(matn.splitlines()[:12]) + "\n  ...")


def compose_tekshir(c):
    x = []; s = c["services"]; portlar = {}
    for nom, v in s.items():
        if ":" not in v.get("image", ":") or v.get("image", "").endswith(":latest"): x.append(f"{nom}: obraz tegi yo'q/latest")
        for p in v.get("ports", []):
            h = p.split(":")[0]
            if h in portlar: x.append(f"{nom}: xost porti {h} band ({portlar[h]})")
            portlar[h] = nom
        for d, shart in (v.get("depends_on") or {}).items():
            if d not in s: x.append(f"{nom}: depends_on noma'lum xizmat {d}")
            elif shart.get("condition") == "service_healthy" and "healthcheck" not in s[d]: x.append(f"{nom}: {d} da healthcheck yo'q")
        for vol in v.get("volumes", []):
            n = vol.split(":")[0]
            if not n.startswith(".") and not n.startswith("/") and n not in c.get("volumes", {}): x.append(f"{nom}: hajm {n} e'lon qilinmagan")
        for n in v.get("networks", []):
            if n not in c.get("networks", {}): x.append(f"{nom}: tarmoq {n} e'lon qilinmagan")
        if any(re.search("KEY|SECRET|PASS", k) for k in (v.get("environment") or {})): x.append(f"{nom}: sir environment ichida")
    # ishga tushish tartibi (topologik saralash) va sikl tekshiruvi
    tartib, qoldi = [], dict((k, set((v.get("depends_on") or {}).keys())) for k, v in s.items())
    while qoldi:
        tayyor = [k for k, d in qoldi.items() if not d - set(tartib)]
        if not tayyor: x.append("depends_on sikli!"); break
        tartib += sorted(tayyor); [qoldi.pop(k) for k in tayyor]
    return x, tartib


x, tartib = compose_tekshir(yaml.safe_load(matn))
print("Compose validatsiya:", "OK" if not x else x, "| ishga tushish tartibi:", " -> ".join(tartib))
buz = yaml.safe_load(matn)
buz["services"]["prometheus"]["ports"] = ["8000:9090"]; buz["services"]["redis"].pop("healthcheck")
buz["services"]["api"]["environment"]["API_KEY"] = "123"; buz["services"]["redis"]["depends_on"] = {"prometheus": {}}
x, _ = compose_tekshir(buz)
print("Buzilgan compose xatolari:"); [print("  -", e) for e in x]
print("\n=== Dockerfile va compose tekshiruvi tugadi ===")
