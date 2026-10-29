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
    # TODO 1: builder (requirements -> pip wheel) va yakuniy bosqich (non-root USER, COPY --from, HEALTHCHECK, exec-CMD) matnini qaytaring
    raise NotImplementedError("TODO 1: Dockerfile shabloni")


YOMON = """FROM python:latest
ENV API_KEY=maxfiy123
COPY . .
RUN pip install fastapi uvicorn scikit-learn
RUN apt-get update
RUN apt-get install -y curl
EXPOSE 8080
CMD uvicorn app.main:app --port 8000
"""

QOIDALAR = ["D01 ko'p bosqichli (>=2 FROM)", "D02 bazaviy obraz teg bilan, ':latest' emas", "D03 slim/distroless baza",
            "D04 USER (root emas)", "D05 ENV da sir yo'q", "D06 'COPY . .' yo'q", "D07 requirements koddan oldin (kesh)",
            "D08 pip --no-cache-dir", "D09 versiyalar qotirilgan (==)", "D10 HEALTHCHECK bor", "D11 EXPOSE = ilova porti",
            "D12 CMD exec shaklida (JSON)", "D13 apt-get update alohida RUN emas"]


def tahlil(txt, port):
    """Qaytaradi: {qoida: True/False} — QOIDALAR ro'yxatidagi har biri uchun."""
    # TODO 2: izohsiz qatorlar, ko'rsatmalar va FROM larni ajrating; har qoidani satr/regex tekshiruvi bilan yozing
    raise NotImplementedError("TODO 2: Dockerfile linter")


YAXSHI = dockerfile(LOYIHA)
print("=== Dockerfile qoidalari: yaxshi vs yomon ===")
# TODO 3: ikkala Dockerfile uchun PASS/FAIL jadvali va ballni chiqaring


def qurish(txt, fayllar, kesh):
    """Har ko'rsatma xeshi = oldingi xesh + ko'rsatma + tegishli fayllar xeshi. Qaytaradi: qayta qurilgan RUN/COPY qatlamlar."""
    # TODO 4: FROM yangi zanjir boshlaydi; COPY ga fayllar mazmuni, COPY --from ga o'sha bosqich xeshi qo'shiladi; keshda yo'q xeshlar qayta quriladi
    raise NotImplementedError("TODO 4: qatlam keshi")


VAQT = {"RUN pip": 95, "RUN groupadd": 2}            # taxminiy qatlam qurish vaqti, s (qolganlari 3 s)
fayllar = {"requirements.txt": "fastapi==0.115.6...", "app/main.py": "v1-kod", "app/model.py": "m"}
print("\n=== Qatlam keshi (multi-stage, requirements alohida) ===")
# TODO 5: avval to'liq qurib keshni to'ldiring; keyin faqat kod va faqat requirements o'zgarganda nechta qatlam (va necha s) qayta qurilishini chiqaring; YOMON bilan solishtiring

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


def compose_tekshir(c):
    """Qaytaradi: (xatolar ro'yxati, ishga tushish tartibi)."""
    # TODO 6: obraz tegi, xost port to'qnashuvi, depends_on + healthcheck, hajm/tarmoq e'loni, environment ichidagi sirlarni tekshiring
    # TODO 7: depends_on bo'yicha topologik saralash va sikl aniqlash
    raise NotImplementedError("TODO 6-7: compose validatsiya")


x, tartib = compose_tekshir(yaml.safe_load(matn))
print("Compose validatsiya:", "OK" if not x else x, "| ishga tushish tartibi:", " -> ".join(tartib))
# TODO 8: compose'ni ataylab buzing (port, healthcheck, sir, sikl) va 4 xato chiqishini ko'rsating
print("\n=== Dockerfile va compose tekshiruvi tugadi ===")
