"""A4 — 3-qadam. Dockerfile tahlili va tasvir hajmini baholash

Docker bo'lmagani uchun Dockerfile matni dasturiy tahlil qilinadi: ko'p bosqichli qurish, qotirilgan versiya, yengil asos, root bo'lmagan USER, pip/apt kesh, HEALTHCHECK, EXPOSE, exec CMD, qatlamlar tartibi, ENV sozlamalar — 11 qoida. Yaxshi (slim + ONNX Runtime) va yomon (pytorch:latest, root) variantlar solishtiriladi; tasvir hajmi, tortish vaqti va 20 tugunli klasterga tarqatish hajmi baholanadi.

Kutilgan natija: Yaxshi variant 11/11, yomon 1/11; hajm bahosi ~270 MB vs ~8 GB (30x); 100 Mbit/s da tortish 22 s vs 11 daqiqa.
"""
# 3-qadam. Dockerfile tahlili: ko'p bosqichli qurish qoidalari va tasvir hajmini baholash (Docker'siz)
import re, pandas as pd

# Ikki variant: yaxshi (ko'p bosqichli, root'siz, ORT) va yomon (bitta bosqich, root, to'liq PyTorch)
YAXSHI = """
FROM python:3.12-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.12-slim
RUN useradd -m -u 10001 infer && mkdir -p /models && chown infer /models
COPY --from=builder /install /usr/local
WORKDIR /app
COPY app/ /app/
USER infer
ENV MODEL_PATH=/models/digits_int8.onnx ORT_THREADS=2 PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/readyz')"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
"""
YOMON = """
FROM pytorch/pytorch:latest
WORKDIR /app
COPY . /app
RUN pip install fastapi uvicorn onnxruntime
RUN apt-get update && apt-get install -y curl
EXPOSE 8000
CMD uvicorn main:app --host 0.0.0.0 --port 8000
"""
# requirements.txt tarkibi (hajm bahosi uchun)
TALABLAR = {"yaxshi": ["fastapi", "uvicorn", "onnxruntime", "numpy", "pydantic"],
            "yomon": ["fastapi", "uvicorn", "onnxruntime", "numpy", "pydantic", "torch", "torchvision"]}
# Taxminiy o'rnatilgan hajmlar, MB (pip show / du bilan o'lchanadi; CPU g'ildiraklar)
HAJM_MB = {"python:3.12-slim": 130, "python:3.12": 1020, "pytorch/pytorch:latest": 7200, "nvidia/cuda:12.4.1-runtime-ubuntu22.04": 2900,
           "fastapi": 15, "uvicorn": 10, "onnxruntime": 60, "numpy": 35, "pydantic": 20, "torch": 900, "torchvision": 25, "apt_curl_kesh": 45}

def tahlil(df_text):
    satrlar = [l.strip() for l in df_text.strip().splitlines() if l.strip() and not l.startswith("#")]
    froms = [l for l in satrlar if l.upper().startswith("FROM ")]
    oxirgi_from = satrlar.index(froms[-1]) if froms else 0
    yakuniy = satrlar[oxirgi_from:]                       # faqat oxirgi bosqich tasvirga kiradi
    asos = froms[-1].split()[1]
    q = {}
    q["Ko'p bosqichli qurish (>=2 FROM)"] = len(froms) >= 2
    q["Asos tasvir 'latest' emas (versiya qotirilgan)"] = not any(f.split()[1].endswith(":latest") or ":" not in f.split()[1] for f in froms)
    q["Yengil asos tasvir (slim/runtime)"] = any(k in asos for k in ("slim", "runtime", "alpine", "distroless"))
    q["USER root emas (yakuniy bosqichda)"] = any(l.upper().startswith("USER ") and l.split()[1] != "root" for l in yakuniy)
    q["pip --no-cache-dir"] = all("--no-cache-dir" in l for l in satrlar if "pip install" in l)
    q["HEALTHCHECK bor"] = any(l.upper().startswith("HEALTHCHECK") for l in yakuniy)
    q["EXPOSE bor"] = any(l.upper().startswith("EXPOSE") for l in yakuniy)
    q["CMD exec (JSON) shaklida"] = any(l.startswith("CMD [") for l in yakuniy)
    q["Bog'liqliklar koddan OLDIN ko'chiriladi (kesh)"] = any("requirements" in l for l in satrlar) and \
        next((i for i, l in enumerate(satrlar) if "requirements" in l), 99) < next((i for i, l in enumerate(satrlar) if re.match(r"COPY (app/?|\.) ", l)), 0)
    q["apt keshi tozalangan"] = all("rm -rf /var/lib/apt/lists" in l for l in satrlar if "apt-get install" in l)
    q["Sozlamalar ENV orqali (12 omil)"] = any(l.startswith("ENV ") and "MODEL_PATH" in l for l in yakuniy)
    return q, asos

def hajm(asos, paketlar, apt_kesh=False):
    mb = HAJM_MB.get(asos, 500) + sum(HAJM_MB.get(p, 20) for p in paketlar) + (HAJM_MB["apt_curl_kesh"] if apt_kesh else 0)
    return mb

print("=== Dockerfile tekshiruvi ===")
natija = {}
for nomi, text in (("yaxshi", YAXSHI), ("yomon", YOMON)):
    q, asos = tahlil(text); natija[nomi] = q
    ball = sum(q.values())
    print(f"\n[{nomi.upper()}] asos: {asos}   ball {ball}/{len(q)}")
    for k, v in q.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
jad = pd.DataFrame({"qoida": list(natija["yaxshi"]), "yaxshi": list(natija["yaxshi"].values()), "yomon": list(natija["yomon"].values())})
print("\nQiyosiy jadval:"); print(jad.replace({True: "+", False: "-"}).to_string(index=False))

print("\n=== Tasvir hajmi bahosi (MB) ===")
rows = [dict(variant="yaxshi: slim + ORT (ko'p bosqichli)", MB=hajm("python:3.12-slim", TALABLAR["yaxshi"])),
        dict(variant="o'rta: python:3.12 (to'liq) + ORT", MB=hajm("python:3.12", TALABLAR["yaxshi"])),
        dict(variant="yomon: pytorch/pytorch + hamma narsa", MB=hajm("pytorch/pytorch:latest", TALABLAR["yomon"], apt_kesh=True)),
        dict(variant="GPU: cuda-runtime + onnxruntime-gpu", MB=hajm("nvidia/cuda:12.4.1-runtime-ubuntu22.04", TALABLAR["yaxshi"]) + 250)]
dh = pd.DataFrame(rows); dh["nisbat"] = (dh.MB / dh.MB[0]).round(1)
# Tortish vaqti 100 Mbit/s tarmoqda va 20 tugunli klasterga tarqatish
dh["tortish_s_100Mbit"] = (dh.MB * 8 / 100).round(0); dh["20_tugun_GB"] = (dh.MB * 20 / 1024).round(1)
print(dh.to_string(index=False))
print("\nXulosa: inferens uchun PyTorch kerak emas — ONNX Runtime yetarli; yengil asos + ko'p bosqichli qurish tasvirni ~15x kichraytiradi,")
print("  bu K8s'da yangi Pod'ning ishga tushish vaqtini (tortish + start) va HPA reaksiyasini bevosita belgilaydi.")
print("Real labda: docker build -t digits-infer:1.0 . && docker images digits-infer && docker run --cpus=2 --memory=1g ...")
