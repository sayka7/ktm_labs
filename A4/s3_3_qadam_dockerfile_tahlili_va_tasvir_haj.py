"""A4 — 3-qadam. Dockerfile tahlili va tasvir hajmini baholash

Docker bo'lmagani uchun Dockerfile matni dasturiy tahlil qilinadi: ko'p bosqichli qurish, qotirilgan versiya, yengil asos, root bo'lmagan USER, pip/apt kesh, HEALTHCHECK, EXPOSE, exec CMD, qatlamlar tartibi, ENV sozlamalar — 11 qoida. Yaxshi (slim + ONNX Runtime) va yomon (pytorch:latest, root) variantlar solishtiriladi; tasvir hajmi, tortish vaqti va 20 tugunli klasterga tarqatish hajmi baholanadi.

Kutilgan natija: Yaxshi variant 11/11, yomon 1/11; hajm bahosi ~270 MB vs ~8 GB (30x); 100 Mbit/s da tortish 22 s vs 11 daqiqa.
"""
# 3-qadam. Dockerfile tahlili: ko'p bosqichli qurish qoidalari va tasvir hajmini baholash (Docker'siz)
import re, pandas as pd

# --- Berilgan: ikki variant (o'zgartirmang)
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
TALABLAR = {"yaxshi": ["fastapi", "uvicorn", "onnxruntime", "numpy", "pydantic"],
            "yomon": ["fastapi", "uvicorn", "onnxruntime", "numpy", "pydantic", "torch", "torchvision"]}
HAJM_MB = {"python:3.12-slim": 130, "python:3.12": 1020, "pytorch/pytorch:latest": 7200, "nvidia/cuda:12.4.1-runtime-ubuntu22.04": 2900,
           "fastapi": 15, "uvicorn": 10, "onnxruntime": 60, "numpy": 35, "pydantic": 20, "torch": 900, "torchvision": 25, "apt_curl_kesh": 45}

def tahlil(df_text):
    satrlar = [l.strip() for l in df_text.strip().splitlines() if l.strip() and not l.startswith("#")]
    # TODO 1: FROM satrlarini toping; yakuniy bosqich = oxirgi FROM dan keyingi satrlar; asos = oxirgi FROM tasviri
    # TODO 2: 11 qoidani q lug'atiga True/False qilib yozing: ko'p bosqich, 'latest' emas, slim/runtime asos, USER root emas,
    #         pip --no-cache-dir, HEALTHCHECK, EXPOSE, CMD [..] exec shakli, requirements koddan oldin, apt keshi tozalangan, ENV MODEL_PATH
    raise NotImplementedError("TODO 1-2: Dockerfile qoidalari")

def hajm(asos, paketlar, apt_kesh=False):
    # TODO 3: asos tasvir + paketlar (+ apt kesh) hajmini HAJM_MB dan yig'ing (noma'lum asos 500, paket 20 MB)
    raise NotImplementedError("TODO 3: hajm")

print("=== Dockerfile tekshiruvi ===")
for nomi, text in (("yaxshi", YAXSHI), ("yomon", YOMON)):
    q, asos = tahlil(text)
    print(f"\n[{nomi.upper()}] asos: {asos}   ball {sum(q.values())}/{len(q)}")
    for k, v in q.items(): print(f"  {'PASS' if v else 'FAIL'}  {k}")

print("\n=== Tasvir hajmi bahosi (MB) ===")
# TODO 4: 4 variant (slim+ORT, python:3.12, pytorch+hamma narsa, cuda-runtime+250) jadvali: MB, nisbat,
#         100 Mbit/s da tortish vaqti (s) va 20 tugunga tarqatish hajmi (GB)
