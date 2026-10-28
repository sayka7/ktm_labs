FROM python:3.12-slim
ENV PIP_NO_CACHE_DIR=1 MPLBACKEND=Agg
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
WORKDIR /work
CMD ["python3"]
