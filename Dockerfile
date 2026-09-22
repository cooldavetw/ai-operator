FROM python:3.12-slim-bookworm AS builder
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake pkg-config libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY requirements.txt .
# Avoid compiling for the build host's CPU. The default target is x86-64 AVX2;
# change the argument and rebuild for an older VMware CPU compatibility level.
ARG CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_AVX512=OFF -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
ENV CMAKE_ARGS=${CMAKE_ARGS} CMAKE_BUILD_PARALLEL_LEVEL=4
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libopenblas0-pthread libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 classifier \
    && useradd --uid 10001 --gid classifier --no-create-home classifier
WORKDIR /srv
COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && pip freeze > /srv/dependency-manifest.txt \
    && rm -rf /wheels
COPY app ./app
COPY config ./config
RUN mkdir -p /data /models && chown classifier:classifier /data
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    CONFIG_PATH=/srv/config/domains.yaml DATABASE_PATH=/data/classifier.sqlite3 \
    MODEL_PATH=/models/model.gguf OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=240s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=3)"
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]

