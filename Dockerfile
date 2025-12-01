# ビルドステージ
FROM python:3.12 AS builder
WORKDIR /app

COPY requirements.txt .

# Install mineru without dependencies first to allow pdfminer.six security update
# mineru pins pdfminer.six to vulnerable version, but we override it for CWE-502 fix
# Note: mineru version must match requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
  pip install --no-cache-dir --prefix=/install --no-deps mineru==2.6.5 && \
  pip install --no-cache-dir --prefix=/install -r requirements.txt

# ランタイムステージ
FROM python:3.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

COPY --from=builder /install /usr/local
