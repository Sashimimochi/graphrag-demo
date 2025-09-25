# ビルドステージ
FROM python:3.12 AS builder
WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel && \
  pip install --no-cache-dir --prefix=/install -r requirements.txt

# ランタイムステージ
FROM python:3.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

COPY --from=builder /install /usr/local
