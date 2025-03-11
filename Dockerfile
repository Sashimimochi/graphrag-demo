# ビルドステージ
FROM python:3.12 AS builder
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ランタイムステージ
FROM python:3.12-slim
WORKDIR /app

COPY --from=builder /install /usr/local
