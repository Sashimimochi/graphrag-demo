# --- ビルドステージ ---
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

# uvのキャッシュを利用して高速化
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

WORKDIR /app

# 入力となる「直接の要求」ファイルをコピー
COPY requirements.in .

# 1. requirements.in から依存関係を解決し、一時的な requirements.txt を生成
# 2. 生成された requirements.txt を元に /install にライブラリをインストール
# これにより、pdfminer.six の上書きなども含めた整合性がこの中で解決されます
RUN uv pip compile requirements.in -o requirements.txt && \
  uv pip install --no-cache --prefix=/install -r requirements.txt

# --- ランタイムステージ ---
FROM python:3.12-slim
WORKDIR /app

# 実行に必要な最小限のシステムライブラリ
RUN apt-get update && apt-get install -y \
  libgl1 \
  libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

# ビルドステージからインストール済みライブラリをコピー
COPY --from=builder /install /usr/local
