# --- ビルドステージ ---
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

# uvのキャッシュを利用して高速化
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

WORKDIR /app

# requirements.txt は requirements.in から uv pip compile で生成したlockfile。
COPY requirements.txt .

# 生成済みのlockfileから、固定された依存関係をインストール
RUN uv pip install --no-cache --prefix=/install -r requirements.txt

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
