import os
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import openai_embed, openai_complete_if_cache
from lightrag.utils import EmbeddingFunc, setup_logger
from utils.common import select_graph_storage

# 環境変数をロード
load_dotenv()

# 定数の設定
DATA_DIR = "./data"
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("API_HOST", "https://generativelanguage.googleapis.com/v1beta/openai")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 4096))
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 768))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
MAX_ASYNC = int(os.getenv("MAX_ASYNC", 1))

setup_logger("lightrag", level="INFO")

# LLMモデル関数の定義
async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    """
    async with semaphore:
        return await openai_complete_if_cache(
            LLM_MODEL,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=API_KEY,
            base_url=BASE_URL,
            **kwargs
        )
    """
    return await openai_complete_if_cache(
      LLM_MODEL,
      prompt,
      system_prompt=system_prompt,
      history_messages=history_messages,
      api_key=API_KEY,
      base_url=BASE_URL,
      **kwargs
    )

# 埋め込み関数の定義
async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embed(
        texts,
        model=EMBEDDING_MODEL,
        api_key=API_KEY,
        base_url=BASE_URL,
    )

# RAGオブジェクトの初期化
async def initialize_rag():
    graph_storage = select_graph_storage()
    working_dir = st.session_state.working_dir
    language = st.session_state.language

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_model_func,
        llm_model_max_async=MAX_ASYNC,
        embedding_func_max_async=MAX_ASYNC,
        chunk_token_size=CHUNK_SIZE,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=MAX_TOKENS,
            func=embedding_func
        ),
        graph_storage=graph_storage,
        addon_params={
            "language": language,
            "entity_types": ["organization", "person", "geo", "event", "category", "product"],
        },
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag

# インデックス作成関数の定義
async def make_index(filepath):
    rag = None
    try:
      rag = await initialize_rag()
      with open(os.path.join(DATA_DIR, f"{filepath}.txt")) as f:
        content = f.read()
      rag.insert(content)
      st.success("インデックスの作成が完了しました。")
    except Exception as e:
      st.error(f"インデックスの作成に失敗しました: {e}")
    finally:
      if rag:
        await rag.finalize_storages()

# 検索関数の定義
async def search(mode, query="この文章を読むとどのような知見が得られるか簡潔にまとめてください。"):
    """
    Perform mix search (Knowledge Graph + Vector Retrieval)
    Mix mode combines knowledge graph and vector search:
    - Uses both structured (KG) and unstructured (vector) information
    - Provides comprehensive answers by analyzing relationships and context
    - Supports image content through HTML img tags
    - Allows control over retrieval depth via top_k parameter
    """
    rag = None
    msg = ""
    try:
        rag = await initialize_rag()
        msg = rag.query(query, param=QueryParam(mode=mode))
        print("="*100)
        print(f"\nquestion: {query}")
        print(f"[Mode: {mode} Search]")
        print("-"*30)
        print("")
        print(msg)
    except Exception as e:
        st.error(f"Search failed: {e}")
        msg = f"Search failed: {e}"
    finally:
        if rag:
            await rag.finalize_storages()
    return msg
