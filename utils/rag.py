import os
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import openai_embed, openai_complete_if_cache
from lightrag.utils import EmbeddingFunc, setup_logger
from utils.common import select_graph_storage
from raganything import RAGAnything, RAGAnythingConfig
from utils.common import ModalType

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

def vision_model_func(
        prompt, system_prompt=None,
        history_messages=[], image_data=None,
        messages=None, **kwargs):
    if messages:
        return openai_complete_if_cache(
            LLM_MODEL,
            "",
            system_prompt=None,
            history_messages=[],
            messages=messages,
            api_key=API_KEY,
            base_url=BASE_URL,
            **kwargs,
        )
    elif image_data:
        return openai_complete_if_cache(
            LLM_MODEL,
            "",
            system_prompt=None,
            history_messages=[],
            messages=[
                {"role": "system", "content": system_prompt}
                if system_prompt
                else None,
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_data}"},
                        },
                    ],
                }
                if image_data
                else {"role": "user", "content": prompt},
            ],
            api_key=API_KEY,
            base_url=BASE_URL,
            **kwargs,
        )
    else:
        return llm_model_func(
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
        )

def initialize_rag_anything(rag):
    config = RAGAnythingConfig(
       working_dir=st.session_state.working_dir,
       parser="mineru",
       parse_method="auto",
       enable_image_processing=True,
       enable_table_processing=True,
       enable_equation_processing=True,
    )

    rag_anything = RAGAnything(
       lightrag=rag,
       config=config,
       vision_model_func=vision_model_func,
    )
    return rag_anything

# インデックス作成関数の定義
async def make_index(filepath):
    rag = None
    try:
      lightrag = await initialize_rag()
      rag = initialize_rag_anything(lightrag)
      input_filepath = os.path.join(DATA_DIR, f"{filepath}.txt")
      st.info(f"インデックスを作成中です。しばらくお待ちください... {input_filepath}")
      with open(input_filepath) as f:
        content = f.read()
      lightrag.insert(content)
      await rag.process_folder_complete(
        folder_path="data",
        output_dir=st.session_state.working_dir,
        file_extensions=[".jpeg", ".jpg", ".png", ".pdf", ".pptx", ".docx"],
        recursive=True,
        max_workers=1,
      )
      st.success("インデックスの作成が完了しました。")
    except Exception as e:
      st.error(f"インデックスの作成に失敗しました: {e}")
    finally:
      if rag:
        await rag.finalize_storages()

# 検索関数の定義
async def search(mode, query="この文章を読むとどのような知見が得られるか簡潔にまとめてください。", modal=ModalType.TEXT_ONLY, img_base64=None):
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
    msg_multimodal = ""
    try:
        rag = await initialize_rag()
        rag_anything = initialize_rag_anything(rag)

        if modal == ModalType.TEXT_ONLY:
            msg = rag.query(query, param=QueryParam(mode=mode))
        elif modal == ModalType.MULTIMODAL:
            msg_multimodal = rag_anything.query(
                query=query,
                mode=mode,
                vlm_enhanced=True,
            )
        elif modal == ModalType.MULTIMODAL_INPUT:
            if not img_base64:
                st.error("Please upload an image for Multimodal Input mode.")
                return "Error: No image provided for Multimodal Input mode."
            msg = rag_anything.query_with_multimodal(
                query,
                multimodal_content=[{
                    "type": "image",
                    "source_type": "base64",
                    "mine_type": "image/png",
                    "data": img_base64,
                }],
                mode=mode,
            )
        elif modal == ModalType.BOTH:
            msg = rag.query(query, param=QueryParam(mode=mode))
            msg_multimodal = rag_anything.query(
                query=query,
                mode=mode,
                vlm_enhanced=True,
            )
        else:
            st.error(f"Invalid modal type: {modal}")
            return f"Error: Invalid modal type: {modal}"

        print("="*100)
        print(f"\nquestion: {query}")
        print(f"[Mode: {mode} Search]")
        print("-"*30)
        print("")
        print(msg)
        if modal == ModalType.MULTIMODAL:
            msg = msg_multimodal
        elif modal == ModalType.BOTH:
            msg = f"#### Text Only\n{msg}\n\n#### Multimodal\n{msg_multimodal}"
    except Exception as e:
        st.error(f"Search failed: {e}")
        msg = f"Search failed: {e}"
    finally:
        if rag:
            await rag.finalize_storages()
    return msg
