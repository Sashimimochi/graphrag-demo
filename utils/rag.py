import os
import numpy as np
import tiktoken
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import (
    finalize_pipeline_ingress,
    initialize_pipeline_status,
)
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc, setup_logger
from utils.common import select_graph_storage
from raganything import RAGAnything, RAGAnythingConfig
from utils.common import ModalType

# 環境変数をロード
load_dotenv()

# 定数の設定
DATA_DIR = "./data"
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("API_HOST", "https://generativelanguage.googleapis.com/v1beta/openai/")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.8-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 2048))
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 768))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
MAX_ASYNC = int(os.getenv("MAX_ASYNC", 1))
EMBEDDING_TOKENIZER = tiktoken.get_encoding("cl100k_base")

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
async def embedding_func(
    texts: list[str],
    embedding_dim: int | None = None,
    max_token_size: int | None = None,
    context: str = "document",
) -> np.ndarray:
    """Generate one normalized Gemini embedding per input text."""
    dimension = embedding_dim or EMBEDDING_DIM
    token_limit = max_token_size or MAX_TOKENS
    if EMBEDDING_MODEL == "gemini-embedding-001":
        token_limit = min(token_limit, 2048)
    if not texts:
        return np.empty((0, dimension), dtype=np.float32)
    if not API_KEY:
        raise RuntimeError("Gemini API key is missing. Set API_KEY in .env.")

    # Match LightRAG's existing OpenAI embedding path: use cl100k_base for
    # unknown model tokenizers and truncate over-limit inputs before the API call.
    truncated_texts = []
    for text in texts:
        tokens = EMBEDDING_TOKENIZER.encode(text, disallowed_special=())
        if len(tokens) > token_limit:
            text = EMBEDDING_TOKENIZER.decode(tokens[:token_limit])
        truncated_texts.append(text)

    task_type = "RETRIEVAL_QUERY" if context == "query" else "RETRIEVAL_DOCUMENT"
    client = genai.Client(api_key=API_KEY)
    async with client.aio as async_client:
        response = await async_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=truncated_texts,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=dimension,
            ),
        )

    embeddings = response.embeddings or []
    if len(embeddings) != len(truncated_texts):
        raise RuntimeError(
            f"Gemini returned {len(embeddings)} embeddings for {len(truncated_texts)} input texts."
        )

    vectors = np.asarray([item.values for item in embeddings], dtype=np.float32)
    if vectors.shape != (len(truncated_texts), dimension):
        raise RuntimeError(
            f"Gemini returned embedding shape {vectors.shape}; expected {(len(truncated_texts), dimension)}."
        )

    # gemini-embedding-001 does not normalize truncated (non-3072) outputs.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, np.finfo(np.float32).eps)

async def vision_model_func(
        prompt, system_prompt=None,
        history_messages=[], image_data=None,
        messages=None, **kwargs):
    if messages:
        return await openai_complete_if_cache(
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
        return await openai_complete_if_cache(
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
        return await llm_model_func(
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
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
            send_dimensions=True,
            func=embedding_func
        ),
        graph_storage=graph_storage,
        addon_params={
            "language": language,
            "entity_types": ["organization", "person", "geo", "event", "category", "product"],
        },
    )
    await rag.initialize_storages()
    await initialize_pipeline_status(workspace=rag.workspace)
    return rag

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

async def _text_chunks_are_empty(rag):
    """Check the authoritative text-chunk store before querying."""
    return await rag.text_chunks.is_empty()


def _chunk_index_error(text_chunks_empty):
    if text_chunks_empty:
        return (
            "テキストチャンクが保存されていません。入力ファイルの内容と"
            "LightRAGのドキュメント処理ログを確認してください。"
        )
    return None


# インデックス作成関数の定義
async def make_index(filepath):
    lightrag = None
    try:
      lightrag = await initialize_rag()
      rag = initialize_rag_anything(lightrag)
      input_filepath = os.path.join(DATA_DIR, f"{filepath}.txt")
      st.info(f"インデックスを作成中です。しばらくお待ちください... {input_filepath}")
      with open(input_filepath) as f:
        content = f.read()
      track_id = await lightrag.ainsert(content)
      st.info(f"テキストのインデックス処理が完了しました。処理ID: {track_id}")

      text_chunks_empty = await _text_chunks_are_empty(lightrag)
      chunk_error = _chunk_index_error(text_chunks_empty)
      if chunk_error:
        raise RuntimeError(
          f"{chunk_error} (text_chunks_empty={text_chunks_empty}, "
          f"working_dir={lightrag.working_dir})"
        )

      await rag.process_folder_complete(
        folder_path=DATA_DIR,
        output_dir=st.session_state.working_dir,
        file_extensions=[".jpeg", ".jpg", ".png", ".pdf", ".pptx", ".docx"],
        recursive=True,
        max_workers=1,
      )
      text_chunks_empty = await _text_chunks_are_empty(lightrag)
      chunk_error = _chunk_index_error(text_chunks_empty)
      if chunk_error:
        raise RuntimeError(
          f"{chunk_error} (text_chunks_empty={text_chunks_empty}, "
          f"working_dir={lightrag.working_dir})"
        )

      if lightrag.graph_storage == "NetworkXStorage":
        graph_dir = os.path.join(lightrag.working_dir, lightrag.workspace) if lightrag.workspace else lightrag.working_dir
        graphml_path = os.path.join(graph_dir, "graph_chunk_entity_relation.graphml")
        if not os.path.isfile(graphml_path):
          raise RuntimeError(
            "LightRAGのテキスト処理は終了しましたが、GraphMLが保存されていません。"
            f" 保存先: {os.path.abspath(graphml_path)}"
          )

      st.success("インデックスの作成が完了しました。")
    except Exception as e:
      st.error(f"インデックスの作成に失敗しました: {e}")
    finally:
      if lightrag:
        try:
          await lightrag.finalize_storages()
        finally:
          # This app creates and tears down one LightRAG instance per request.
          # No sibling instance for this workspace remains, so its loop-bound
          # single-process ingress can be dropped before asyncio.run() closes.
          await finalize_pipeline_ingress(workspace=lightrag.workspace)

async def _query_lightrag(rag, query, mode):
    """Run a non-streaming query and surface LightRAG failure details."""
    result = await rag.aquery_llm(
        query,
        param=QueryParam(mode=mode, stream=False),
    )
    if result.get("status") != "success":
        reason = result.get("message") or "LightRAG query failed without a message"
        raise RuntimeError(reason)

    response = result.get("llm_response") or {}
    content = response.get("content")
    if content is None:
        raise RuntimeError("LightRAG completed without a text response")
    return content


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

        if modal == ModalType.TEXT_ONLY or modal == ModalType.BOTH:
            if mode in {"naive", "mix", "hybrid"}:
                text_chunks_empty = await _text_chunks_are_empty(rag)
                chunk_error = _chunk_index_error(text_chunks_empty)
                if chunk_error:
                    raise RuntimeError(
                        f"{chunk_error} 検索前チェック: "
                        f"text_chunks_empty={text_chunks_empty}, "
                        f"working_dir={rag.working_dir}"
                    )
        if modal == ModalType.TEXT_ONLY:
            msg = await _query_lightrag(rag, query, mode)
        elif modal == ModalType.MULTIMODAL:
            msg_multimodal = await rag_anything.aquery(
                query=query,
                mode=mode,
                vlm_enhanced=True,
            )
        elif modal == ModalType.MULTIMODAL_INPUT:
            if not img_base64:
                st.error("Please upload an image for Multimodal Input mode.")
                return "Error: No image provided for Multimodal Input mode."
            msg = await rag_anything.aquery_with_multimodal(
                query,
                multimodal_content=[{
                    "type": "image",
                    "source_type": "base64",
                    "mime_type": "image/png",
                    "data": img_base64,
                }],
                mode=mode,
            )
        elif modal == ModalType.BOTH:
            msg = await _query_lightrag(rag, query, mode)
            msg_multimodal = await rag_anything.aquery(
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
            try:
                await rag.finalize_storages()
            finally:
                await finalize_pipeline_ingress(workspace=rag.workspace)
    return msg
