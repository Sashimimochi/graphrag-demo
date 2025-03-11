import os
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.llm import openai_embedding, openai_complete_if_cache
from utils.graph_visualize import visualize_graphml, show_hierarchy_graph
from neo4j import GraphDatabase

# 環境変数をロード
load_dotenv()

# 定数の設定
DATA_DIR = "./data"
API_KEY = os.getenv("API_KEY")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"

# Streamlitのページ設定
def configure_page():
    st.set_page_config(
        page_title="GraphRAG Demo",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://docs.streamlit.io/',
            'Report a bug': "https://docs.streamlit.io/",
            'About': "# This is a header. This is an *extremely* cool app!"
        }
    )

# セッションステートの初期化
def initialize_session_state():
    if "conversation" not in st.session_state:
        st.session_state.conversation = []

# LLMモデル関数の定義
async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return await openai_complete_if_cache(
        "gemini-1.5-flash",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=API_KEY,
        base_url=BASE_URL,
        **kwargs
    )

# 埋め込み関数の定義
async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embedding(
        texts,
        model="text-embedding-004",
        api_key=API_KEY,
        base_url=BASE_URL,
    )

# インデックス作成関数の定義
def make_index(filepath):
    with open(os.path.join(DATA_DIR, f"{filepath}.txt")) as f:
        st.session_state.rag.insert(f.read())
    st.success("インデックスの作成が完了しました。")

# 検索関数の定義
def search(mode, query="この文章を読むとどのような知見が得られるか簡潔にまとめてください。"):
    """
    Perform mix search (Knowledge Graph + Vector Retrieval)
    Mix mode combines knowledge graph and vector search:
    - Uses both structured (KG) and unstructured (vector) information
    - Provides comprehensive answers by analyzing relationships and context
    - Supports image content through HTML img tags
    - Allows control over retrieval depth via top_k parameter
    """
    msg = st.session_state.rag.query(query, param=QueryParam(mode=mode))
    print("="*100)
    print(f"\nquestion: {query}")
    print(f"[Mode: {mode} Search]")
    print("-"*30)
    print("")
    print(msg)
    return msg

# データセットの選択
def select_dataset():
    return st.text_input(
        "Select Data",
        help="dataディレクトリに事前にテキスト形式で同名のインデックス用ファイルを保存しておく必要があります。サンプルデータを使用する場合は、`dickens`を選択してください。",
        placeholder="dickens"
    )

# 言語の選択
def select_language():
    return st.selectbox(
        "Select Language",
        ["Japanese", "English"],
        help="日本語のデータセット/質問をする場合は`Japanese`を選択してください。"
    )

# グラフストレージの選択
def select_graph_storage():
    return "Neo4JStorage" if os.getenv("NEO4J_URI") else "NetworkXStorage"

# Neo4jストレージの接続確認
def verify_neo4j_connection():
    neo4j_uri = os.getenv('NEO4J_URI')
    username = os.getenv('NEO4J_USERNAME')
    password = os.getenv('NEO4J_PASSWORD')
    with GraphDatabase.driver(neo4j_uri, auth=(username, password)) as driver:
        driver.verify_connectivity()
        st.success("Successfully access neo4j")

# RAGオブジェクトの初期化
def initialize_rag(working_dir, language, graph_storage):
    if "rag" not in st.session_state:
        rag = LightRAG(
            working_dir=working_dir,
            llm_model_func=llm_model_func,
            llm_model_max_async=1,
            embedding_func=EmbeddingFunc(
                embedding_dim=768,
                max_token_size=4096,
                func=embedding_func
            ),
            graph_storage=graph_storage,
            addon_params={
                "language": language,
                "entity_types": ["organization", "person", "geo", "event", "category", "product"],
            },
        )
        st.session_state.rag = rag

# 検索モードの選択
def select_search_mode():
    return st.selectbox(
        "Select Search Mode",
        ["naive", "local", "global", "hybrid", "mix"],
        key="mode",
        help="- `naive`: 単純な類似検索\n- `local`: 人物相関など特定の関係性について質問をする\n- `global`: 文章全体にまたがる抽象的な質問をする\n- `hybrid`: `local`と`global`の両方を混ぜたもの\n- `mix`: 知識グラフとベクトル検索を組み合わせて検索する"
    )

# 知識グラフの表示
def display_knowledge_graph(graph_storage, filename):
    if graph_storage == "Neo4JStorage":
        st.markdown(f"""
            <a href="{os.getenv("NEO4J_BROWSER_URI")}" target="_blank">
                Neo4j
            </a>
        """, unsafe_allow_html=True)
    else:
        filepath = f"./visualize/knowledge_graph_{filename}.html"
        if not os.path.exists(filepath):
            visualize_graphml(filename, filepath)
        with open(filepath, "r") as f:
            components.html(f.read(), height=500)
        df = show_hierarchy_graph(filename)
        st.dataframe(df)

# チャット履歴の初期化
def initialize_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "私はお助けBotです。何かお手伝いできることがあれば聞いてください。"
            }
        ]

# チャット履歴の表示
def display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# ユーザー入力に反応
def handle_user_input(mode):
    if prompt := st.chat_input("LLMへの質問内容を入力してください。ex.この文章の主要なテーマはなんですか？", key="query"):
        # ユーザーメッセージの表示
        st.chat_message("user").markdown(prompt)
        # ユーザーメッセージをチャット履歴に追加
        st.session_state.messages.append({"role": "user", "content": prompt})
        # アシスタントメッセージの表示
        with st.chat_message("assistant"):
            with st.spinner("Assistant is thinking..."):
                placeholder = st.empty()
                msg = search(mode, prompt)
                placeholder.markdown(msg)
                # アシスタントメッセージをチャット履歴に追加
                st.session_state.messages.append({"role": "assistant", "content": msg})

# メイン関数の定義
def main():
    configure_page()
    initialize_session_state()

    st.title("GraphRAG Demo")
    st.write("GraphRAGの威力を体感できるデモアプリです。")

    DATASET = select_dataset()
    WORKING_DIR = DATASET

    if not WORKING_DIR:
        return

    LANGUAGE = select_language()
    filename = DATASET

    if not os.path.exists(WORKING_DIR):
        os.mkdir(WORKING_DIR)
        st.warning(f"`data`ディレクトリ配下に`{filename}.txt`を配置し、インデックスを作成してください。")

    graph_storage = select_graph_storage()
    st.info(f"Using Graph Storage: {graph_storage}")

    if graph_storage == "Neo4JStorage":
        verify_neo4j_connection()

    initialize_rag(WORKING_DIR, LANGUAGE, graph_storage)

    mode = select_search_mode()

    if st.button("Create Index", help="初めて使用するデータの場合は、質問の前にインデックスを作成してください。"):
        with st.spinner("Creating index..."):
            make_index(filename)

    if st.button("View Knowledge Graph", help="知識グラフを確認する"):
        display_knowledge_graph(graph_storage, filename)

    initialize_chat_history()
    display_chat_history()
    handle_user_input(mode)

# メイン関数の実行
if __name__ == "__main__":
    main()
