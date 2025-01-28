import os
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.llm import openai_embedding, openai_complete_if_cache
from utils.graph_visualize import visualize_graphml
from neo4j import GraphDatabase

load_dotenv()
DATA_DIR = "./data"
API_KEY = os.getenv("API_KEY")
BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai"

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

if "conversation" not in st.session_state:
    st.session_state.conversation = []

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gemini-1.5-flash",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=API_KEY,
        base_url=BASE_URL,
        **kwargs
    )

async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embedding(
        texts,
        model="text-embedding-004",
        api_key=API_KEY,
        base_url=BASE_URL,
    )


def make_index(filepath):
    with open(os.path.join(DATA_DIR, f"{filepath}.txt")) as f:
        st.session_state.rag.insert(f.read())
    st.success("インデックスの作成が完了しました。")

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

def main():
    st.title("GraphRAG Demo")
    st.write("GraphRAGの威力を体感できるデモアプリです。")

    DATASET = st.text_input("Select Data", help="dataディレクトリに事前にテキスト形式で同名のインデックス用ファイルを保存しておく必要があります。サンプルデータを使用する場合は、`dickens`を選択してください。", placeholder="dickens")
    WORKING_DIR = DATASET

    if WORKING_DIR is None or WORKING_DIR == "":
        return

    LANGUAGE = st.selectbox("Select Language", ["Japanese", "English"], help="日本語のデータセット/質問をする場合は`Japanese`を選択してください。")

    filename = DATASET

    if not os.path.exists(WORKING_DIR):
        os.mkdir(WORKING_DIR)
        st.warning(f"`data`ディレクトリ配下に`{filename}.txt`を配置し、インデックスを作成してください。")

    graph_storage = "Neo4JStorage" if os.getenv("NEO4J_URI") else "NetworkXStorage"
    st.info(f"Using Graph Storage: {graph_storage}")
    
    if graph_storage == "Neo4JStorage":
        neo4j_uri = os.getenv('NEO4J_URI')
        username = os.getenv('NEO4J_USERNAME')
        password = os.getenv('NEO4J_PASSWORD')
        with GraphDatabase.driver(neo4j_uri, auth=(username, password)) as driver:
            driver.verify_connectivity()
            st.success("Successfully access neo4j")

    if "rag" not in st.session_state:
        rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=llm_model_func,
            llm_model_max_async=1,
            embedding_func=EmbeddingFunc(
                embedding_dim=768,
                max_token_size=4096,
                func=embedding_func
            ),
            graph_storage=graph_storage,
            addon_params = {
                "language": LANGUAGE,
                "entity_types": ["organization", "person", "geo", "event", "category", "product"],
            },
        )
        st.session_state.rag = rag

    mode = st.selectbox("Select Search Mode", ["naive", "local", "global", "hybrid", "mix"], key="mode", help="- `naive`: \n- `local`: 人物相関など特定の関係性について質問をする\n- `global`: 文章全体にまたがる抽象的な質問をする\n- `hybrid`: `local`と`global`の両方を混ぜたもの\n- `mix`: 知識グラフとベクトル検索を組み合わせて検索する")
    if st.button("Create Index", help="初めて使用するデータの場合は、質問の前にインデックスを作成してください。"):
        with st.spinner("Creating index..."):
            make_index(filename)

    if st.button("View Knowledge Graph", help="知識グラフを確認する"):
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

    # initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "私はお助けBotです。何かお手伝いできることがあれば聞いてください。"
            }
        ]

    # display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    # react to user input
    if prompt := st.chat_input("LLMへの質問内容を入力してください。ex.この文章の主要なテーマはなんですか？", key="query"):
        # display user message
        st.chat_message("user").markdown(prompt)
        # add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # display assistant message
        with st.chat_message("assistant"):
            with st.spinner("Assistant is thinking..."):
                placeholder = st.empty()
                msg = search(mode, prompt)
                placeholder.markdown(msg)
                # add assistant message to chat history
                st.session_state.messages.append({"role": "assistant", "content": msg})

if __name__ == "__main__":
    main()
