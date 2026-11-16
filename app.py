import os
import asyncio
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from utils.graph_visualize import resolve_graphml_path, visualize_graphml, show_hierarchy_graph
from utils.rag import make_index, search
from utils.common import select_dataset, select_language, select_graph_storage, check_storage, select_search_mode, select_modal, upload_image, ModalType

# 環境変数をロード
load_dotenv()

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
    if "language" not in st.session_state:
        st.session_state.language = ""
    if "working_dir" not in st.session_state:
        st.session_state.working_dir = ""

# 知識グラフの表示
def display_knowledge_graph(graph_storage, filename):
    if graph_storage == "Neo4JStorage":
        st.markdown(f"""
            <a href="{os.getenv("NEO4J_BROWSER_URI")}" target="_blank">
                Neo4j
            </a>
        """, unsafe_allow_html=True)
    else:
        graphml_file = resolve_graphml_path(filename)
        if not os.path.isfile(graphml_file):
            workspace = os.getenv("WORKSPACE", "").strip() or "(未設定)"
            st.error("GraphMLファイルが見つかりません。インデックス作成が完了しているか、保存先を確認してください。")
            st.caption(f"確認したパス: {os.path.abspath(graphml_file)}")
            st.caption(f"WORKSPACE: {workspace}")
            return

        filepath = Path("./visualize") / f"knowledge_graph_{filename}.html"
        # Regenerate on every explicit view request so old cached HTML from a
        # previous visualization implementation cannot remain visible.
        visualize_graphml(graphml_file, str(filepath))

        st.iframe(
            filepath,
            height=800,
            width="stretch",
            alt="Knowledge graph visualization",
        )
        with st.expander("クラスタ階層データ", expanded=False):
            df = show_hierarchy_graph(graphml_file)
            st.dataframe(df, width="stretch")

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
async def handle_user_input(mode, modal, img_base64=None):
    if prompt := st.chat_input("LLMへの質問内容を入力してください。ex.この文章の主要なテーマはなんですか？", key="query"):
        # ユーザーメッセージの表示
        st.chat_message("user").markdown(prompt)
        # ユーザーメッセージをチャット履歴に追加
        st.session_state.messages.append({"role": "user", "content": prompt})
        # アシスタントメッセージの表示
        with st.chat_message("assistant"):
            with st.spinner("Assistant is thinking..."):
                placeholder = st.empty()
                msg = await search(mode, query=prompt, modal=modal, img_base64=img_base64)
                placeholder.markdown(msg)
                # アシスタントメッセージをチャット履歴に追加
                st.session_state.messages.append({"role": "assistant", "content": msg})

# メイン関数の定義
async def main():
    configure_page()
    initialize_session_state()

    st.title("GraphRAG Demo")
    st.write("LightRAGを使ったグラフベースRAGの動作を体感できるデモアプリです。")

    DATASET = select_dataset()
    st.session_state.working_dir = DATASET
    filename = DATASET
    if not st.session_state.working_dir:
        return

    st.session_state.language = select_language()

    graph_storage = select_graph_storage()
    check_storage(st.session_state.working_dir, filename)

    if st.button("Create Index", help="初めて使用するデータの場合は、質問の前にインデックスを作成してください。"):
        with st.spinner("Creating index..."):
            await make_index(filename)
    if st.button("View Knowledge Graph", help="知識グラフを確認する"):
        display_knowledge_graph(graph_storage, filename)

    mode = select_search_mode()
    modal = select_modal()
    initialize_chat_history()
    display_chat_history()
    img_base64 = None
    if modal == ModalType.MULTIMODAL_INPUT:
        img_base64 = upload_image()
    await handle_user_input(mode, modal, img_base64)

# メイン関数の実行
if __name__ == "__main__":
    asyncio.run(main())
