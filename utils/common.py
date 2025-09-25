import os
import io
import base64
import streamlit as st
from PIL import Image
from neo4j import GraphDatabase
from dotenv import load_dotenv
from enum import Enum

# 環境変数をロード
load_dotenv()

class ModalType(Enum):
    TEXT_ONLY = "Text Only"
    MULTIMODAL = "Multimodal"
    MULTIMODAL_INPUT = "Multimodal Input"
    BOTH = "Both"

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
    graph_storage = "Neo4JStorage" if os.getenv("NEO4J_URI") else "NetworkXStorage"
    st.info(f"Using Graph Storage: {graph_storage}")
    if graph_storage == "Neo4JStorage":
        verify_neo4j_connection()
    return graph_storage

# Neo4jストレージの接続確認
def verify_neo4j_connection():
    neo4j_uri = os.getenv('NEO4J_URI')
    username = os.getenv('NEO4J_USERNAME')
    password = os.getenv('NEO4J_PASSWORD')
    with GraphDatabase.driver(neo4j_uri, auth=(username, password)) as driver:
        driver.verify_connectivity()
        st.success("Successfully access neo4j")

# インデックスデータの確認
def check_storage(working_dir, filename):
    if not os.path.exists(working_dir):
        os.mkdir(working_dir)
        st.warning(f"`data`ディレクトリ配下に`{filename}.txt`を配置し、インデックスを作成してください。")

# 検索モードの選択
def select_search_mode():
    return st.selectbox(
        "Select Search Mode",
        ["naive", "local", "global", "hybrid", "mix"],
        key="mode",
        help="- `naive`: 単純な類似検索\n- `local`: 人物相関など特定の関係性について質問をする\n- `global`: 文章全体にまたがる抽象的な質問をする\n- `hybrid`: `local`と`global`の両方を混ぜたもの\n- `mix`: 知識グラフとベクトル検索を組み合わせて検索する"
    )

def select_modal():
    modal = st.selectbox(
        "Select Query Modal",
        [modal.value for modal in ModalType],
        key="modal",
        help="- `Text Only`: テキストだけで検索する\n- `Multimodal`: 画像や表を含むマルチモーダル検索を行う\n- `Multimodal Input`: 画像を入力して検索を行う（画像アップロードが必要です）\n- `Both`: テキストとマルチモーダル両方の出力をする"
    )
    return ModalType(modal)

def upload_image():
    uploaded_file = st.file_uploader("Upload an image for Multimodal Input mode", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_container_width=True)

        img_buffer = io.BytesIO()
        image.save(img_buffer, format="PNG")
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
        return img_base64
    return None
