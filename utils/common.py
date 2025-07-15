import os
import streamlit as st
from neo4j import GraphDatabase
from dotenv import load_dotenv

# 環境変数をロード
load_dotenv()

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
