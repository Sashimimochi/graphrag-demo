# GraphRAG Demo

GraphRAGを試すサンプルコード

## Settings

### 環境変数の設定

`.env`ファイルを用意する。

```conf
API_KEY = "your_api_key" # OpenAI API KEY, Gemini API Key, etc.
NEO4J_URI = "bolt://host.docker.internal:7687" # Auraを使用する場合はインスタンス作成時に表示されるURI名をファイルベースストレージを使う場合は空文字
NEO4J_USERNAME = "neo4j" # Auraを使用する場合はインスタンス作成時に表示される値
NEO4J_PASSWORD = "graphrag" # Auraを使用する場合はインスタンス作成時に表示される値
NEO4J_BROWSER_URI = "http://localhost:7474" # Auraを使用する場合はNeo4jのコンソール画面 https://console-preview.neo4j.io/tools/query
```

原則、Geminiを使う想定なので、別のAPIを使用する場合は`BASE_URL`などを適宜変更する必要がある。

```python
BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai"

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
```

### データの準備

`data`ディレクトリを作成し、使用するテキストデータを`.txt`形式で配置する。

サンプルデータを使用する場合は以下のコマンドで取得できる。
サンプルデータは小説クリスマスキャロルになっている。
```bash
make setup
```

### コンテナ起動

```bash
make up
```

で起動できる。
起動が完了したら適当なブラウザで http://localhost:8501 を開く。

画面の案内にしたがって、順次処理を実行していく。

## Neo4jの使い方について

グラフストレージとしてNeo4jを使用する場合のTipsを記載する。

### サンプルクエリ

Neo4jでは、Cypherというグラフクエリ言語を用いてデータを操作する。
たとえば、全件取得する場合は以下を実行する。

```cypher
match (n) return n
```

### ローカルからマネージドサービスへのマイグレーション

※ neo4jのver.4.4.0以降でのみ有効

ローカル（主にDockerコンテナ）のNeo4jに保存したデータをマネージドサービスのNeo4j Auraに移行する手順を示す。
基本的には[公式ドキュメント](https://neo4j.com/docs/aura/classic/tutorials/migration/)にしたがう。

コンテナ内に入って以下のコマンドを実行する。
進捗が適宜表示されるので完了まで待つ。

```bash
bin/neo4j-admin dump --database=neo4j --to=/dumps/neo4j
bin/neo4j-admin push-to-cloud --dump=/dumps/neo4j/file.dump --bolt-uri=neo4j+s://xxxxxxxx.databases.neo4j.io --overwrite
Selecting JVM - Version:11.0.16+8, Name:OpenJDK 64-Bit Server VM, Vendor:Oracle Corporation
Neo4j aura username (default: neo4j):neo4j
Neo4j aura password for neo4j:
Upload
....................  10%
....................  20%
....................  30%
....................  40%
....................  50%
....................  60%
....................  70%
....................  80%
....................  90%
.................... 100%
We have received your export and it is currently being loaded into your Aura instance.
You can wait here, or abort this command and head over to the console to be notified of when your database is running.
Import progress (estimated)
....................  10%
....................  20%
....................  30%
....................  40%
....................  50%
....................  60%
....................  70%
....................  80%
....................  90%
.................... 100%
Your data was successfully pushed to Aura and is now running.
It is safe to delete the dump file now: /dumps/neo4j
```
