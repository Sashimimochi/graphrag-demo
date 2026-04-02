setup:
	mkdir -p data
	curl https://raw.githubusercontent.com/gusye1234/nano-graphrag/main/tests/mock_data.txt > ./data/dickens.txt
	@make up

launch:
	docker-compose up -d

launch-local:
	docker-compose --profile local up -d

open:
	open http://localhost:8501
	open http://localhost:9621

up:
	@make launch
	@make open

up-local:
	@make launch-local
	@make open

check-mineru:
	docker compose exec app mineru --version
	docker compose exec app python -c "from raganything import RAGAnything; rag = RAGAnything(); print('✅ MinerU installed properly' if rag.check_parser_installation() else '❌ MinerU installation issue')"

jupyter:
	docker-compose exec app jupyter lab --allow-root --ip 0.0.0.0

server:
	open http://localhost:9621

down:
	docker-compose down --volumes
	docker system prune -f

down-local:
	docker-compose --profile local down --volumes
	docker system prune -f

# Dockerを使って依存関係を再計算し、ホスト側の requirements.txt を更新したい場合
# (基本はビルド内で完結しますが、手元に書き出しておきたい場合に便利です)
update-requirements:
	docker run --rm -v $(PWD):/app ghcr.io/astral-sh/uv:python3.12-bookworm-slim \
		sh -c "cd /app && uv pip compile requirements.in -o requirements.txt --upgrade"

# ビルドして起動
build-force:
	docker-compose build --no-cache
	@make up

clean:
	@make down
	rm -rf sample/ neo4j/*
	docker-compose down --rmi all --volumes --remove-orphans

clean-local:
	@make down-local
	rm -rf sample/ neo4j/*
	docker-compose --profile local down --rmi all --volumes --remove-orphans

allclean:
	@make clean
	rm lightrag.log
	rm -rf data/
	rm -rf visualize/
	rm -rf dickens/
	rm -rf lib/
	rm -rf neo4j/
