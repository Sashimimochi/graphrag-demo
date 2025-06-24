setup:
	mkdir -p data
	curl https://raw.githubusercontent.com/gusye1234/nano-graphrag/main/tests/mock_data.txt > ./data/dickens.txt
	@make up

up:
	docker-compose up -d
	open http://localhost:8501
	open http://localhost:9621

jupyter:
	docker-compose exec app jupyter lab --allow-root --ip 0.0.0.0

server:
	open http://localhost:9621

down:
	docker-compose down --volumes
	docker system prune -f

clean:
	@make down
	rm -rf sample/ neo4j/*
	docker-compose down --rmi all --volumes --remove-orphans

allclean:
	@make clean
	rm lightrag.log
	rm -rf data/
	rm -rf visualize/
	rm -rf dickens/
	rm -rf lib/
	rm -rf neo4j/
