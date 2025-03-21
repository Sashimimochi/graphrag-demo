setup:
	mkdir -p data
	curl https://raw.githubusercontent.com/gusye1234/nano-graphrag/main/tests/mock_data.txt > ./data/dickens.txt
	@make up

up:
	docker-compose up -d
	open http://localhost:8501

down:
	docker-compose down --volumes
	rm -rf sample/ neo4j/*
	docker system prune -f

clean:
	@make down
	docker-compose down --rmi all --volumes --remove-orphans

allclean:
	@make clean
	rm lightrag.log
	rm -rf data/
	rm -rf visualize/
	rm -rf dickens/
	rm -rf lib/
	rm -rf neo4j/
