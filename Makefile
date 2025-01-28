setup:
	mkdir -p data
	curl https://raw.githubusercontent.com/gusye1234/nano-graphrag/main/tests/mock_data.txt > ./data/dickens.txt
	@make up

up:
	docker-compose up -d

down:
	docker-compose down --volumes
	rm -rf sample/ neo4j/*

clean:
	docker-compose down --rmi all --volumes --remove-orphans
