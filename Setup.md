# Setup

## Install Milvus in Docker

milvus-standalone
````bash
bash milvus.sh start
````
milvus docker-compose
````bash
docker commpose up -d
````

## Install Chroma in Docker

````bash
docker run -d --name chromadb -p 8000:8000 -v "$PWD"/volumes/chroma/data:/chroma/chroma -e IS_PERSISTENT=TRUE -e ANONYMIZED_TELEMETRY=TRUE -e ALLOW_RESET=TRUE chromadb/chroma:0.5.0
````
Version of chroma docker image and chromadb python package must be the same.


## Install Redis in Docker

````bash
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 -v "$PWD"/volumes/redis/data:/data redis/redis-stack:latest
````

## Install pgvector in Docker

````bash
docker volume create pgvector-data
docker run --name pgvector -e POSTGRES_PASSWORD=pwd -p 5432:5432 -v pgvector-data:/var/lib/postgresql/data -d pgvector/pgvector:pg16
 ````