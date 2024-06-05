# Setup

## Install Milvus in Docker

Milvus provides an installation script to install it as a docker container. The script is available in the [Milvus repository](https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh). To install Milvus in Docker, just run

````bash
bash standalone_embed.sh start
````

## Install Chroma in Docker

````bash
docker run -d --name chromadb -p 8000:8000 -v "$PWD"/volumes/chroma/data:/chroma/chroma -e IS_PERSISTENT=TRUE -e ANONYMIZED_TELEMETRY=TRUE -e ALLOW_RESET=TRUE chromadb/chroma:latest
````

## Install Redis in Docker

````bash
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 -v "$PWD"/volumes/redis/data:/data redis/redis-stack:latest
````

## Install pgvector in Docker

````bash
docker volume create pgvector-data
docker run --name pgvector -e POSTGRES_PASSWORD=pwd -p 5432:5432 -v pgvector-data:/var/lib/postgresql/data -d pgvector/pgvector:pg16
 ````