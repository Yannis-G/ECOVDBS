# ECOVDBS - Experimental comparison of vector database systems

ECOVDBS (Experimental comparison of vector database systems) is a Python-based framework designed to benchmark and compare the performance of various vector database systems. This tool provides an end-to-end solution to run extensive tests, analyze the results, and visualize the performance of different systems using customizable scenarios and datasets.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Setup](#setup)
  - [Running Tests](#running-tests)
  - [Result Visualization](#result-visualization)
- [Supported Systems](#supported-systems)
- [Configuration](#configuration)
- [Contributing](#Contributing)

## Features
- **Multi-System Benchmarking:** Compare multiple vector databases (Chroma, Milvus, Redis, Pgvector) on the same dataset with consistent configurations.
- **Flexible Scenarios:** Define custom scenarios to test with pre-indexing and post-indexing setups.
- **Hyperparameter Testing:** Run queries with varying hyperparameters to understand the trade-offs between accuracy and query time.
- **Result Visualization:** Generate graphs to compare performance metrics across different systems and configurations.
- **Docker Integration:** Each database system runs in a Docker container to ensure a consistent and isolated environment.
- **Customizable Datasets:** Easily integrate and adapt various datasets for testing.

## Requirements
- Python 3.11 or higher
- Docker

## Installation
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Yannis-G/ECOVDBS.git
   cd ECOVDBS
   ```

2. **Install Python Dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set Up Docker:**
   Ensure Docker is installed and running on your system. Start the required Docker container for the database systems:
   ```bash
   docker run -d --name chromadb -p 8000:8000 -v "$PWD"/volumes/chroma/data:/chroma/chroma -e IS_PERSISTENT=TRUE -e ANONYMIZED_TELEMETRY=TRUE -e ALLOW_RESET=TRUE chromadb/chroma:0.5.0
   bash milvus.sh start
   docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 -v "$PWD"/volumes/redis/data:/data redis/redis-stack:7.2.0-v8
   docker volume create pgvector-data
   docker run --name pgvector -e POSTGRES_PASSWORD=pwd -p 5432:5432 -v pgvector-data:/var/lib/postgresql/data -d pgvector/pgvector:0.6.0-pg16 
   ```

## Usage
### Running Tests
1. **Initialize the Test Run:**
   Use the provided CLI tool to start a benchmark test. Example:
   ```bash
   python run.py --dataset sift_small --clients chroma milvus redis pgvector
   ```
   This command will run the pre-index scenario on the "Sift small" dataset using all four database systems. For more options, run `python run.py --help`.

2. **Monitor Execution:**
   All systems are tested sequentially to avoid resource contention. Progress is logged, and results are stored in the `results/` directory.

### Result Visualization
1. **Generate Graphs:**
   After completing all tests, generate visualizations to compare the results (automatically generated if run.py finished successfully):
   ```bash
   python plot_results.py
   ```
   This will produce graphs comparing metrics like recall, query time, and index size across all tested systems.

## Supported Systems
- **Chroma** (`chromadb/chroma:0.5.0`)
- **Milvus** (`milvusdb/milvus:v2.4.5`)
- **Redis** (`redis/redis-stack:7.2.0-v8`)
- **Pgvector** (`pgvector/pgvector:0.6.0-pg16`)

## Configuration
The behavior of ECOVDBS can be customized via configuration files.
- **`ecovdbs/config.py`:** Global configuration settings for folder paths.
- **`ecovdbs/client/[name]/[name]_config.py`:** Configuration for each database client.
- **`ecovdbs/runner/case_config.py`:** Configuration for the hyperparameters.

## Contributing
### Add new clients
To add a new client, create a new directory in `ecovdbs/client/` with the client's name. The directory should contain the following files:
- `__init__.py`
- `[name]_client.py`: Client implementation (should inherit from `BaseClient`)
- `[name]_config.py`: Client configuration (should inherit from `BaseConfig` and `BaseIndexConfig`)

Also, create a new directory in `ecovdbs/runner/` with the client's name. The directory should contain the following files:
- `__init__.py`
- `[name]_task.py`: Task implementation (should inherit from `HNSWTask`)

Finally, add the mapping from client name to the client task to the `client_mapper` variable in `ecovdbs/runner/utility.py`.

### Add new datasets
To add a new dataset, add two methods to `ecovdbs/dataset/dataset_reader.py`:
- `download_[dataset_name]`: Downloads the dataset from the internet and saves it to the file.
- `read_[dataset_name]`: Reads the dataset from the file and returns the data.

Also add the mapping from dataset name to the dataset reader to the `dataset_mapper` variable in `ecovdbs/dataset/dataset_reader.py`.
