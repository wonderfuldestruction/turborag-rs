# turborag-rs

A Rust-based Retrieval-Augmented Generation (RAG) system for codebases, designed to provide relevant context to Large Language Models (LLMs).

## Overview

`turborag-rs` processes your codebase, generates embeddings, and stores them in a vector database. It then retrieves and reranks code snippets based on natural language queries, making it ideal for agentic workflows and enhancing LLM understanding of your project.

Key features include:

*   **Codebase Ingestion**: Scans Rust code (and other specified file types), filters irrelevant files, generates embeddings, and stores them in PostgreSQL with `pgvector`.
*   **Natural Language Querying**: Transforms natural language queries into embeddings, performs vector similarity searches, and reranks results for highly relevant code context.
*   **Ollama Integration**: Seamlessly integrates with Ollama for local inference of embedding and reranker models.
*   **TimescaleDB/pgvector**: Leverages PostgreSQL with the `pgvector` extension for efficient vector storage and search.

## Getting Started

### Prerequisites

Ensure you have the following installed:

*   [**Rust and Cargo**](https://rustup.rs/): The Rust programming language and its package manager.
*   [**Docker and Docker Compose**](https://docs.docker.com/get-docker/): For easy setup of PostgreSQL/TimescaleDB and Ollama.
*   [**Ollama**](https://ollama.ai/): A tool for running large language models locally.

### Setup

1.  **Clone the Repository**

    ```bash
git clone <your-repo-url>
cd turborag-rs
    ```

2.  **Database Setup (PostgreSQL with pgvector)**

    Create a `docker-compose.yml` file in your project root (or a dedicated `db` folder) with the following content. Replace `your_user`, `your_password`, and `your_database` with your desired credentials.

    ```yaml
version: '3.8'
services:
  db:
    image: "ankane/pgvector:latest"
    restart: always
    environment:
      POSTGRES_USER: your_user
      POSTGRES_PASSWORD: your_password
      POSTGRES_DB: your_database
    ports:
      - "5432:5432"
    volumes:
      - db_data:/var/lib/postgresql/data

volumes:
  db_data:
    ```

    Start the database:

    ```bash
docker-compose up -d db
    ```

    Once the database is running, connect to it (e.g., using `psql`) and enable the `vector` extension and create the `embeddings` table:

    ```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS embeddings (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    vector VECTOR(2560), -- Adjust dimension based on your embedding model
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
    ```

3.  **Ollama Model Setup**

    Ensure Ollama is running. Then, pull the recommended models. These models are optimized for performance and context length (32k tokens).

    ```bash
ollama run dengcao/Qwen3-Embedding-4B:Q4_K_M
ollama run hf.co/mradermacher/Qwen3-Reranker-4B-GGUF:Q4_K_M
    ```

    *Note: Both models are Q4_K_M quantized and require at least 2.5GB VRAM each for an 8k token base context window. They run concurrently during query. If VRAM is a constraint, the embedding model can serve as a reranker. GPU inference is recommended for best results; CPU users might consider `Qwen3 0.6B Embedding @ Q4_K_M` (484MB) for 32k context.*

4.  **Configure Environment Variable**

    Set the `DATABASE_URL` environment variable to match your PostgreSQL connection details:

    ```bash
export DATABASE_URL="postgres://your_user:your_password@localhost:5432/your_database"
    ```

5.  **Build the Project**

    ```bash
cargo build --release
    ```

## Usage

1.  **Ingest Codebase**

    Run the `rag-system` binary to scan your codebase, generate embeddings, and store them in the database. This process can take time depending on your codebase size.

    ```bash
cargo run --release --bin rag-system
    ```

2.  **Query the Codebase**

    Use the `query` binary to ask natural language questions about your codebase.

    ```bash
cargo run --release --bin query -- --query "Explain the core functionalities of the ABC and XYZ modules."
    ```

    Configurable in `src/bin/query.rs`, you can also manually adjust the `--limit` (initial documents to retrieve) and `--top-n` (final reranked documents to display) parameters:

    ```bash
cargo run --release --bin query -- --query "How do I handle errors in the API module?" --limit 50 --top-n 10
    ```

## Benchmarks

`turborag-rs` is designed for medium-sized projects (e.g., codebases exceeding 750k tokens, with scripts up to 150 lines, and mixed YAML/Markdown documentation).

**Test Environment:**
*   **OS:** Ubuntu 24.04.02 LTS
*   **CPU:** Ryzen 9950X
*   **GPU:** RTX5090 32GB
*   **RAM:** 96GB DDR5 6000
*   **Storage:** Samsung 990 Pro SSD

**Performance Highlights:**

*   **Ingestion:** RAG ingestion of 107 documents took under 10 seconds. Subsequent ingests are faster for unchanged documents.
*   **Query Performance (Token Reduction & Speed):**
    *   **Scenario 1 (Smaller Context):** Querying a module (~5,000 tokens).
        *   Without RAG: ~5,000 tokens, 23s
        *   With RAG: ~1,250 tokens, 28s (61% token reduction, 22% longer query time due to cold start/Ollama latency)
    *   **Scenario 2 (Larger Context):** Querying two interconnected modules (~18,000 tokens).
        *   Without RAG: ~18,000 tokens, 42s
        *   With RAG: 1,476 tokens, 30s (91% token reduction, 29% faster query time)

*Note: Benchmarks were conducted using Gemini 2.5 Flash via Gemini CLI (free tier, subject to latency variations). Qwen3 models were cold-started during queries. Both Qwen3 models accurately retrieved information on the first attempt, aligning with their 99% zero-shot performance on the MTEB leaderboard. Peak VRAM consumption during inference was approximately 10.3GB.*

## Configuration

*   **`DATABASE_URL`**: Environment variable for PostgreSQL connection.
*   **Embedding Model**: Configured in `src/main.rs` and `src/bin/query.rs`.
*   **Reranker Model**: Configured in `src/bin/query.rs`.
*   **Ignored Directories/Files**: Defined in `src/main.rs` within the `load_documents` function.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License.
