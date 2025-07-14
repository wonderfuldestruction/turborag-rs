use clap::Parser;
use ollama_rs::generation::completion::request::GenerationRequest;
use ollama_rs::Ollama;
use sqlx::postgres::PgPoolOptions;
use std::error::Error;

/// A simple CLI to query and rerank documents from a pgvector database.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The query to search for
    #[arg(short, long)]
    query: String,

    /// The number of initial documents to retrieve
    #[arg(short, long, default_value_t = 25)]
    limit: i32,

    /// The number of final documents to return after reranking
    #[arg(short, long, default_value_t = 5)]
    top_n: usize,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    // --- 1. Initialize Clients ---
    let ollama = Ollama::new("http://localhost:11434".to_string(), 11434);
    let database_url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(database_url)
        .await?;

    // --- 2. Generate Embedding for the User Query ---
    println!("Generating embedding for query...");
    let query_embedding_request = ollama_rs::generation::embeddings::request::GenerateEmbeddingsRequest::new(
        "dengcao/Qwen3-Embedding-4B:Q4_K_M".to_string(),
        ollama_rs::generation::embeddings::request::EmbeddingsInput::Single(args.query.clone()),
    );
    let query_embedding_response = ollama.generate_embeddings(query_embedding_request).await?;
    let query_vector = query_embedding_response.embeddings.into_iter().next().ok_or("Failed to get query embedding")?;
    let query_vector_str = format!("[{}]", query_vector.iter().map(|f| f.to_string()).collect::<Vec<_>>().join(","));

    // --- 3. Initial Retrieval from Database ---
    println!("Retrieving initial documents from database...");
    let retrieved_docs: Vec<(String, String)> = sqlx::query_as(
        r#"
        SELECT id, text
        FROM embeddings
        ORDER BY vector <=> $1::vector
        LIMIT $2;
        "#,
    )
    .bind(query_vector_str)
    .bind(args.limit)
    .fetch_all(&pool)
    .await?;

    println!("Retrieved {} documents for reranking...", retrieved_docs.len());

    // --- 4. Rerank the Retrieved Documents ---
    let mut reranked_docs = Vec::new();
    for (id, document_text) in retrieved_docs {
        let rerank_prompt = format!(
            "Given the query: '{}' and the document: '{}'. Output only a single floating-point number between 0.0 and 1.0 representing the relevance score. No other text, explanation, or formatting.",
            args.query,
            document_text
        );

        let rerank_request = GenerationRequest::new(
            "hf.co/mradermacher/Qwen3-Reranker-4B-GGUF:Q4_K_M".to_string(),
            rerank_prompt,
        );

        let response = ollama.generate(rerank_request).await?;
        let last_line = response.response.trim().lines().last().unwrap_or("");
        if let Ok(score) = last_line.parse::<f32>() {
            reranked_docs.push((id, document_text, score));
        } else {
            eprintln!("Warning: Could not parse rerank score from line '{}' for document {}", last_line, id);
        }
    }

    // Sort by the new relevance score in descending order
    reranked_docs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    // --- 5. Display Final Results ---
    println!("\n--- Top {} Reranked Results ---", args.top_n);
    for (i, (id, text, score)) in reranked_docs.iter().take(args.top_n).enumerate() {
        println!("\n{}. ID: {} (Score: {:.4})", i + 1, id, score);
        println!("--------------------------------------------------");
        println!("{}", text.chars().take(500).collect::<String>());
        if text.len() > 500 {
            println!("... (truncated)");
        }
    }

    Ok(())
}
