use sqlx::postgres::PgPoolOptions;
use std::error::Error;
use std::path::Path;
use walkdir::WalkDir;
use serde_json::json;
use ollama_rs::{Ollama, generation::embeddings::request::GenerateEmbeddingsRequest};

// Helper function to format a vector for SQL insertion
fn format_vector(vector: &[f32]) -> String {
    format!("[{}]", vector.iter().map(|f| f.to_string()).collect::<Vec<_>>().join(","))
}

// Helper function to get language from file extension
fn get_language(path: &Path) -> &str {
    match path.extension().and_then(|s| s.to_str()) {
        Some("rs") => "rust",
        Some("py") => "python",
        Some("js") => "javascript",
        Some("ts") => "typescript",
        Some("go") => "go",
        Some("java") => "java",
        Some("c") => "c",
        Some("cpp") => "cpp",
        Some("h") => "c++",
        Some("md") => "markdown",
        Some("toml") => "toml",
        Some("json") => "json",
        Some("sql") => "sql",
        _ => "text",
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // 1. Load the project's codebase (excluding the /target/ folder)
    let documents = load_documents().await?;
    println!("Loaded {} documents.", documents.len());

    // 2. Initialize the Ollama client for embeddings
    let ollama = Ollama::new("http://localhost:11434".to_string(), 11434);
    println!("Ollama client initialized.");

    // 3. Generate embeddings for the documents
    let embeddings = generate_embeddings(&ollama, &documents).await?;
    println!("Generated {} embeddings.", embeddings.len());

    // 4. Initialize the database connection pool
    let database_url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(database_url)
        .await?;
    println!("Database pool initialized.");

    // 5. Store the embeddings in the TimescaleDB database
    store_embeddings(&pool, &embeddings).await?;
    println!("Successfully stored embeddings in the database.");

    Ok(())
}

async fn load_documents() -> Result<Vec<(String, String)>, Box<dyn Error>> {
    let mut documents = Vec::new();
    // Choose the filetypes to ignore during ingestion to reduce query noise
    let ignored_files: Vec<&str> = vec![".gitignore", "Cargo.lock", "yarn.lock", "package-lock.json", "debug_log.txt", "Cargo.toml", "Dockerfile", ".env"];
    let ignored_dirs: Vec<&str> = vec!["/target/", "/.git/", "/venv/", "/__pycache__/", "/.sqlx/"];

    for entry in WalkDir::new("..")
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| {
            let path_str = e.path().to_string_lossy();
            // Check if any part of the path contains an ignored directory
            if ignored_dirs.iter().any(|dir| path_str.contains(dir)) {
                return false;
            }
            // Check if the file name itself is in the ignored_files list
            if let Some(file_name) = e.path().file_name().and_then(|n| n.to_str()) {
                if ignored_files.contains(&file_name) {
                    return false;
                }
            }
            true
        })
        .filter(|e| e.file_type().is_file())
    {
        let path = entry.path();
        let path_str = path.to_string_lossy().to_string();
        if let Ok(content) = tokio::fs::read_to_string(path).await {
            // Filter out specific auto-generated or boilerplate code that adds noise but little
            // semantic value for RAG. Users should customize these filters based on their project's
            // specific needs to improve context quality and reduce token count.
            if path.extension().and_then(|s| s.to_str()) == Some("rs") && 
               (content.contains("/// This module was auto-generated with ethers-rs Abigen.") || 
                content.contains("pub struct OnnxModels {")) {
                continue;
            }
            documents.push((path_str, content));
        } else {
            // If reading as UTF-8 fails, it's likely a binary file, so skip it.
            continue;
        }
    }
    Ok(documents)
}

async fn generate_embeddings(ollama: &Ollama, documents: &[(String, String)]) -> Result<Vec<(String, String, Vec<f32>)>, Box<dyn Error>> {
    let mut embeddings = Vec::new();
    for (path, content) in documents {
        let request = GenerateEmbeddingsRequest::new(
            "dengcao/Qwen3-Embedding-4B:Q4_K_M".to_string(),
            ollama_rs::generation::embeddings::request::EmbeddingsInput::Single(content.clone()),
        );

        match ollama.generate_embeddings(request).await {
            Ok(response) => {
                if let Some(embedding) = response.embeddings.into_iter().next() {
                    embeddings.push((path.clone(), content.clone(), embedding));
                }
            },
            Err(e) => {
                eprintln!("Failed to generate embedding for {}: {}", path, e);
            }
        }
    }
    Ok(embeddings)
}

async fn store_embeddings(pool: &sqlx::PgPool, embeddings: &[(String, String, Vec<f32>)]) -> Result<(), Box<dyn Error>> {
    for (path, content, vector) in embeddings {
        let metadata = json!({
            "source": "codebase",
            "language": get_language(Path::new(path)),
            "path": path,
        });
        let vector_str = format_vector(vector);

        // Use INSERT ON CONFLICT to update existing entries
        sqlx::query(
            r#"
            INSERT INTO embeddings (id, text, vector, metadata)
            VALUES ($1, $2, $3::vector, $4)
            ON CONFLICT (id) DO UPDATE
            SET text = EXCLUDED.text,
                vector = EXCLUDED.vector,
                metadata = EXCLUDED.metadata;
            "#,
        )
        .bind(path)
        .bind(content)
        .bind(vector_str)
        .bind(metadata)
        .execute(pool)
        .await?;
    }
    Ok(())
}