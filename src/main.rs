mod analyzer;
mod config;
mod models;
mod vision;

use axum::{
    extract::{Multipart, State},
    http::StatusCode,
    response::{Html, IntoResponse, Json},
    routing::{get, post},
    Router,
};
use config::Config;
use models::{AnalyzeResponse, CostBreakdown, Pattern};
use reqwest::Client;
use serde::Serialize;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_http::services::ServeDir;
use tracing::{error, info, warn};

// Replicate DeepSeek-VL2 pricing: Nvidia A100 80GB @ $0.001400/sec
const REPLICATE_GPU_RATE: f64 = 0.001400;

#[derive(Clone, Serialize)]
pub struct WarmupStatus {
    pub state: String,       // "starting" | "warming" | "ready" | "failed"
    pub message: String,
    pub elapsed_secs: u64,
}

struct AppState {
    config: Config,
    client: Client,
    patterns: Vec<Pattern>,
    warmup: RwLock<WarmupStatus>,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let config = Config::from_env();
    let port = config.port;

    let patterns = load_patterns("candlestick_patterns.csv");
    info!("Loaded {} candlestick patterns", patterns.len());

    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()
        .expect("Failed to create HTTP client");

    let state = Arc::new(AppState {
        config,
        client,
        patterns,
        warmup: RwLock::new(WarmupStatus {
            state: "starting".to_string(),
            message: "server starting...".to_string(),
            elapsed_secs: 0,
        }),
    });

    // Spawn background warmup
    let warmup_state = state.clone();
    tokio::spawn(async move {
        run_warmup(warmup_state).await;
    });

    let app = Router::new()
        .route("/", get(index_handler))
        .route("/analyze", post(analyze_handler))
        .route("/patterns", get(patterns_handler))
        .route("/warmup", get(warmup_handler))
        .nest_service("/static", ServeDir::new("static"))
        .with_state(state);

    let addr = format!("0.0.0.0:{}", port);
    info!("Server starting on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("Failed to bind");

    axum::serve(listener, app).await.expect("Server failed");
}

async fn run_warmup(state: Arc<AppState>) {
    let start = std::time::Instant::now();

    // Update status: warming
    {
        let mut w = state.warmup.write().await;
        w.state = "warming".to_string();
        w.message = "sending warmup request to replicate...".to_string();
        w.elapsed_secs = 0;
    }
    info!("Warmup: sending dummy prediction to wake VL2 model...");

    // Send a minimal prediction to force Replicate to boot the model
    let request = serde_json::json!({
        "version": vision::VL2_VERSION,
        "input": {
            "image": "https://replicate.delivery/pbxt/MTtsBStHRqLDgNZMkt0J7PptoJ3lseSUNcGaDkG230ttNJlT/workflow.png",
            "prompt": "Say OK <image>",
            "max_length_tokens": 10
        }
    });

    let resp = state.client
        .post("https://api.replicate.com/v1/predictions")
        .header("Authorization", format!("Bearer {}", state.config.replicate_api_token))
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await;

    let prediction_id = match resp {
        Ok(r) => {
            let body: serde_json::Value = match r.json().await {
                Ok(v) => v,
                Err(e) => {
                    let mut w = state.warmup.write().await;
                    w.state = "failed".to_string();
                    w.message = format!("warmup parse error: {}", e);
                    error!("Warmup failed: {}", e);
                    return;
                }
            };

            if let Some(err) = body.get("detail").and_then(|v| v.as_str()) {
                let mut w = state.warmup.write().await;
                w.state = "failed".to_string();
                w.message = format!("replicate error: {}", err);
                error!("Warmup failed: {}", err);
                return;
            }

            let status = body.get("status").and_then(|v| v.as_str()).unwrap_or("");
            if status == "succeeded" {
                let mut w = state.warmup.write().await;
                w.state = "ready".to_string();
                w.message = "model ready".to_string();
                w.elapsed_secs = start.elapsed().as_secs();
                info!("Warmup: model already warm, ready in {}s", w.elapsed_secs);
                return;
            }

            match body.get("id").and_then(|v| v.as_str()) {
                Some(id) => id.to_string(),
                None => {
                    let mut w = state.warmup.write().await;
                    w.state = "failed".to_string();
                    w.message = format!("no prediction id: {}", body);
                    return;
                }
            }
        }
        Err(e) => {
            let mut w = state.warmup.write().await;
            w.state = "failed".to_string();
            w.message = format!("warmup request failed: {}", e);
            error!("Warmup request failed: {}", e);
            return;
        }
    };

    info!("Warmup: prediction {} created, polling...", prediction_id);

    // Poll until complete
    let poll_url = format!("https://api.replicate.com/v1/predictions/{}", prediction_id);

    for attempt in 1..=120 {
        tokio::time::sleep(std::time::Duration::from_secs(3)).await;
        let elapsed = start.elapsed().as_secs();

        {
            let mut w = state.warmup.write().await;
            w.elapsed_secs = elapsed;
            w.message = format!("warming up model... {}s", elapsed);
        }

        let resp = state.client
            .get(&poll_url)
            .header("Authorization", format!("Bearer {}", state.config.replicate_api_token))
            .send()
            .await;

        match resp {
            Ok(r) => {
                let body: serde_json::Value = match r.json().await {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                let status = body.get("status").and_then(|v| v.as_str()).unwrap_or("");

                match status {
                    "succeeded" => {
                        let mut w = state.warmup.write().await;
                        w.state = "ready".to_string();
                        w.message = format!("model ready ({}s)", elapsed);
                        w.elapsed_secs = elapsed;
                        info!("Warmup: model ready in {}s", elapsed);
                        return;
                    }
                    "failed" | "canceled" => {
                        let err_msg = body.get("error").and_then(|v| v.as_str()).unwrap_or("unknown");
                        let mut w = state.warmup.write().await;
                        w.state = "failed".to_string();
                        w.message = format!("warmup failed: {}", err_msg);
                        error!("Warmup prediction failed: {}", err_msg);
                        return;
                    }
                    _ => {
                        if attempt % 10 == 0 {
                            warn!("Warmup: still waiting ({}s, status: {})...", elapsed, status);
                        }
                    }
                }
            }
            Err(_) => continue,
        }
    }

    let mut w = state.warmup.write().await;
    w.state = "failed".to_string();
    w.message = "warmup timed out after 6 minutes".to_string();
    error!("Warmup timed out");
}

fn load_patterns(path: &str) -> Vec<Pattern> {
    let mut reader = csv::Reader::from_path(path).expect("Failed to open CSV");
    let mut patterns = Vec::new();

    for result in reader.records() {
        let record = result.expect("Failed to read CSV record");
        if record.len() >= 4 {
            patterns.push(Pattern {
                name: record[0].to_string(),
                category: record[1].to_string(),
                direction: record[2].to_string(),
                description: record[3].to_string(),
            });
        }
    }

    patterns
}

async fn index_handler() -> impl IntoResponse {
    let html = include_str!("../static/index.html");
    Html(html)
}

async fn warmup_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let w = state.warmup.read().await;
    Json(w.clone())
}

async fn patterns_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(state.patterns.clone())
}

async fn analyze_handler(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    // Check warmup status
    {
        let w = state.warmup.read().await;
        if w.state != "ready" {
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                format!("Model not ready: {}", w.message),
            ));
        }
    }

    let mut image_bytes: Option<Vec<u8>> = None;
    let mut content_type = "image/png".to_string();

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Multipart error: {}", e)))?
    {
        if field.name() == Some("image") {
            if let Some(ct) = field.content_type() {
                content_type = ct.to_string();
            }
            let bytes = field
                .bytes()
                .await
                .map_err(|e| (StatusCode::BAD_REQUEST, format!("Failed to read image: {}", e)))?;
            image_bytes = Some(bytes.to_vec());
        }
    }

    let image_bytes = image_bytes
        .ok_or((StatusCode::BAD_REQUEST, "No image field in request".to_string()))?;

    if image_bytes.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "Empty image".to_string()));
    }

    info!(
        "Received image: {} bytes, type: {}",
        image_bytes.len(),
        content_type
    );

    // Stage 1: Vision — get chart description
    let vision_result = vision::describe_chart(
        &state.client,
        &state.config.replicate_api_token,
        &image_bytes,
        &content_type,
    )
    .await
    .map_err(|e| {
        error!("Vision stage failed: {}", e);
        (StatusCode::INTERNAL_SERVER_ERROR, format!("Vision analysis failed: {}", e))
    })?;

    let vision_cost = vision_result.predict_seconds * REPLICATE_GPU_RATE;
    info!(
        "Vision: {:.1}s predict time — ${:.6}",
        vision_result.predict_seconds, vision_cost
    );
    info!(
        "Chart description: {}",
        &vision_result.description[..vision_result.description.len().min(200)]
    );

    // Stage 2: Pattern analysis
    let analysis = analyzer::analyze_pattern(
        &state.client,
        &state.config.deepseek_api_key,
        &vision_result.description,
        &state.patterns,
    )
    .await
    .map_err(|e| {
        error!("Analysis stage failed: {}", e);
        (StatusCode::INTERNAL_SERVER_ERROR, format!("Pattern analysis failed: {}", e))
    })?;

    let total_cost = vision_cost + analysis.cost_usd;
    info!("Total cost: ${:.6} (vision ${:.6} + reasoner ${:.6})", total_cost, vision_cost, analysis.cost_usd);

    let response = AnalyzeResponse {
        pattern: analysis.pattern,
        category: analysis.category,
        direction: analysis.direction,
        confidence: analysis.confidence,
        reasoning: analysis.reasoning,
        chain_of_thought: analysis.chain_of_thought,
        chart_description: vision_result.description,
        cost: CostBreakdown {
            vision_seconds: vision_result.predict_seconds,
            vision_cost_usd: vision_cost,
            reasoner_prompt_tokens: analysis.prompt_tokens,
            reasoner_completion_tokens: analysis.completion_tokens,
            reasoner_reasoning_tokens: analysis.reasoning_tokens,
            reasoner_cost_usd: analysis.cost_usd,
            total_cost_usd: total_cost,
        },
    };

    Ok(Json(response))
}
