use serde::{Deserialize, Serialize};

// --- Domain types ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    pub name: String,
    pub category: String,
    pub direction: String,
    pub description: String,
}

#[derive(Debug, Serialize)]
pub struct AnalyzeResponse {
    pub pattern: String,
    pub category: String,
    pub direction: String,
    pub confidence: String,
    pub reasoning: String,
    pub chain_of_thought: Option<String>,
    pub chart_description: String,
    pub cost: CostBreakdown,
}

#[derive(Debug, Serialize)]
pub struct CostBreakdown {
    pub vision_seconds: f64,
    pub vision_cost_usd: f64,
    pub reasoner_prompt_tokens: u64,
    pub reasoner_completion_tokens: u64,
    pub reasoner_reasoning_tokens: u64,
    pub reasoner_cost_usd: f64,
    pub total_cost_usd: f64,
}

// --- Replicate API types ---

#[derive(Debug, Serialize)]
pub struct ReplicateRequest {
    pub version: String,
    pub input: ReplicateInput,
}

#[derive(Debug, Serialize)]
pub struct ReplicateInput {
    pub image: String,
    pub prompt: String,
    pub temperature: f64,
    pub top_p: f64,
    pub max_length_tokens: u32,
    pub repetition_penalty: f64,
}

#[derive(Debug, Deserialize)]
pub struct ReplicateResponse {
    pub id: String,
    pub status: String,
    pub output: Option<serde_json::Value>,
    pub error: Option<String>,
    pub metrics: Option<ReplicateMetrics>,
}

#[derive(Debug, Deserialize)]
pub struct ReplicateMetrics {
    pub predict_time: Option<f64>,
}

// --- DeepSeek API types ---

#[derive(Debug, Serialize)]
pub struct DeepSeekRequest {
    pub model: String,
    pub messages: Vec<DeepSeekMessage>,
    pub stream: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DeepSeekMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct DeepSeekResponse {
    pub choices: Vec<DeepSeekChoice>,
    pub usage: Option<DeepSeekUsage>,
}

#[derive(Debug, Deserialize)]
pub struct DeepSeekChoice {
    pub message: DeepSeekResponseMessage,
}

#[derive(Debug, Deserialize)]
pub struct DeepSeekResponseMessage {
    #[allow(dead_code)]
    pub role: String,
    pub content: String,
    pub reasoning_content: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct DeepSeekUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    #[serde(default)]
    pub reasoning_tokens: u64,
    #[serde(default)]
    pub prompt_cache_hit_tokens: u64,
}
