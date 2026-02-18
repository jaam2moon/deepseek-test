use reqwest::Client;
use tracing::info;

use crate::models::{DeepSeekMessage, DeepSeekRequest, DeepSeekResponse, Pattern};

const DEEPSEEK_URL: &str = "https://api.deepseek.com/chat/completions";

// DeepSeek Reasoner pricing (per million tokens)
const REASONER_INPUT_PRICE: f64 = 0.55;      // $0.55/M input tokens (cache miss)
const REASONER_INPUT_CACHE_PRICE: f64 = 0.14; // $0.14/M input tokens (cache hit)
const REASONER_OUTPUT_PRICE: f64 = 2.19;      // $2.19/M output tokens
const REASONER_REASONING_PRICE: f64 = 2.19;   // reasoning tokens priced as output

pub struct AnalyzerResult {
    pub pattern: String,
    pub category: String,
    pub direction: String,
    pub confidence: String,
    pub reasoning: String,
    pub chain_of_thought: Option<String>,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub reasoning_tokens: u64,
    pub cache_hit_tokens: u64,
    pub cost_usd: f64,
}

fn build_system_prompt(patterns: &[Pattern]) -> String {
    let mut prompt = String::from(
        "You are an expert candlestick pattern analyst. Given a text description of a candlestick chart, \
         identify which pattern it most closely matches from the taxonomy below.\n\n\
         PATTERN TAXONOMY:\n",
    );

    for p in patterns {
        prompt.push_str(&format!(
            "- {} | Category: {} | Direction: {} | {}\n",
            p.name, p.category, p.direction, p.description
        ));
    }

    prompt.push_str(
        "\nINSTRUCTIONS:\n\
         1. Carefully analyze the chart description\n\
         2. Compare against all 71 patterns in the taxonomy\n\
         3. Identify the best matching pattern\n\
         4. If no pattern matches well, say \"No Clear Pattern\" with explanation\n\n\
         Respond with ONLY a JSON object (no markdown, no code fences) in this exact format:\n\
         {\"pattern\": \"<pattern name>\", \"category\": \"<Single/Two/Three/Multi/Continuation/Special>\", \
         \"direction\": \"<Bullish/Bearish/Neutral>\", \"confidence\": \"<High/Medium/Low>\", \
         \"reasoning\": \"<brief explanation of why this pattern matches>\"}\n",
    );

    prompt
}

pub async fn analyze_pattern(
    client: &Client,
    api_key: &str,
    chart_description: &str,
    patterns: &[Pattern],
) -> Result<AnalyzerResult, String> {
    let system_prompt = build_system_prompt(patterns);

    let request = DeepSeekRequest {
        model: "deepseek-reasoner".to_string(),
        messages: vec![
            DeepSeekMessage {
                role: "system".to_string(),
                content: system_prompt,
            },
            DeepSeekMessage {
                role: "user".to_string(),
                content: format!(
                    "Analyze this candlestick chart description and identify the pattern:\n\n{}",
                    chart_description
                ),
            },
        ],
        stream: false,
    };

    info!("Sending chart description to DeepSeek Reasoner...");

    let resp = client
        .post(DEEPSEEK_URL)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await
        .map_err(|e| format!("DeepSeek request failed: {}", e))?;

    let status = resp.status();
    let body = resp
        .text()
        .await
        .map_err(|e| format!("Failed to read DeepSeek response: {}", e))?;

    if !status.is_success() {
        return Err(format!("DeepSeek API error ({}): {}", status, body));
    }

    let ds_resp: DeepSeekResponse = serde_json::from_str(&body)
        .map_err(|e| format!("Failed to parse DeepSeek response: {} — body: {}", e, body))?;

    let choice = ds_resp
        .choices
        .first()
        .ok_or("DeepSeek returned no choices")?;

    let content = &choice.message.content;
    let chain_of_thought = choice.message.reasoning_content.clone();

    // Parse token usage
    let (prompt_tokens, completion_tokens, reasoning_tokens, cache_hit_tokens) =
        if let Some(usage) = &ds_resp.usage {
            (
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.reasoning_tokens,
                usage.prompt_cache_hit_tokens,
            )
        } else {
            (0, 0, 0, 0)
        };

    // Calculate cost
    let cache_miss_tokens = prompt_tokens.saturating_sub(cache_hit_tokens);
    let cost_usd = (cache_miss_tokens as f64 / 1_000_000.0) * REASONER_INPUT_PRICE
        + (cache_hit_tokens as f64 / 1_000_000.0) * REASONER_INPUT_CACHE_PRICE
        + (completion_tokens as f64 / 1_000_000.0) * REASONER_OUTPUT_PRICE
        + (reasoning_tokens as f64 / 1_000_000.0) * REASONER_REASONING_PRICE;

    info!(
        "DeepSeek usage: {} prompt ({} cached), {} completion, {} reasoning — ${:.6}",
        prompt_tokens, cache_hit_tokens, completion_tokens, reasoning_tokens, cost_usd
    );

    // Parse JSON from content (strip markdown fences if present)
    let json_str = content
        .trim()
        .strip_prefix("```json")
        .unwrap_or(content.trim())
        .strip_prefix("```")
        .unwrap_or(content.trim())
        .strip_suffix("```")
        .unwrap_or(content.trim())
        .trim();

    let parsed: serde_json::Value = serde_json::from_str(json_str).map_err(|e| {
        format!(
            "Failed to parse pattern JSON from DeepSeek: {} — content: {}",
            e, content
        )
    })?;

    Ok(AnalyzerResult {
        pattern: parsed["pattern"]
            .as_str()
            .unwrap_or("Unknown")
            .to_string(),
        category: parsed["category"]
            .as_str()
            .unwrap_or("Unknown")
            .to_string(),
        direction: parsed["direction"]
            .as_str()
            .unwrap_or("Unknown")
            .to_string(),
        confidence: parsed["confidence"]
            .as_str()
            .unwrap_or("Unknown")
            .to_string(),
        reasoning: parsed["reasoning"]
            .as_str()
            .unwrap_or("No reasoning provided")
            .to_string(),
        chain_of_thought,
        prompt_tokens,
        completion_tokens,
        reasoning_tokens,
        cache_hit_tokens,
        cost_usd,
    })
}
