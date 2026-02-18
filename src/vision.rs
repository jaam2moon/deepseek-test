use reqwest::Client;
use serde::Deserialize;
use tracing::{info, warn};

use crate::models::{ReplicateRequest, ReplicateInput, ReplicateResponse};

const REPLICATE_URL: &str = "https://api.replicate.com/v1/predictions";
const REPLICATE_UPLOAD_URL: &str = "https://api.replicate.com/v1/files";
pub const VL2_VERSION: &str =
    "e5caf557dd9e5dcee46442e1315291ef1867f027991ede8ff95e304d4f734200";

const VISION_PROMPT: &str = "\
Describe this candlestick chart <image> in detail. Focus on:
- Number of candles visible
- Body colors (red/green) of each candle in order
- Relative body sizes (large, medium, small, doji)
- Wick/shadow lengths (long upper, long lower, short, none)
- Gaps between candles (gap up, gap down, overlapping)
- Overall trend direction before/during the pattern
- Any notable features (engulfing, inside bars, identical highs/lows)

Be precise and systematic. Describe each candle from left to right.";

pub struct VisionResult {
    pub description: String,
    pub predict_seconds: f64,
}

#[derive(Debug, Deserialize)]
struct FileUploadResponse {
    urls: FileUploadUrls,
}

#[derive(Debug, Deserialize)]
struct FileUploadUrls {
    get: String,
}

async fn upload_image(
    client: &Client,
    token: &str,
    image_bytes: &[u8],
    content_type: &str,
) -> Result<String, String> {
    info!("Uploading image to Replicate file storage...");

    let part = reqwest::multipart::Part::bytes(image_bytes.to_vec())
        .file_name("chart.png")
        .mime_str(content_type)
        .map_err(|e| format!("Failed to create multipart: {}", e))?;

    let form = reqwest::multipart::Form::new()
        .part("content", part);

    let resp = client
        .post(REPLICATE_UPLOAD_URL)
        .header("Authorization", format!("Bearer {}", token))
        .multipart(form)
        .send()
        .await
        .map_err(|e| format!("File upload request failed: {}", e))?;

    let status = resp.status();
    let body = resp
        .text()
        .await
        .map_err(|e| format!("Failed to read upload response: {}", e))?;

    if !status.is_success() {
        return Err(format!("File upload failed ({}): {}", status, body));
    }

    let upload_resp: FileUploadResponse = serde_json::from_str(&body)
        .map_err(|e| format!("Failed to parse upload response: {} — body: {}", e, body))?;

    info!("Image uploaded: {}", upload_resp.urls.get);
    Ok(upload_resp.urls.get)
}

pub async fn describe_chart(
    client: &Client,
    replicate_token: &str,
    image_bytes: &[u8],
    content_type: &str,
) -> Result<VisionResult, String> {
    let image_url = upload_image(client, replicate_token, image_bytes, content_type).await?;

    let request = ReplicateRequest {
        version: VL2_VERSION.to_string(),
        input: ReplicateInput {
            image: image_url,
            prompt: VISION_PROMPT.to_string(),
            temperature: 0.1,
            top_p: 0.9,
            max_length_tokens: 2048,
            repetition_penalty: 1.1,
        },
    };

    info!("Sending image to Replicate DeepSeek-VL2...");

    let resp = client
        .post(REPLICATE_URL)
        .header("Authorization", format!("Bearer {}", replicate_token))
        .header("Prefer", "wait")
        .json(&request)
        .send()
        .await
        .map_err(|e| format!("Replicate request failed: {}", e))?;

    let status = resp.status();
    let body = resp
        .text()
        .await
        .map_err(|e| format!("Failed to read Replicate response: {}", e))?;

    if !status.is_success() && !status.is_redirection() {
        return Err(format!("Replicate API error ({}): {}", status, body));
    }

    let prediction: ReplicateResponse =
        serde_json::from_str(&body).map_err(|e| format!("Failed to parse Replicate response: {} — body: {}", e, body))?;

    if let Some(err) = prediction.error {
        return Err(format!("Replicate prediction error: {}", err));
    }

    match prediction.status.as_str() {
        "succeeded" => extract_result(&prediction),
        "processing" | "starting" => {
            info!("Prediction still running ({}), polling...", prediction.status);
            poll_prediction(client, replicate_token, &prediction.id).await
        }
        other => Err(format!("Unexpected prediction status: {}", other)),
    }
}

async fn poll_prediction(
    client: &Client,
    token: &str,
    prediction_id: &str,
) -> Result<VisionResult, String> {
    let url = format!(
        "https://api.replicate.com/v1/predictions/{}",
        prediction_id
    );

    for attempt in 1..=100 {
        tokio::time::sleep(std::time::Duration::from_secs(3)).await;

        let resp = client
            .get(&url)
            .header("Authorization", format!("Bearer {}", token))
            .send()
            .await
            .map_err(|e| format!("Poll request failed: {}", e))?;

        let prediction: ReplicateResponse = resp
            .json()
            .await
            .map_err(|e| format!("Failed to parse poll response: {}", e))?;

        if let Some(err) = &prediction.error {
            return Err(format!("Prediction failed: {}", err));
        }

        match prediction.status.as_str() {
            "succeeded" => return extract_result(&prediction),
            "failed" | "canceled" => {
                return Err(format!("Prediction {}: {:?}", prediction.status, prediction.error))
            }
            _ => {
                if attempt % 10 == 0 {
                    warn!("Still waiting for prediction (attempt {}/100)...", attempt);
                }
            }
        }
    }

    Err("Prediction timed out after 5 minutes".to_string())
}

fn extract_result(prediction: &ReplicateResponse) -> Result<VisionResult, String> {
    let description = match &prediction.output {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Array(arr)) => {
            let text: String = arr
                .iter()
                .filter_map(|v| v.as_str())
                .collect::<Vec<_>>()
                .join("");
            if text.is_empty() {
                return Err("Replicate returned empty output array".to_string());
            }
            text
        }
        Some(other) => other.to_string(),
        None => return Err("Replicate returned no output".to_string()),
    };

    let predict_seconds = prediction
        .metrics
        .as_ref()
        .and_then(|m| m.predict_time)
        .unwrap_or(0.0);

    Ok(VisionResult {
        description,
        predict_seconds,
    })
}
