use std::env;

pub struct Config {
    pub deepseek_api_key: String,
    pub replicate_api_token: String,
    pub port: u16,
}

impl Config {
    pub fn from_env() -> Self {
        Self {
            deepseek_api_key: env::var("DEEPSEEK_API_KEY")
                .expect("DEEPSEEK_API_KEY must be set"),
            replicate_api_token: env::var("REPLICATE_API_TOKEN")
                .expect("REPLICATE_API_TOKEN must be set"),
            port: env::var("PORT")
                .unwrap_or_else(|_| "3000".to_string())
                .parse()
                .expect("PORT must be a valid u16"),
        }
    }
}
