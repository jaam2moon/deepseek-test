# Deepseek candle chart image reader
* HTTP Server using Axum (Rust) wrap with a single Dockerfile (not docker-compose) that can deploy to Railway via github repo (aleray init)
* Simple Web UI for human interaction in the same server with backend
* Use Deepseek LLM (latest docs: /Users/bhubadiinn/dev/koom/deepseek-test/docs) to read image and determine its pattern (pattern examples in: ) as answer
* Stateless app, no need for db yet.