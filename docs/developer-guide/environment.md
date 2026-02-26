---
title: Environment variables
---

# Environment variables

| Variable            | Description         | Used by                               |
| ------------------- | ------------------- | ------------------------------------- |
| `OPENAI_API_KEY`    | OpenAI API key      | OpenAIGateway, ConfigurableEmbeddings |
| `MISTRAL_API_KEY`   | Mistral API key     | MistralGateway                        |
| `COHERE_API_KEY`    | Cohere API key      | ConfigurableEmbeddings                |
| `HUGGINGFACE_TOKEN` | HuggingFace token   | ConfigurableEmbeddings (private models) |
| `POSTGRES_HOST`     | PostgreSQL host     | DatabaseConfig, PostgresChatHistory   |
| `POSTGRES_PORT`     | PostgreSQL port     | DatabaseConfig, PostgresChatHistory   |
| `POSTGRES_DB`       | PostgreSQL database | DatabaseConfig, PostgresChatHistory   |
| `POSTGRES_USER`     | PostgreSQL user     | DatabaseConfig, PostgresChatHistory   |
| `POSTGRES_PASSWORD` | PostgreSQL password | DatabaseConfig, PostgresChatHistory   |
