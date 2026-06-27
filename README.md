# Sylvi

Sylvi is a multimodal, stateful memory copilot designed to index, query, and manage personal context directly from a Telegram client. It utilizes LangGraph for intent routing and pipeline orchestration, storing long-term vector records in Pinecone and relational transaction histories in PostgreSQL.

## Architecture

The system is split into two core asynchronous components: the Ingestion Pipeline (which indexes user documents, links, and media) and the Query Pipeline (which resolves intent, handles conversational state, and triggers RAG synthesis).

```mermaid
graph TD
    User([Telegram User]) -->|Message| Bot[Telegram Client]
    
    %% Ingestion Flow
    Bot -->|Media/Link/Voice| Queue[Async Ingestion Queue]
    Queue -->|Worker| IngestionGraph[LangGraph: Ingestion Graph]
    IngestionGraph -->|Text & Transcripts| Embed[Gemini Embeddings]
    Embed -->|768d Vectors| Pinecone[(Pinecone Vector DB)]
    
    %% Query Flow
    Bot -->|Text Query| QueryGraph[LangGraph: Query Graph]
    QueryGraph -->|Intent Route| Router{Intent Router}
    
    Router -->|Chitchat| Chit[Chitchat Node]
    Router -->|Reminder| Rem[Reminder Node]
    Router -->|Profile Query| SQLite[Supabase Postgres]
    Router -->|Retrieval RAG| Retrieve[Pinecone Fetch]
    Router -->|Fact Deletion| Del[Fact Deletion Node]
    
    Chit --> Gen[Generation Node]
    SQLite --> Gen
    Retrieve --> Gen
    
    Gen -->|Synthesized Reply| User
    Rem -->|Confirm/Trigger| User
    Del -->|Delete Fact| SQLite
```

## System Requirements

- **Python**: 3.11+
- **Database (Relational)**: PostgreSQL (Supabase) for profile facts, chat logs, and reminder states.
- **Database (Vector)**: Pinecone (768-dimension Index, Cosine metric) for document embeddings.
- **Large Language Models**: 
  - Text & Structured Output: Llama-3-70b-8192 (via Groq Cloud API)
  - Vision Capabilities: Llama-3.2-11b-vision-preview (via Groq Cloud API)
  - Text Embeddings: text-embedding-004 (via Google Gemini API)

## Configuration

Duplicate `.env.example` to `.env` and configure the following variables:

```env
# API Credentials
GEMINI_API_KEYS=your_gemini_api_keys
GROQ_API_KEYS=your_groq_api_keys
PINECONE_API_KEY=your_pinecone_api_key
TELEGRAM_BOT_TOKEN=your_telegram_bot_token

# Connection Strings
DATABASE_URL=postgresql://username:password@hostname:port/database
PINECONE_INDEX_NAME=sylvi-memory
```

## Local Development

Install project dependencies and activate the virtual environment using `uv`:

```bash
# Sync dependencies
uv sync

# Run database schema migration tests
PYTHONPATH="." DATABASE_URL="your_postgres_url" uv run python tests/test_profile_db.py

# Run intent routing tests
PYTHONPATH="." uv run python tests/test_query.py

# Start the application locally
uv run python main.py
```

## Deployment

The application includes a `Dockerfile` designed for automated containerized hosting platforms (e.g. Render Web Services).

1. Connect your repository to Render.
2. Select **Docker** as the environment runtime.
3. Configure the required environment secrets (listed in the configuration section) inside the Render dashboard.
4. Bind the deployment to port `7860` for container health checks.
