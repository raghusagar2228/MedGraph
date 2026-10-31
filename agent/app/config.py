import os
from typing import Optional, Literal
from pydantic_settings import BaseSettings, SettingsConfigDict

RetrievalMode = Literal["mock", "hybrid_live", "external_api"]


class Settings(BaseSettings):
    # Retrieval Mode:
    # - "mock": Fast, deterministic offline in-memory data for unit testing & development
    # - "hybrid_live": Direct native Neo4j (Cypher) + local persistent dense vector search (FastEmbed)
    # - "external_api": Proxies retrieval calls to the companion Express backend
    retrieval_mode: RetrievalMode = "mock"

    # Backwards-compatible flag: if use_mock_api is set explicitly, it aligns with retrieval_mode
    use_mock_api: bool = True

    # Neo4j Graph Database Configuration (Used in hybrid_live mode)
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"
    neo4j_connection_timeout: float = 5.0

    # Vector Semantic Search Configuration (Used in hybrid_live mode)
    vector_model_name: str = "BAAI/bge-small-en-v1.5"
    vector_index_path: str = "data/vector_store.json"

    # External Backend API Endpoints (Used in external_api mode)
    api_base_url: str = "http://localhost:5000"
    graph_search_endpoint: str = "/api/graph/search"
    vector_search_endpoint: str = "/api/vector/search"
    admission_endpoint: str = "/api/patients/{patient_id}/admissions"

    # LLM configuration
    llm_provider: Literal["mock", "openai", "google_genai", "anthropic"] = "mock"
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o"
    google_api_key: Optional[str] = None
    google_model: str = "gemini-2.0-flash"
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-3-5-sonnet-20241022"
    temperature: float = 0.0

    # Agent / Workflow Settings
    max_retries: int = 2
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def get_effective_retrieval_mode(self) -> RetrievalMode:
        """Resolve the effective retrieval mode, ensuring the system never

        silently reports mock data when live mode is requested.
        """
        # If explicitly specified in environment or settings
        mode = os.getenv("RETRIEVAL_MODE", self.retrieval_mode).lower().strip()
        if mode in ("mock", "hybrid_live", "external_api"):
            return mode  # type: ignore

        if not self.use_mock_api:
            return "external_api"
        return "mock"

    def resolve_provider(self) -> str:
        """Auto-detect provider if default is mock or explicit."""
        if self.llm_provider != "mock":
            return self.llm_provider
        if self.openai_api_key or os.getenv("OPENAI_API_KEY"):
            return "openai"
        if self.google_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
            return "google_genai"
        if self.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic"
        return "mock"


settings = Settings()
