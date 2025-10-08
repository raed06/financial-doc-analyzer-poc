from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os

class Settings(BaseSettings):
    """Environment and application settings"""

    # -- LLM Configuration --

    # LLM URL and Port
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    # Model -> "Gemma2 2B is a lightweight decoder-only transformer language model with 2 billion parameters, optimized for efficiency and capable of handling context windows up to 8K tokens.”
    ollama_model: str = Field(default="gemma2:2b", env="OLLAMA_MODEL")

    # -- LangSmith Configuration --

    # Debug, test, evaluate, and monitor chains and intelligent agents
    langsmith_tracing: bool = Field(default=False, env="LANGSMITH_TRACING")
    langsmith_api_key: Optional[str] = Field(default=None, env="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="financial-doc-analyser-poc", env="LANGSMITH_PROJECT")

    # Application Settings

    # Sets the maximum file upload size (in megabytes)
    max_upload_size_mb: int = Field(default=50, env="MAX_UPLOAD_SIZE_MB")

    # Defines how large each text chunk should be when splitting data for processing
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")

    # Specifies how many characters overlap between chunks to preserve context continuity
    chunk_overlap: int = Field(default=200, env="CHUNK_SIZE")

    # Local directory path where the vector store (e.g., embeddings database) will be saved
    vector_store_path: str = Field(default="./data/vector_store", env="VECTOR_STORE_PATH")

    # -- MCP Server Configuration --
    mcp_words_port: int = Field(default=5001, env="MCP_WORDS_PORT")
    mcp_proxy_words_port: int = Field(default=5101, env="MCP_PROXY_WORDS_PORT")

    # -- Security --

    # API key used for authenticating requests between MCP components
    mcp_api_key: str = Field(default="demo-key-insecure", env="MCP_API_KEY")

    # Paths
    upload_dir: str = "./data/uploads"
    processed_dir: str = "./data/processed"
    logs_dir: str = "./logs"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        for directory in [self.upload_dir, self.processed_dir, self.vector_store_path, self.logs_dir]:
            os.makedirs(directory, exist_ok=True)


settings = Settings()