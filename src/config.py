"""
Global configuration for RAG 
"""

from pathlib import Path
import os
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    groq_api_key: str = Field(..., description="Required: GROQ_API_KEY")
    pinecone_api_key: str = Field(..., description="Required: PINECONE_API_KEY")
    
    embedding_model_name: str = Field(
        default_factory=lambda: os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
    )
    llm_model_name: str = Field(
        default_factory=lambda: os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    )
    pinecone_index_name: str = Field(
        default_factory=lambda: os.getenv("PINECONE_INDEX_NAME", "rag-chatbot-opt")
    )

    pinecone_environment: str = Field(
        default_factory=lambda: os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
    )
    
    embedding_dim: int = 384
    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 4
    
    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path("data"))
    raw_docs_dir: Path = Field(default_factory=lambda: Path("data/input"))
    
    @field_validator('embedding_model_name', mode='after')
    @classmethod
    def set_embedding_dim(cls, v):
        if "mini" in v.lower():
            Settings.embedding_dim = 384
        elif "large" in v.lower():
            Settings.embedding_dim = 768
        return v
    
    class Config:
        env_file = ".env"
        extra = "ignore"
        case_sensitive = False

try:
    config = Settings()
except Exception as e:
    print(f"CONFIG ERROR: {e}")
    print("Check .env: GROQ_API_KEY, PINECONE_API_KEY required")
    raise

# Legacy exports
GROQ_API_KEY = config.groq_api_key
PINECONE_API_KEY = config.pinecone_api_key
PINECONE_ENVIRONMENT = config.pinecone_environment
PINECONE_INDEX_NAME = config.pinecone_index_name
EMBEDDING_MODEL_NAME = config.embedding_model_name
LLM_MODEL_NAME = config.llm_model_name
CHUNK_SIZE = config.chunk_size
CHUNK_OVERLAP = config.chunk_overlap
TOP_K = config.top_k
