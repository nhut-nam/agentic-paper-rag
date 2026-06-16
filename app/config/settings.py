from pydantic_settings import BaseSettings
from typing import List, Optional
import os
from app.utils.helper import load_yaml

class Settings(BaseSettings):
    # Marker Settings
    USE_LLM: bool = False
    OUTPUT_FORMAT: str = "markdown"
    DISABLE_IMAGE_EXTRACTION: bool = False
    DISABLE_OCR: bool = True
    
    # Pipeline Settings
    OUTPUT_ROOT: str = "marker_output"
    IMAGE_PRODUCING_TYPES: List[str] = [
        "Table", "Code", "Equation", "Form", 
        "Handwriting", "Figure", "Picture"
    ]
    TRANSFORM_TYPES: List[str] = ["Table", "Code", "Equation", "Form"]
    
    # App Settings
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "logs"

    # Database Settings
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "postgres"
    DB_NAME: str = "paper_intel"
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432

    # Ollama Settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_LLM_MODEL: str = "qwen3.5:cloud"
    
    # Local Embedding Settings
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

    @property
    def database_url(self) -> str:
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    class Config:
        env_file = ".env"
        extra = "ignore"

# Load config from YAML if it exists
config_path = "config.yaml"
yaml_config = load_yaml(config_path)

# Initialize settings with YAML values (YAML overrides defaults)
settings = Settings(**yaml_config)
