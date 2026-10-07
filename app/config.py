import os
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Base directory of the repository
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env file
load_dotenv(BASE_DIR / ".env")

class Settings(BaseModel):
    # App Settings
    APP_NAME: str = "Hybrid AI Engine"
    VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "True").lower() in ("true", "1")
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8000"))

    # Pinecone Configuration
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "hybrid-ai-engine")
    PINECONE_CLOUD: str = os.getenv("PINECONE_CLOUD", "aws")
    PINECONE_REGION: str = os.getenv("PINECONE_REGION", "us-east-1")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))

    # OpenAI & LLM Settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # Classical ML Configuration
    MIN_SAMPLES_TO_TRAIN: int = int(os.getenv("MIN_SAMPLES_TO_TRAIN", "100"))
    MODEL_ARTIFACTS_DIR: Path = BASE_DIR / os.getenv("MODEL_ARTIFACTS_DIR", "artifacts/models")
    DATA_STORAGE_DIR: Path = BASE_DIR / os.getenv("DATA_STORAGE_DIR", "artifacts/data")

    def ensure_directories(self):
        self.MODEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        self.DATA_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.ensure_directories()
