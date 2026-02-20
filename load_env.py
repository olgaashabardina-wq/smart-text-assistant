import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# грузим .env строго из папки проекта (где лежит этот файл)
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    normal_model: str
    thinking_model: str
    history_path: str
    timeout_seconds: float

def get_settings() -> Settings:
    return Settings(
        api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        base_url=os.getenv("OPENAI_BASE_URL", "").strip(),
        normal_model=os.getenv("NORMAL_MODEL", "grok-code-fast-1").strip(),
        thinking_model=os.getenv("THINKING_MODEL", "claude-sonnet-4-5-20250929").strip(),
        history_path=os.getenv("HISTORY_PATH", "history.json").strip(),
        timeout_seconds=float(os.getenv("TIMEOUT_SECONDS", "30").strip()),
    )


