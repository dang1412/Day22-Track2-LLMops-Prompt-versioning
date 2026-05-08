import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gpt-5.4-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "day22-prompt-versioning")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

os.environ.setdefault("LANGSMITH_TRACING", LANGSMITH_TRACING)
os.environ.setdefault("LANGSMITH_PROJECT", LANGSMITH_PROJECT)
os.environ.setdefault("LANGSMITH_ENDPOINT", LANGSMITH_ENDPOINT)
if LANGSMITH_API_KEY:
    os.environ.setdefault("LANGSMITH_API_KEY", LANGSMITH_API_KEY)


def _require(name: str, value: str | None) -> str:
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def validate() -> None:
    _require("OPENAI_API_KEY", OPENAI_API_KEY)
    _require("LANGSMITH_API_KEY", LANGSMITH_API_KEY)


if __name__ == "__main__":
    validate()
    print("✅ Config loaded successfully")
    print(f"   LangSmith project : {LANGSMITH_PROJECT}")
    print(f"   OpenAI endpoint   : {OPENAI_BASE_URL}")
    print(f"   Default LLM model : {DEFAULT_LLM_MODEL}")
    print(f"   Embedding model   : {EMBEDDING_MODEL}")
