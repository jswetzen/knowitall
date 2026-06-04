from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KNOWITALL_", env_file=".env", extra="ignore")

    token: str = "dev-insecure-token"
    ollama_url: str = "http://192.168.1.33:11434"
    ollama_model: str = "nomic-embed-text-v2-moe"
    data_dir: Path = Path("./data")
    host: str = "0.0.0.0"
    port: int = 8765
    embedding_dim: int = 768
    # When true, instrumented tools emit one structured timing line to stderr
    # per call (stage breakdown: embed / lance / kuzu + total + row counts).
    # Off by default; flip with KNOWITALL_PROFILE=1 to find the bottleneck.
    profile: bool = False
    # Unix socket of the sibling cango-daemon. Shared volume in the deployed
    # podman-compose; the calendar shims in server/cango.py dial it.
    cango_socket: str = "/run/cango/cango.sock"


settings = Settings()
