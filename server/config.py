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
    # Re-run Lance compaction + reindex after this many inserts. 0 disables the
    # periodic pass (startup maintenance still runs). 50 keeps fragment count
    # and index staleness bounded without optimizing on every write.
    maint_interval: int = 50
    # Unix socket of the sibling cango-daemon. Shared volume in the deployed
    # podman-compose; the calendar shims in server/cango.py dial it.
    cango_socket: str = "/run/cango/cango.sock"
    # Opt-in: only register the cango calendar shims when explicitly enabled.
    # The personal deployment runs alongside cango-daemon and wants them; a work
    # deployment has no daemon and shouldn't surface calendar tools at all. Off
    # by default; flip with KNOWITALL_CANGO=1.
    cango: bool = False
    # Opt-in OAuth Protected Resource Metadata (RFC 9728): when both are set, an
    # unauthenticated /.well-known/oauth-protected-resource route advertises
    # oauth_authorization_server as this resource's OAuth issuer, so an MCP client
    # (e.g. claude.ai) can discover it instead of guessing an /authorize endpoint on
    # this same origin. Doesn't touch BearerTokenMiddleware or FastMCP's own auth
    # settings — purely a passive discovery document for a reverse-proxy-level OAuth
    # gateway (e.g. this deployment's Authelia instance) that this app has no other
    # awareness of. Off by default; a work deployment with no such gateway just
    # never registers the route.
    oauth_resource_url: str | None = None
    oauth_authorization_server: str | None = None


settings = Settings()
