from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from server import config

OPEN_PATHS = {"/healthz"}


class BearerTokenMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path in OPEN_PATHS:
            return await call_next(request)
        header = request.headers.get("authorization", "")
        expected = f"Bearer {config.settings.token}"
        if header != expected:
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)
