from fastapi import FastAPI, Request, Response, HTTPException
import httpx
import os
import time
import jwt
import logging
import sys

from config.settings import settings

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MCP_BACKEND_URL = f"http://localhost:{settings.mcp_words_port}/mcp"
SECRET_KEY = os.getenv("JWT_SECRET_KEY", settings.mcp_api_key)
ISSUER = "internal-auth-service"
AUDIENCE = "mcp-internal-api"
JWT_ALGORITHM = "HS256"


def generate_jwt() -> str:
    now = int(time.time())
    payload = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "iat": now,
        "exp": now + 300,
        "sub": "proxy-client"
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token


@app.api_route("/mcp/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy(full_path: str, request: Request):
    logger.info("=== Proxy received request ===")
    logger.info("Client → Proxy headers: %s", dict(request.headers))

    query = request.url.query
    target_path = full_path
    if query:
        full_url = f"{MCP_BACKEND_URL.rstrip('/')}/{target_path}?{query}"
    else:
        full_url = f"{MCP_BACKEND_URL.rstrip('/')}/{target_path}"

    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() != "host"
    }

    token = generate_jwt()
    headers["Authorization"] = f"Bearer {token}"

    logger.info("Proxy → Backend headers: %s", headers)

    body = await request.body()

    async with httpx.AsyncClient() as client:
        try:
            proxied_request = client.build_request(
                request.method,
                full_url,
                headers=headers,
                content=body
            )
            proxied_response = await client.send(proxied_request, follow_redirects=False, stream=True)
        except httpx.RequestError as exc:
            logger.error("Error contacting backend MCP: %s", exc)
            raise HTTPException(status_code=502, detail=f"Error contacting backend MCP: {exc}") from exc

        # Handle 307 redirect manually: forward same request to location
        if proxied_response.status_code == 307:
            redirect_url = proxied_response.headers.get("location")
            if redirect_url:
                logger.info("Received 307 redirect, following to %s", redirect_url)
                # Rebuild request to the redirect URL
                proxied_request2 = client.build_request(
                    request.method,
                    redirect_url,
                    headers=headers,
                    content=body
                )
                proxied_response = await client.send(proxied_request2, stream=True)

        content = await proxied_response.aread()

        resp_headers = {
            k: v for k, v in proxied_response.headers.items()
            if k.lower() not in ("transfer-encoding", "content-encoding")
        }

        return Response(
            content=content,
            status_code=proxied_response.status_code,
            headers=resp_headers,
            media_type=proxied_response.headers.get("content-type")
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.mcp_proxy_words_port)
