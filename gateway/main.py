from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request

from .schemas import ChatCompletionRequest


UPSTREAM_URL = "http://localhost:11434/v1/chat/completions"


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with httpx.AsyncClient(timeout=60.0, trust_env=False) as client:
        app.state.http_client = client
        yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "llm-gateway"}


@app.post("/v1/chat/completions")
async def completion(request: Request, payload: ChatCompletionRequest):
    # Пока мы не реализовали передачу SSE-потока, stream=True запрещён.
    if payload.stream:
        raise HTTPException(
            status_code=400,
            detail="Streaming пока не поддерживается",
        )

    try:
        response = await request.app.state.http_client.post(
            UPSTREAM_URL,
            json=payload.model_dump(exclude_none=True),
        )
        response.raise_for_status()

    except httpx.ConnectError as exc:
        raise HTTPException(
            status_code=503,
            detail="Нет соединения с LLM-бэкендом",
        ) from exc

    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail="Превышено время ожидания LLM-бэкенда",
        ) from exc

    except httpx.HTTPStatusError as exc:
        upstream_status = exc.response.status_code
        passthrough_statuses = {400, 404, 422, 429, 503, 504}
        gateway_status = (
            upstream_status if upstream_status in passthrough_statuses else 502
        )

        raise HTTPException(
            status_code=gateway_status,
            detail=f"Ошибка LLM-бэкенда: HTTP {upstream_status}",
        ) from exc

    return response.json()
