from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request

from gateway.backends.factory import build_backends
from gateway.config import load_config
from gateway.routing.builder import build_model_routers

from .schemas import ChatCompletionRequest


CONFIG_PATH = (
        Path(__file__).resolve().parent.parent
        / "configs"
        / "backends.yaml"
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with httpx.AsyncClient(timeout=60.0, trust_env=False) as client:
        config = load_config(str(CONFIG_PATH))
        backends = build_backends(config, client)
        routers = build_model_routers(config, backends)

        app.state.routers = routers
        app.state.config = config
        #временная переменная
        app.state.router = routers["qwen2.5:0.5b"]
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

    model_name = payload.model
    router = request.app.state.routers.get(model_name)

    if router is None:
        raise HTTPException(
                 status_code=404,
                 detail="Запрашиваемый ресурс не найден"
                 )

    try:
        response = await router.generate(payload.model_dump(exclude_none=True))
        
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

    return response
