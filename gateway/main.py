from fastapi import FastAPI
from fastapi import HTTPException
import httpx

app = FastAPI()

@app.get("/health")
async def health():
    res = {
	"status": "ok",
	"service": "llm-gateway",
	}
    return res

post_completion_v1 = "/v1/chat/completions"

@app.post(post_completion_v1)
async def completion(payload: dict):
    async with httpx.AsyncClient(timeout=60, trust_env=False) as client:
        try:
            response = await client.post(
                "http://localhost:11434/v1/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            
            return response.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 503:
                raise HTTPException(
                        status_code=503,
                        detail="ollama выключена"
                      )
            elif exc.response.status_code == 504:
                raise HTTPException(
                        status_code=504,
                        detail="backend не отвечает вовремя"
                      )
