from fastapi import FastAPI
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
        response = await client.post(
            "http://localhost:11434/v1/chat/completions",
            json=payload,
        )

        return response.json()

