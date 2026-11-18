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
        except httpx.ConnectError:
            raise HTTPException(
                    status_code=503,
                    detail="Ошибка подключения",
                  )
        except httpx.TimeoutException:
            raise HTTPException(
                     status_code=504,
                     detail="Превышен лимит ожидания",
                  )
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
            elif exc.response.status_code == 502 or exc.response.status_code == 500:
                raise HTTPException(
                        status_code=502,
                        detail="Сервис временно недоступен",
                      )
            elif exc.response.status_code == 400:
                raise HTTPException(
                         status_code=400,
                         detail="Bad request",
                       )
            elif exc.response.status_code == 404:
                raise HTTPException(
                         status_code=404,
                         detail="Сервиса не существует",
                       )
            elif exc.response.status_code == 429:
                raise HTTPException(
                         status_code=429,
                         detail="Слишком много запросов",
                       )

            else:
                raise HTTPException(
                        status_code=exc.response.status_code,
                        detail="Неизвестная ошибка",
                      )

