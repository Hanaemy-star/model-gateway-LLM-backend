import httpx

class OllamaBackend:
    def __init__(
        self,
        client: httpx.AsyncClient,
        endpoint_url: str
    ):

        self.client = client
        self.endpoint_url = endpoint_url

    async def generate(self, payload: dict) -> dict:
        response = await self.client.post(
                        self.endpoint_url,
                        json=payload,
                    )

        response.raise_for_status()

        return response.json()
    
