import json

import httpx
import pytest

from gateway.backends.ollama import OllamaBackend


ENDPOINT_URL = "http://ollama.test/v1/chat/completions"

PAYLOAD = {
    "model": "qwen2.5:0.5b",
    "messages": [
        {
            "role": "user",
            "content": "Hello!",
        }
    ],
    "stream": False,
}

EXPECTED_RESPONSE = {
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "Hello!",
            }
        }
    ]
}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_generate_success():
    received = {}

    def handler(request: httpx.Request) -> httpx.Response:
        received["method"] = request.method
        received["url"] = str(request.url)
        received["payload"] = json.loads(request.content)

        return httpx.Response(
            status_code=200,
            json=EXPECTED_RESPONSE,
        )

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as client:
        backend = OllamaBackend(
            client=client,
            endpoint_url=ENDPOINT_URL,
        )

        result = await backend.generate(PAYLOAD)

    assert result == EXPECTED_RESPONSE
    assert received["method"] == "POST"
    assert received["url"] == ENDPOINT_URL
    assert received["payload"] == PAYLOAD


@pytest.mark.anyio
async def test_generate_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=500,
            json={"error": "Internal server error"},
        )

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as client:
        backend = OllamaBackend(
            client=client,
            endpoint_url=ENDPOINT_URL,
        )

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await backend.generate(PAYLOAD)

    assert exc_info.value.response.status_code == 500


@pytest.mark.anyio
async def test_generate_connection_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(
            "Connection refused",
            request=request,
        )

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as client:
        backend = OllamaBackend(
            client=client,
            endpoint_url=ENDPOINT_URL,
        )

        with pytest.raises(httpx.ConnectError):
            await backend.generate(PAYLOAD)
