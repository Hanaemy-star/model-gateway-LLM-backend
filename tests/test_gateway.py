import httpx
import pytest
import respx

from fastapi.testclient import TestClient

from gateway.main import app


UPSTREAM_URL = "http://localhost:11434/v1/chat/completions"

client = TestClient(app)


REQUEST_PAYLOAD = {
    "model": "qwen2.5:0.5b",
    "messages": [
        {
            "role": "user",
            "content": "Hello!"
        }
    ],
    "stream": False
}


SUCCESS_RESPONSE = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you?"
            },
            "finish_reason": "stop"
        }
    ]
}


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_successful_completion():

    with respx.mock:
        route = respx.post(UPSTREAM_URL).mock(
            return_value=httpx.Response(
                200,
                json=SUCCESS_RESPONSE
            )
        )

        response = client.post(
            "/v1/chat/completions",
            json=REQUEST_PAYLOAD
        )

    assert response.status_code == 200
    assert response.json() == SUCCESS_RESPONSE
    assert route.called


def test_backend_connection_error():

    with respx.mock:
        respx.post(UPSTREAM_URL).mock(
            side_effect=httpx.ConnectError(
                "Connection refused"
            )
        )

        response = client.post(
            "/v1/chat/completions",
            json=REQUEST_PAYLOAD
        )

    assert response.status_code == 503


def test_backend_timeout():

    with respx.mock:
        respx.post(UPSTREAM_URL).mock(
            side_effect=httpx.ReadTimeout(
                "Backend timeout"
            )
        )

        response = client.post(
            "/v1/chat/completions",
            json=REQUEST_PAYLOAD
        )

    assert response.status_code == 504


def test_backend_internal_error():

    with respx.mock:
        respx.post(UPSTREAM_URL).mock(
            return_value=httpx.Response(
                500,
                json={"error": "Internal server error"}
            )
        )

        response = client.post(
            "/v1/chat/completions",
            json=REQUEST_PAYLOAD
        )

    assert response.status_code == 502

def test_backend_too_many_request():

    with respx.mock:
        respx.post(UPSTREAM_URL).mock(
              return_value=httpx.Response(
                        429,
                        json={"error": "Too many requests"},
              )
          )
        response = client.post(
            "/v1/chat/completions",
            json=REQUEST_PAYLOAD,
        )

    assert response.status_code == 429

def test_backend_unhandled_status_returns_502():

    with respx.mock:
        respx.post(UPSTREAM_URL).mock(
            return_value=httpx.Response(
                502,
                json={"error": "undefiend error"},
            )
        )
        response = client.post(
            "/v1/chat/completions",
            json=REQUEST_PAYLOAD,
        )

    assert response.status_code == 502
