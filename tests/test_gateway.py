import httpx
import pytest
import respx
import copy

from fastapi.testclient import TestClient

from gateway.main import app


UPSTREAM_URL = "http://localhost:11434/v1/chat/completions"

@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


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


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_successful_completion(client):

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


def test_backend_connection_error(client):

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


def test_backend_timeout(client):

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


def test_backend_internal_error(client):

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

def test_backend_too_many_request(client):

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

def test_backend_unhandled_status_returns_502(client):

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

def test_missing_model(client):
    payload = copy.deepcopy(REQUEST_PAYLOAD)
    del payload["model"]

    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(UPSTREAM_URL).mock(
            return_value=httpx.Response(
                200,
                json=SUCCESS_RESPONSE,
            )
        )

        response = client.post(
            "/v1/chat/completions",
            json=payload,
        )

        assert response.status_code == 422
        assert route.call_count == 0


def test_empty_messages(client):
    payload = copy.deepcopy(REQUEST_PAYLOAD)
    payload["messages"] = []

    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(UPSTREAM_URL).mock(
            return_value=httpx.Response(
                200,
                json=SUCCESS_RESPONSE,
            )
        )

        response = client.post(
            "/v1/chat/completions",
            json=payload,
        )

        assert response.status_code == 422
        assert route.call_count == 0


def test_incorrect_temperature(client):
    payload = copy.deepcopy(REQUEST_PAYLOAD)
    payload["temperature"] = 5.0

    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(UPSTREAM_URL).mock(
            return_value=httpx.Response(
                200,
                json=SUCCESS_RESPONSE,
            )
        )

        response = client.post(
            "/v1/chat/completions",
            json=payload,
        )

        assert response.status_code == 422
        assert route.call_count == 0


def test_incorrect_role(client):
    payload = copy.deepcopy(REQUEST_PAYLOAD)
    payload["messages"][0]["role"] = "unknown"

    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(UPSTREAM_URL).mock(
            return_value=httpx.Response(
                200,
                json=SUCCESS_RESPONSE,
            )
        )

        response = client.post(
            "/v1/chat/completions",
            json=payload,
        )

        assert response.status_code == 422
        assert route.call_count == 0
