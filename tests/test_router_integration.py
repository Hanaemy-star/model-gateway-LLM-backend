from fastapi.testclient import TestClient

from gateway.backends.base import BaseBackend
from gateway.main import app
from gateway.routing.router import RoundRobinRouter


class FakeBackend(BaseBackend):

    def __init__(self, name: str):
        self.name = name
        self.received_payloads = []

    async def generate(self, payload: dict) -> dict:
        self.received_payloads.append(payload)

        return {
            "backend": self.name,
        }


def test_gateway_round_robin_integration():

    # ARRANGE
    backend_a = FakeBackend("A")
    backend_b = FakeBackend("B")

    router = RoundRobinRouter(
        backends=[backend_a, backend_b]
    )

    payload = {
        "model": "qwen2.5:0.5b",
        "messages": [
            {
                "role": "user",
                "content": "Hello!",
            }
        ],
        "stream": False,
    }

    with TestClient(app) as client:

        original_router = app.state.router
        app.state.router = router

        try:
            # ACT
            responses = [
                client.post(
                    "/v1/chat/completions",
                    json=payload,
                )
                for _ in range(4)
            ]

        finally:
            app.state.router = original_router

    # ASSERT
    assert all(
        response.status_code == 200
        for response in responses
    )

    assert [
        response.json()["backend"]
        for response in responses
    ] == ["A", "B", "A", "B"]

    assert backend_a.received_payloads == [
        payload,
        payload,
    ]

    assert backend_b.received_payloads == [
        payload,
        payload,
    ]
