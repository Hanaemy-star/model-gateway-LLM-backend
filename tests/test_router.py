import pytest

from gateway.backends.base import BaseBackend
from gateway.routing.router import RoundRobinRouter


class FakeBackend(BaseBackend):
    """Тестовый бэкенд без обращения к реальной LLM."""

    def __init__(self, name: str):
        self.name = name
        self.received_payloads = []

    async def generate(self, payload: dict) -> dict:
        self.received_payloads.append(payload)

        return {"backend": self.name}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_round_robin_two_backends():
    backend_a = FakeBackend("A")
    backend_b = FakeBackend("B")

    router = RoundRobinRouter(
        backends=[backend_a, backend_b]
    )

    results = []

    for _ in range(4):
        result = await router.generate({"message": "hello"})
        results.append(result["backend"])

    assert results == ["A", "B", "A", "B"]


@pytest.mark.anyio
async def test_round_robin_three_backends():
    backends = [
        FakeBackend("A"),
        FakeBackend("B"),
        FakeBackend("C"),
    ]

    router = RoundRobinRouter(backends=backends)

    results = []

    for _ in range(7):
        result = await router.generate({"message": "hello"})
        results.append(result["backend"])

    assert results == [
        "A", "B", "C",
        "A", "B", "C",
        "A",
    ]


def test_router_empty_backends():
    with pytest.raises(ValueError):
        RoundRobinRouter(backends=[])


@pytest.mark.anyio
async def test_router_forwards_payload():
    backend = FakeBackend("A")

    router = RoundRobinRouter(backends=[backend])

    payload = {
        "model": "qwen2.5:0.5b",
        "messages": [
            {"role": "user", "content": "Hello!"}
        ],
    }

    result = await router.generate(payload)

    assert result == {"backend": "A"}
    assert backend.received_payloads == [payload]
