"""Unit tests for the model-name binding decorator."""

import pytest

from gateway.backends.base import BaseBackend
from gateway.backends.bound import ModelBoundBackend


class FakeBackend(BaseBackend):
    def __init__(self):
        self.received_payloads: list[dict] = []

    async def generate(self, payload: dict) -> dict:
        self.received_payloads.append(payload)
        return {"backend": "fake", "model": payload["model"]}


class FailingBackend(BaseBackend):
    async def generate(self, payload: dict) -> dict:
        raise RuntimeError("upstream failed")


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_rewrites_model_and_delegates_to_backend():
    inner = FakeBackend()
    bound = ModelBoundBackend(inner, "qwen2.5:0.5b")
    payload = {"model": "qwen-auto", "messages": [{"role": "user", "content": "Hi"}]}

    result = await bound.generate(payload)

    assert result == {"backend": "fake", "model": "qwen2.5:0.5b"}
    assert inner.received_payloads == [
        {"model": "qwen2.5:0.5b", "messages": [{"role": "user", "content": "Hi"}]}
    ]


@pytest.mark.anyio
async def test_does_not_mutate_original_payload():
    inner = FakeBackend()
    bound = ModelBoundBackend(inner, "qwen2.5:1.5b")
    payload = {"model": "qwen-auto", "messages": [{"role": "user", "content": "Hi"}]}

    await bound.generate(payload)

    assert payload["model"] == "qwen-auto"
    assert inner.received_payloads[0] is not payload


@pytest.mark.anyio
async def test_preserves_inner_backend_errors():
    bound = ModelBoundBackend(FailingBackend(), "qwen2.5:0.5b")

    with pytest.raises(RuntimeError, match="upstream failed"):
        await bound.generate({"model": "qwen-auto", "messages": []})
