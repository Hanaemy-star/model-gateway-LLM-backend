import pytest

from gateway.config import GatewayConfig
from gateway.registry import ModelRegistry


@pytest.fixture
def registry() -> ModelRegistry:
    config = GatewayConfig.model_validate(
        {
            "backends": {
                "ollama-a": {
                    "type": "ollama",
                    "endpoint_url": "http://localhost:11434/v1/chat/completions",
                    "model": "qwen2.5:0.5b",
                },
                "ollama-b": {
                    "type": "ollama",
                    "endpoint_url": "http://localhost:11435/v1/chat/completions",
                    "model": "qwen2.5:1.5b",
                },
            },
            "models": {
                "qwen-auto": {
                    "backends": [
                        {"name": "ollama-a", "priority": 1, "weight": 3},
                        {"name": "ollama-b", "priority": 2, "weight": 1},
                    ]
                },
                "qwen-small": {
                    "backends": [
                        {"name": "ollama-a", "priority": 1, "weight": 1}
                    ]
                },
            },
        }
    )
    return ModelRegistry(config)


def test_get_candidates_returns_matching_backends(registry: ModelRegistry) -> None:
    candidates = registry.get_candidates("qwen-auto")

    assert [ref.name for ref in candidates] == ["ollama-a", "ollama-b"]


def test_get_candidates_preserves_priority_and_weight(registry: ModelRegistry) -> None:
    candidates = registry.get_candidates("qwen-auto")

    assert [(ref.priority, ref.weight) for ref in candidates] == [(1, 3), (2, 1)]


def test_different_logical_model_has_own_candidates(registry: ModelRegistry) -> None:
    candidates = registry.get_candidates("qwen-small")

    assert len(candidates) == 1
    assert candidates[0].name == "ollama-a"


def test_unknown_model_raises_key_error(registry: ModelRegistry) -> None:
    with pytest.raises(KeyError):
        registry.get_candidates("unknown-model")
