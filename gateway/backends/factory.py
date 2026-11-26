import httpx

from gateway.backends.base import BaseBackend
from gateway.backends.ollama import OllamaBackend
from gateway.config import GatewayConfig


def build_backends(
    config: GatewayConfig,
    client: httpx.AsyncClient,
) -> dict[str, BaseBackend]:
    adapters = {}

    for backend_name, backend_config in config.backends.items():
        if backend_config.type == "ollama":
            adapters[backend_name] = OllamaBackend(client, backend_config.endpoint_url)
        else:
            raise ValueError(
                f"Неподдерживаемый тип бэкенда: {backend_config.type}"
            )
    return adapters
