from gateway.backends.base import BaseBackend
from gateway.backends.bound import ModelBoundBackend
from gateway.config import GatewayConfig
from gateway.registry import ModelRegistry
from gateway.routing.router import RoundRobinRouter


def build_model_routers(
    config: GatewayConfig,
    backends: dict[str, BaseBackend],
) -> dict[str, RoundRobinRouter]:

    registry = ModelRegistry(config)

    routers: dict[str, RoundRobinRouter] = {}

    for model_name in config.models:
        candidates = registry.get_candidates(model_name)

        bound_backends: list[BaseBackend] = []

        for ref in candidates:
            adapter = backends[ref.name]
            physical_name = config.backends[ref.name].model
            model_bound = ModelBoundBackend(adapter, physical_name)

            bound_backends.append(model_bound)

        routers[model_name] = RoundRobinRouter(bound_backends)

    return routers
