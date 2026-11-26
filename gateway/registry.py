from gateway.config import BackendRef, GatewayConfig


class ModelRegistry:
    def __init__(self, config: GatewayConfig):
        self.config = config

    def get_candidates(self, model_name: str) -> list[BackendRef]:
        if model_name not in self.config.models:
            raise KeyError(f"Неизвестная модель: {model_name}")
        return self.config.models[model_name].backends
