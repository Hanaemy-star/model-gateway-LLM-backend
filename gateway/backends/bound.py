from gateway.backends.base import BaseBackend


class ModelBoundBackend(BaseBackend):
    def __init__(
        self,
        backend: BaseBackend,
        physical_model: str
    ):

        self.backend = backend
        self.model = physical_model

    async def generate(self, payload: dict) -> dict:
        payload_copy = payload.copy()
        payload_copy["model"] = self.model

        return await self.backend.generate(payload_copy)
