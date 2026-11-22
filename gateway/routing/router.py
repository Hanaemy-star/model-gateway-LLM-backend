from gateway.backends.base import BaseBackend


class RoundRobinRouter:
    def __init__(
        self,
        backends: list[BaseBackend]
    ):

        if not backends:
            raise ValueError
        self.backends = backends
        self.index = 0
        self.num_backends = len(backends)

    async def generate(self, payload: dict) -> dict:
        backend = self.backends[self.index]
        self.index = (self.index + 1) % self.num_backends
        response = await backend.generate(payload)

        return response
    
