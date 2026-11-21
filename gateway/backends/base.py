
from abc import ABC, abstractmethod


class BaseBackend(ABC):

    @abstractmethod
    async def generate(self, payload: dict) -> dict:
        pass
