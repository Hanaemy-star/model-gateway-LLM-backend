from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class BackendConfig(BaseModel):
    type: Literal["ollama"]
    endpoint_url: str = Field(min_length=1)
    model: str = Field(min_length=1)

class BackendRef(BaseModel):
    name: str = Field(min_length=1)
    priority: int = Field(ge=1)
    weight: int = Field(gt=0)

class ModelConfig(BaseModel):
    backends: list[BackendRef] = Field(min_length=1)

class GatewayConfig(BaseModel):
    backends: dict[str, BackendConfig]
    models: dict[str, ModelConfig]

    @model_validator(mode="after")
    def validate_backend_reference(self):
        for model_name, model_config in self.models.items():
            for ref in model_config.backends:
                if ref.name not in self.backends:
                    raise ValueError(
                        f"Модель {model_name} имеет ссылку на не существующий бекэнд {ref.name}"
                    ) 
        return self

def load_config(path: str) -> GatewayConfig:
    file = Path(path)
    
    text = file.read_text(encoding="utf-8")
    data = yaml.safe_load(text)

    return GatewayConfig.model_validate(data)
