from dataclasses import dataclass
from pydantic import BaseModel

from eden_subnet.base.data_models import ModuleSettings


class GenerateRequest(BaseModel):
    messages: list
    model: str

@dataclass
class ValidatorSettings(ModuleSettings):
    model: str
    api_url: str
    api_key: str