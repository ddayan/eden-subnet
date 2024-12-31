from dataclasses import dataclass
from pydantic import BaseModel
from eden_subnet.base.data_models import ModuleSettings


class TokenUsage(BaseModel):
    """Token usage model"""

    total_tokens: int = 0
    prompt_tokens: int = 0
    request_tokens: int = 0
    response_tokens: int = 0



# class EmbeddingRequest(BaseModel):
#     messages: List[Union[Message, Dict[str, str]]]
#     models: Optional[List[str]]
#     api_key: Optional[str]

@dataclass
class MinerSettings(ModuleSettings):
    tokeniser_encoding: str = "cl100k_base"

