from dotenv import dotenv_values
import uvicorn
import tiktoken
import sys

from loguru import logger
from pydantic import Field, BaseModel

from communex.balance import to_nano
from communex.module.module import Module, endpoint
from communex.module.server import ModuleServer
from communex.client import Ss58Address  # type: ignore
from communex.compat.key import classic_load_key
from communex.module._rate_limiters.limiters import StakeLimiterParams

from eden_subnet.miner.tiktokenizer import TikTokenizer
from eden_subnet.miner.data_models import MinerSettings
from eden_subnet.base.data_models import EmbeddingRequest, EmbeddingResponse


logger.remove()
logger.add(sys.stderr, level="DEBUG")

NETUID = 10

tokenizer = TikTokenizer()
tokenizer.embedding_function = tiktoken.get_encoding("cl100k_base")


class Miner(Module):
    tokenizer: TikTokenizer = TikTokenizer()
    ss58_address: Ss58Address = Field(default_factory=None)
    use_testnet: bool
    call_timeout: int = 60

    settings: MinerSettings
    testnet: bool
    def __init__(
        self,
        key_name: str,
        host: str,
        port: int,
        testnet: bool = False,
    ) -> None:
        super().__init__()
        self.settings = MinerSettings(host, port, classic_load_key(key_name))
        self.testnet = testnet

    @endpoint
    def generate(self, request: dict, m: str) -> EmbeddingResponse:
        # Process the request
        embedding_req = EmbeddingRequest.model_validate(request)
        embedding = self.tokenizer.embedding_function.encode(embedding_req.message)
        return EmbeddingResponse(embedding=list(embedding))


def stake_to_ratio(stake: int, multiplier: int = 1) -> float:
    '''
    https://github.com/renlabs-dev/synthia/blob/c1689c0243dcea669d74754a14c54712a41ad45e/src/synthia/miner/cli.py#L22C1-L42C51
    '''
    max_ratio = 4
    base_ratio = 2
    if multiplier <= 1/max_ratio:
        raise ValueError(
            f"Given multiplier {multiplier} would set 0 tokens for all stakes"
        )

    def mult_2(x: int) -> int:
        return x * 2

    # 10x engineer switch case (btw, this actually optimizes)
    match stake:
        case _ if stake < to_nano(10_000):
            return 0
        # 20 * 10 ** -1 request per 4000 * 10 ** -1 second
        case _ if stake < to_nano(500_000):
            return base_ratio * multiplier
        case _:
            # 30 * 10 ** -1 requests per 4000 * 10 ** -1 second
            return mult_2(base_ratio) * multiplier


def serve(key_name: str, host: str = "0.0.0.0", port: int = 10011):
    miner_module = Miner(key_name, host, port)
    stake_limiter = StakeLimiterParams(
    epoch=800,
    cache_age=600,
    get_refill_per_epoch=stake_to_ratio,
    )
    server = ModuleServer(
        miner_module, 
        miner_module.settings.keypair, 
        subnets_whitelist=[NETUID],
        limiter=stake_limiter
    )
    miner_app = server.get_fastapi_app()
    uvicorn.run(miner_app, host=host, port=port)

if __name__ == '__main__':
    # miner = Miner("miner_key", "0.0.0.0", 10001)
    config = dotenv_values(".env.miner")
    serve("test_key")