import random
import json
import time
import sys

import argparse
from typing import Optional
import requests
import requests.sessions
from requests.exceptions import ConnectionError
import aiohttp
import asyncio
from asyncio import Semaphore
import numpy as np
from loguru import logger
from dotenv import dotenv_values
from scipy.spatial.distance import cosine

from communex.module.client import ModuleClient  # type: ignore 
from communex.compat.key import try_classic_load_key  # type: ignore
from communex.client import CommuneClient  # type: ignore
from communex._common import get_node_url  # type: ignore

from eden_subnet.base.data_models import EmbeddingRequest, EmbeddingResponse, Message
from eden_subnet.validator.data_models import ValidatorSettings, GenerateRequest
from eden_subnet.miner.tiktokenizer import TikTokenizer
# from eden_subnet.validator.validator import Ss58Address
from communex.types import Ss58Address  # type: ignore
from eden_subnet.validator.topics import TOPICS

DEBUG = True
NETUID = 10

tokenizer = TikTokenizer()
comx = CommuneClient(get_node_url())

logger.remove()
logger.add(sys.stderr, level="DEBUG")
#TODO direct error level and down to stderr, while warning and up go to stdout



class Validator:
    settings: ValidatorSettings

    @logger.catch()
    def __init__(self, config: dict) -> None:
        self.key_name = config["KEY_NAME"]  # Store the key name
        keypair = try_classic_load_key(config["KEY_NAME"])
        self.settings = ValidatorSettings(
            host=config["HOST"],
            port=config["PORT"],
            keypair=keypair,
            model = config["MODEL"],
            api_url = config["API_URL"],
            api_key = config["API_KEY"],
        )
    
    def get_uid(self):
        """
        Retrieves the unique identifier associated with the validator.
        """
        key_map = comx.query_map_key(netuid=NETUID)
        ss58_address = self.settings.keypair.ss58_address
        for uid, ss58 in key_map.items():
            if ss58 == ss58_address:
                return uid
        raise ValueError(
            f"UID not found, {ss58_address} please check your validator is registered with: comx module info {self.key_name}"
        )
     
    def get_sample_result(self):
        logger.info("Getting sample result")
        topic = random.choice(TOPICS)
        logger.debug(f"sample topic: {topic}")
        topic_message = Message(
            content=f"Please write a short alegory about this topic: {topic}",
            role="user",
        )
        sample_result = self.make_llm_request(message=topic_message)
        logger.debug(f"Sample Result: {sample_result}")
        return sample_result

    async def get_miner_responses(self, selfuid, encoding, prompt_message, addresses):
        """
        Retrieves similarities from different addresses by making concurrent requests and validating the responses.

        Parameters:
            selfuid: The unique identifier of the calling entity.
            encoding: The encoding type for the validation.
            prompt_message: The message used for generating the response.
            addresses: A dictionary containing UIDs and corresponding addresses.

        Returns:
            A dictionary containing the responses from different addresses after validation.
        """
        miner_responses = {}
        semaphore = Semaphore(50 if not DEBUG else 1)  # Limit concurrent requests to 50

        async def process_address(uid, address):
            if uid == selfuid:
                return
            url = f"http://{address}/generate"
            if f"http://{self.settings.host}:{self.settings.port}/generate" == url:
                return
            
            async with semaphore:
                try:
                    async with aiohttp.ClientSession() as session:
                        response = await self.make_request_async(session, prompt_message, url)
                        logger.debug(f"Miner ({uid}) response: {response}")
                    if response:
                        miner_responses[uid] = self.validate_input(encoding, response)
                except Exception as e:
                    logger.debug(f"Error getting similarities for {uid}: {e}\n{e.args}\n")

        tasks = [process_address(uid, address) for uid, address in addresses.items()]
        await asyncio.gather(*tasks)
        
        return miner_responses
    
    def make_llm_request(self, message: Message):
        data = {
            "messages": [message.model_dump()],
            "model": self.settings.model,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.settings.api_key}" or None
        }
        try:
            response = requests.post(self.settings.api_url, data=json.dumps(data), headers=headers)
        except Exception as e:
            logger.debug(f"Error making request: {e}\n{e.args}")
            return
        logger.debug(f"Response text: {response.text}")
        return response.text

    async def make_request_async(self, session, message: Message, url):
        logger.debug(f"Making async request to: {url}")
        
        payload = json.dumps({
        #   "model": self.settings.model,
          "messages": [
            {
              "role": "user",
              "content": message.content
            }
          ],
        })
        headers = {
        #   'Authorization': f'Bearer {self.settings.api_key}',
          'Content-Type': 'application/json'
        }
        
        try:
            async with session.post(url, headers=headers, data=payload, timeout=30) as response:
                if response.status == 200:
                    try:
                        response_json = await response.json()
                        logger.success(f"Request successfull: {response_json[:150]}...")
                        return response_json
                    except json.JSONDecodeError:
                        logger.error(f"Failed to decode JSON from response. Status: {response.status}, Content-Type: {response.headers.get('Content-Type')}")
                        return None
                else:
                    logger.debug(f"Request failed with status {response.status}.")
                    response_text = await response.text()
                    logger.debug(f"Response content: {response_text[:200]}...")  # Log first 200 characters of responvalidation resulse
                    return None
        except ConnectionError as e:
            logger.debug(f"Network error occurred: {e}\n{e.args}\n")
            return None
        except Exception as e:
            logger.debug(f"Unexpected error occurred: {e}\n{e.args}\n")
            return None

    def cosine_similarity(self, embedding1, embedding2):
        """
        Calculates the cosine similarity between two embeddings.
        Returns:
            The normalized similarity value multiplied by 99.
        """
        logger.info("Calculating cosine similarity")

        # Example vectors
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Normalizing vectors
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)

        # Calculating cosine distance
        cosine_distance = 1 -cosine(vec1_norm, vec2_norm) * 99 + 99
        if cosine_distance < 0:
            cosine_distance = 1
        logger.success(f"cosine distance: {cosine_distance}")
        return round(cosine_distance - 30)

    def validate_input(self, embedding1, embedding2):
        """
        A function to validate input embeddings by evaluating sample similarity.

        Parameters:
            embedding1: The first embedding.
            embedding2: The second embedding.

        Returns:
            The result of the validation after evaluating the similarity.
        """
        if not embedding1:
            logger.error("embedding1 is empty, cannot validate")
            return 1
        
        if not embedding2:
            logger.warning("embedding2 is empty, setting score to 1")
            return 1
        
        try:
            response = self.cosine_similarity(
                embedding1=embedding1, embedding2=embedding2
            )
            result = round(response, 2)
            logger.success(f"validation result: {result}")
        except Exception as e:
            logger.warning(f"could not connect to miner, adjusting score.\n{e}")
            result = 1
        return result * 100
    
    async def validate_loop(self):
        selfuid = self.get_uid()

        # Query the chain for the addresses, weights, and keys
        address_dict = comx.query_map_address(netuid=NETUID)
        
        if DEBUG:
            debug_miners = [1, 5, 530, 532, 82]
            address_dict = {k : v for k,v in address_dict.items() if k in debug_miners}

        weights_dict = self.get_querymaps_weights()
        weights_dict = self.set_default_weights(selfuid, weights_dict, address_dict)

        keys_dict = comx.query_map_key(netuid=NETUID)

        # Generate a sample result and convert it into an embedding
        sample_result = self.get_sample_result()
        prompt_message = Message(content=str(sample_result), role="user")
        encoding = tokenizer.embedding_function.encode(str(sample_result))
        
        # Get the responses from the miners
        responses_dict = await self.get_miner_responses(
            selfuid, encoding, prompt_message, address_dict
        )
        
        # Score the modules
        score_dict = self.score_modules(
            weights_dict, address_dict, keys_dict, responses_dict
        )
        logger.debug(f"score_dict: {score_dict}")
        
        # Convert the weights into a list of uids and weight values for voting.
        logger.info("Loading weights")
        uids = []
        weights = []
        UINT16_MAX = 2 ** 16 - 1
        for uid, weight in score_dict.items():
            if uid == selfuid:
                continue
            uids.append(uid)
            weights.append(int(weight * UINT16_MAX))
        logger.debug(f"uids: {uids}\nweights: {weights}")
        subnet_weights = {"uids": uids, "weights": weights}
        
        # Save a copy for debugging
        with open("data/weights.json", "w") as f:
            f.write(json.dumps(subnet_weights, indent=4))

        # Vote on the modules   
        try:
            result = comx.vote(key=self.settings.keypair, netuid=NETUID, weights=weights, uids=uids)
            if result.is_success:
                logger.info("Voted successfully")
            else:
                logger.error(f"{result}")
        except Exception as e:
            logger.exception(f"Error voting: {e}")
        
        time.sleep(60)

    def run_voteloop(self):
        while True:
            asyncio.run(self.validate_loop())

    def get_querymaps_weights(self):
        """
        Parses existing weights and creates a dictionary mapping UID to weight.

        Parameters:
            None

        Returns:
            dict: A dictionary mapping UID to weight based on the existing weights.
        """
        logger.info("Parsing existing weights")
        # sourcery skip: dict-comprehension, identity-comprehension, inline-immediately-returned-variable
        result = comx.query_map_weights(netuid=10)
        weight_dict = {}
        if result is None:
            logger.warning("No weights found, returning empty dictionary")
            return weight_dict
            
        weights = result[1]
        if weights is None:
            logger.warning("Weights are None, returning empty dictionary")
            return weight_dict
            
        for uid, weight in weights:
            if uid not in weight_dict:
                weight_dict[uid] = weight
        return weight_dict

    def set_default_weights(self, selfuid, weights, addresses):
        """
        Checks for unranked weights and assigns a default weight of 30 to the missing weights.

        Parameters:
            selfuid: The unique identifier of the validator.
            weights: A dictionary mapping UID to weight.
            addresses: A dictionary containing addresses of validators.

        Returns:
            dict: A dictionary with updated weights.
        """
        logger.info("Checking for unranked weights")
        for uid, _ in addresses.items():
            for id in  range(820):
                if id not in weights and uid != selfuid:
                    weights[uid] = 30

        return weights

    def scale_numbers(self, numbers):
        """
        A function that scales a list of numbers between 0 and 1 based on their minimum and maximum
        """
        logger.info("Scaling numbers")
        min_value = 1
        max_value = max(numbers)
        return [(number - min_value) / (max_value - min_value) for number in numbers]

    def list_to_dict(self, list):
        return {i: list[i] for i in range(len(list))}

    def scale_dict_values(self, dictionary):
        """
        A function that scales the values of a dictionary between 0 and 1 based on their minimum and maximum values.
        """
        #TODO check what if dictionary is empty
        logger.info("Scaling dictionary values")
        min_value = 0.000000000000001
        logger.debug(f"min_value: {min_value}")
        max_value = max(dictionary.values()) or 1
        logger.debug(f"max_value: {max_value}")
        
        return {
            key: (value - min_value) / (max_value - min_value)
            for key, value in dictionary.items()
        }
    
    def get_staketo_values(self):
        staketo_map = comx.query_map_staketo()
        key_map = comx.query_map_key(netuid=10)
        staketo_dict = {}
        for uid, key in key_map.items():
            staketo_value = 0.00001
            if key not in staketo_map:
                continue
            staketo = staketo_map[key]
            for stakeitems in staketo:
                _, value = stakeitems
                staketo_value += value
            staketo_dict[uid] = staketo_value
        return staketo_dict

    def score_modules(self, weights_dict, staketos_dict, keys_dict, similairity_dict):
        """
        Calculates the scores for modules based on weights, staketos, keys, and similarity values.

        Parameters:
            self: The instance of the class.
            weights_dict (dict): A dictionary containing weights for each module.
            staketos_dict (dict): A dictionary containing staketos for each module.
            keys_dict (dict): A dictionary containing keys for each module.
            similarity_dict (dict): A dictionary containing similarity values for each module.

        Returns:
            dict: A dictionary containing the calculated scores for each module.
        """
        logger.info("Calculating scores")
        logger.debug(f"weights_dict: {weights_dict}\nsimilairity_dict: {similairity_dict}")
    
        scaled_weight_dict = self.scale_dict_values(weights_dict)
        scaled_similairity_dict = self.scale_dict_values(similairity_dict)
        staketo_dict = self.get_staketo_values()
        scaled_staketo_dict = self.scale_dict_values(staketo_dict)
        scaled_scores = {}
        for uid in keys_dict.keys():     
            if uid not in scaled_similairity_dict:
                continue       
            calculated_score = (
                (scaled_weight_dict[uid] * 0.4) + (scaled_similairity_dict[uid] * 0.2) + (scaled_staketo_dict[uid] * 0.2)
            ) 
            if calculated_score <= 0:
                calculated_score = 0.00001
            logger.debug(f"UID: {uid} Score: {calculated_score}")
            scaled_scores[uid] = calculated_score
            
        print(scaled_scores)
        
        return self.scale_dict_values(scaled_scores)
    

def parseargs():
    """
    Parse command line arguments using argparse module.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--key_name", type=str, default=None, help="Name of the validator as it will appear on chain")
    parser.add_argument("--module_path", type=str, default=None, help="Filename with out the .json suffix in the ~/.commune/key directory")
    parser.add_argument("--host", type=str, default=None, help="Host address for the validator")
    parser.add_argument("--port", type=int, default=None, help="Port for the validator")
    parser.add_argument("--url", type=str, default=None, help="Base URL for inference request.")
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI or Agent Artificial API Key")
    parser.add_argument("--model", type=str, default=None, help="OpenAI or Agent Artificial Model")
    return parser.parse_args()


async def get_miner_generation_async(
    miner_info: tuple[tuple[str, int], Ss58Address],
    input: EmbeddingRequest,
) -> Optional[EmbeddingResponse]:
    try:
        connection, miner_key = miner_info
        module_ip, module_port = connection
        logger.debug(f"Call {miner_key} - {module_ip}:{module_port}")
        key = try_classic_load_key("valis_key")
        client = ModuleClient(host=module_ip, port=int(module_port), key=key)
        params = {
            "request": input.model_dump(),  # Pass message at top level
            "m": "hello world"  # Second parameter
        }
        logger.debug(f"Params being sent: {params}")
        result: EmbeddingResponse = await client.call(
            fn="generate",
            target_key=miner_key, 
            params=params,
            # timeout=100
        )
        result = EmbeddingResponse.validate(result)
        logger.info(f"result: {result}")
        return result
    except Exception as e:
        logger.debug(f"Call error: {e}")
        return None
    
def serialize(data) -> bytes:
    txt = json.dumps(data)
    return txt.encode()

if __name__ == "__main__":
    if True:
        host:str = "127.0.0.1"
        port: int = 10011
        miner = ((host, port), Ss58Address("5HpUf4T6LFExajSLNYGxYpzNe9dBkWaNEzJLBrVxt7nFN8HP"))
        msg = EmbeddingRequest(
            message = "hello world"
        )
        # url = f"http://{host}:{port}/method/generate"
        logger.info(f"test local miner {msg.message}")
        # try:
            
        #         logger.info(f"async request to: {url}")
        #         msg = Message(
        #             role = "user",
        #             content = "hello world")
                
                
        #         make_request(msg, url)
        # except Exception as e:
        #     logger.info(e)
        # finally:
        #     exit(0)
        asyncio.run(get_miner_generation_async(miner, msg))
        
        exit(0)


    config = dotenv_values(".env")
    args = vars(parseargs())
    for k, v in args.items():
        if v:
            config[k] = v
    config["MODEL"] = config.get("AGENTARTIFICIAL_MODEL", config.get("OPENAI_MODEL", None))
    config["API_URL"] = config.get("AGENTARTIFICIAL_URL", config.get("OPENAI_URL", None))
    config["API_KEY"] = config.get("AGENTARTIFICIAL_API_KEY", config.get("OPENAI_API_KEY", None))
    
    logger.debug(f'Starting validator with MODEL: {config["MODEL"]}  API ENDPOINT: {config["API_URL"]}')
    
    validator = Validator(config)
    validator.run_voteloop()



# def make_request(message: Message, url):
#         data = {
#             "messages": [message.model_dump()],
#         }
#         headers = {
#             "Content-Type": "application/json"
#             # "Authorization": f"Bearer {self.settings.api_key}" or None
#         }
#         try:
#             response = requests.post(url, data=json.dumps(data), headers=headers)
#         except Exception as e:
#             logger.debug(f"Error making request: {e}\n{e.args}")
#             return
#         logger.debug(f"Response text: {response}")
#         return response.text
