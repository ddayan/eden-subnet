# Eden Subnet 10

## Introduction

The Eden Subnet simplifies the way machines learn and manage data through its Commune Subnet, which centers around a Vector Store. Think of it as creating memories for AI: it takes textual content—such as PDFs, markdown, and text files—transforms them into embeddings, and stores them for AI retrieval. This is part of a process known as Retrieval-Augmented Generation (RAG), crucial for enhancing AI interactions.

Beyond this fundamental operation, the Vector Store serves as a crucial repository for embeddings—dense numerical representations of text tokens produced by machine learning algorithms. As a key component of a broader AI ecosystem, it significantly improves the accessibility and utility of data for large language models (LLMs) and other AI systems. The store is populated and maintained by dedicated nodes called "embedding miners," who are instrumental in ensuring the availability of high-quality, semantically rich embeddings. These miners, along with validators, play a vital role in generating and refining these embeddings to power a variety of AI-driven applications and innovations.

The strength of this subnet lies in its ability to enhance AI capabilities through Retrieval-Augmented Generation (RAG). By integrating this mechanism, AI agents, chatbots, and tools can expand their understanding and discuss topics outside their inherent knowledge base, simply by referencing materials like PDFs. This approach is computationally efficient compared to the alternatives of training new models or manually incorporating extensive documents into a model’s context window.

Vector stores play a crucial role in augmenting the abilities and knowledge base of existing AI models. We incentivize miners to deliver high-quality embedding services, rewarding them based on the excellence of their contributions. This not only enhances the overall quality of AI interactions but also creates a win-win situation for both miners and AI developers.

### Future Work

We are initiating our subnet by deploying simple embedding miners. These miners utilize embedding functions to generate token embeddings, which are then stored in vectorstores. The second phase of our project involves leveraging these embedding miners to establish a distributed vector store. Our strategy includes creating domain-specific knowledge stores, which will be indexed on the blockchain. These stores will provide data as a service to large language models (LLMs). Additionally, we are exploring the integration of knowledge graphs and the storage of synthetic data. Potential collaborations are being considered, including one with the Synthia subnet, to enhance our capabilities in synthetic data handling.

## Setup

### Minimum Requirements

- **VRAM**: 8GB minimum

### Docker
<!-- #### With Docker

- [Install Docker](https://docs.docker.com/get-docker/)
- Run `docker pull ghcr.io/agicommies/synthia:9d23f1f`
- Run `docker run -v ~/.commune:/root/.commune -it [-p <port>:<port>] ghcr.io/agicommies/synthia:9d23f1f`
- Run `poetry shell` to enter the enviroment
  
##### Operating with docker

- You can quit docker with ctrl+d
- You can dettach from your session with ctrl+p followed by ctrl+q
- You can attach back to your session by running `docker attach <id>`
- You can list the ids of your containers with `docker ps`
- Note that you should pass the ports you're going to use to the container (with `-p <port>:<port>`) to bind them to your host machine.
- You can pass enviroments variables to docker with `-e <VARIABLE>=<value>`.
    e.g `docker run -e ANTHROPIC_API_KEY=<your-anthropic-api-key> -v ~/.commune:/root/.commune -it ghcr.io/agicommies/synthia:9d23f1f` -->

### Local

- Install Python 3
  - `sudo apt install python3`
- [Install Poetry](https://python-poetry.org/docs/)
- Install the Python dependencies with `poetry install`
- **! IMPORTANT** Enter the Python environment with `poetry shell`

#### Running A Miner

1. Generate a key for your miner.

    ```bash
        comx key create <key-name>
    ```

2. Create and edit your `.env` file with your miner key name and port number.

    ```bash
        cp .env.example .env
    ```

3. Run your miner directly or via pm2

    ```bash
        poetry run python -m eden_subnet.miner.cli
    ```

    *Note: you need to keep this process alive, running in the background. Some options are tmux, pm2 or nohup.*

    ```bash
        pm2 start "poetry run python -m eden_subnet.miner.cli" --name eden-subnet-miner
    ```

4. Register your miner with the following command:

    ```bash
        comx module register <key-name> <machine-public-ip> <port> --netuid 10
    ```

#### Running A Validator

1. Generate a key for your validator.

    ```bash
        comx key create <key-name>
    ```

2. Create and edit your `.env` file with your key name and API keys

    ```bash
        cp .env.example .env
    ```

3. Run your validator directly or via pm2

    ```bash
        poetry run python -m eden_subnet.validator.cli

    ```

    *Note: you need to keep this process alive, running in the background. Some options are tmux, pm2 or nohup.*

    ```bash
        pm2 start "poetry run python -m eden_subnet.validator.cli" --name eden-subnet-validator
    ```

4. Register your validator with the following command:

    ```bash
        comx module register <key-name> --netuid 10
    ```

### Important Notes

- Global burn fee for registering modules starts at 10 COM
- Fee doubles each threshold reached per epoch
- Validator staking requirement: 5000 COM
- Stake requirements:
  - Must cover burn fee
  - Must meet minimum vote requirement for validators
- Warning: Anomalous validator behavior will result in blacklisting

Everyone should start off on pretty even footing as there is not a wild difference in embedding model. So the tie breaker is how much stake you have on the miner. Pulling all your stake off a miner will not be wise if you do not want to get bumped off the subnet. That will be replaced with how much storage you are providing to the network as the features roll out.

### Support

- Discord: Contact coolrazor or bakobiibizo
- Email: <info@agentartificial.com>

## Contributions

Built by [Eden](https://twitter.com/project_eden_ai) in partnership with [Agent Artificial](https://agentartificial.com)
Subnet of [Commune](https://github.com/commune-ai/commune), based on the [Communex Synthia repo](https://github.com/agicommies/synthia).

### Contributing

- Submit pull requests with detailed change descriptions
- Open issues for larger changes
