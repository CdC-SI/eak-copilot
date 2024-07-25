from typing import List, Union
import logging

# Import env vars
from config.base_config import rag_config
from config.openai_config import clientAI

import openai

supported_models = ["text-embedding-ada-002"]

logger = logging.getLogger(__name__)


# Function to get embeddings for a text
def get_embedding(text: Union[List[str], str]):
    model = rag_config["embedding"]["model"]
    if model in supported_models:
        try:
            response = clientAI.embeddings.create(
                input=text,
                model=model,
            )
            return str(response.data[0].embedding)
        except openai.BadRequestError as e:
            logger.error(e.message)
            logger.error(f"Failed to get embeddings for text of length: {len(text)}")
            return None
    else:
        raise NotImplementedError(f"Model '{model}' is not supported")
