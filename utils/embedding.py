from typing import List, Union

# Import env vars
from config.base_config import rag_config
from config.openai_config import openai

# Function to get embeddings for a text
def get_embedding(text: Union[List[str], str]):
    model = rag_config["embedding"]["model"]
    if model == "text-embedding-ada-002":
        response = openai.Embedding.create(
            input=text,
            engine=model,
        )
        return response['data']
    else:
        raise NotImplementedError("Model not supported")