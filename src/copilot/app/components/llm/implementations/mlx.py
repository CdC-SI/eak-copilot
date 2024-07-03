"""
This module provides an implementation of the MLX LLM model.

Classes:
    MlxLLM: A class that encapsulates methods to interact with an MLX LLM deployed on a server.
"""
import os
import logging
from dotenv import load_dotenv
from typing import List
import requests

from components.llm.base import LLM
from components.config import SUPPORTED_MLX_LLM_MODELS, DEFAULT_MLX_LLM_MODEL

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import env vars
load_dotenv()
MLX_GENERATION_ENDPOINT = os.environ["MLX_GENERATION_ENDPOINT"]


class MlxLLM(LLM):
    """
    Class used to generate responses using a MLX Large Language Model (LLM) running on a local server.

    Attributes
    ----------
    model_name : str
        The name of the MLX LLM model to use for response generation.
    stream : bool
        Whether to stream the response generation.

    Methods
    -------
    generate(messages: List[dict]) -> str
        Generates a response for a list of messages using the MLX LLM model.
    stream()
        Placeholder method for streaming. Currently not implemented.
    """
    def __init__(self, model_name: str = DEFAULT_MLX_LLM_MODEL, stream: bool = True):
        self.model_name = model_name if model_name is not None and model_name in SUPPORTED_MLX_LLM_MODELS else DEFAULT_MLX_LLM_MODEL
        self.stream = stream

    # TO DO: proper return type for generate
    def generate(self, messages: List[dict]) -> str:
        """
        Generate a response using the MLX LLM model.

        Parameters
        ----------
        messages : List[dict]
            The messages to generate a response for.

        Returns
        -------
        Generator[str, None, None]
            A generator that yields the generated responses line by line.

        Raises
        ------
        Exception
            If an error occurs during generation.
        """
        try:
            response = requests.get(MLX_GENERATION_ENDPOINT, params={'prompt': messages, 'stream': self.stream}, stream=self.stream)
            for line in response.iter_lines():
                if line:
                    yield line.decode('utf-8')
        except Exception as e:
            raise e

    def stream(self):
        """
        Placeholder method for streaming. Currently not implemented.
        """
