"""
This module provides an implementation of the OpenAI LLM model.

Classes:
    OpenAILLM: A class that encapsulates methods to interact with OpenAI's language model APIs.
"""

import logging

from typing import List
from components.llm.base import LLM
from components.config import SUPPORTED_OPENAI_LLM_MODELS, DEFAULT_OPENAI_LLM_MODEL

from config.openai_config import openai

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OpenAILLM(LLM):
    """
    Class used to generate responses using an OpenAI API Large Language Model (LLM).

    Attributes
    ----------
    model_name : str
        The name of the OpenAI LLM model to use for response generation.
    stream : bool
        Whether to stream the response generation.
    temperature : float
        The temperature to use for response generation.
    top_p : float
        The top-p value to use for response generation.
    top_k : int
        The top-k value to use for response generation.
    max_tokens : int
        The maximum number of tokens to generate.
    verbose : bool
        Whether to print verbose output.
    client : openai.OpenAI
        The OpenAI client used to generate responses.

    Methods
    -------
    generate(messages: List[dict]) -> str
        Generates a response for a list of messages using the OpenAI LLM model.
    stream()
        Placeholder method for streaming. Currently not implemented.
    """
    def __init__(self, model_name: str = DEFAULT_OPENAI_LLM_MODEL, stream: bool = True, temperature: float = 0.0, top_p: float = 0.95, max_tokens: int = 512, verbose: bool = False):
        self.model_name = model_name if model_name is not None and model_name in SUPPORTED_OPENAI_LLM_MODELS else DEFAULT_OPENAI_LLM_MODEL
        self.stream = stream
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.client = openai.OpenAI()

    def generate(self, messages: List[dict]) -> str:
        """
        Generate a response using the OpenAI LLM model.

        Parameters
        ----------
        messages : List[dict]
            The messages to generate a response for.

        Returns
        -------
        str
            The generated response.

        Raises
        ------
        Exception
            If an error occurs during generation.
        """
        try:
            return self.client.chat.completions.create(
                model=self.model_name,
                stream=self.stream,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                messages=messages
            )
        except Exception as e:
            raise e

    def stream(self):
        """
        Placeholder method for streaming. Currently not implemented.
        """
