"""
This module provides an implementation of the Llama-cpp LLM model.

Classes:
    LlamaCppLLM: A class that encapsulates methods to interact with a Llama-cpp LLM instanciated locally with llama-cpp-python.
"""

import logging

from components.llm.base import LLM
from components.config import SUPPORTED_LLAMACPP_LLM_MODELS, DEFAULT_LLAMACPP_LLM_MODEL
from typing import List

from llama_cpp import Llama

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LlamaCppLLM(LLM):
    """
    Class used to generate responses using a LlamaCpp Large Language Model (LLM) running locally.

    Attributes
    ----------
    model_name : str
        The name of the LlamaCpp LLM model to use for response generation.
    stream : bool
        Whether to stream the response generation.
    temperature : float
        The temperature to use for response generation.
    top_p : float
        The top-p value to use for response generation.
    top_k : int
        The top-k value to use for response generation.
    quantization : int
        The quantization level to use for the LlamaCpp model.
    max_tokens : int
        The maximum number of tokens to generate.
    n_ctx : int
        The context size to use for the LlamaCpp model.
    n_gpu_layers : int
        The number of GPU layers to use for the LlamaCpp model.
    verbose : bool
        Whether to print verbose output.
    client : Llama
        The Llama client used to generate responses.

    Methods
    -------
    generate(messages: List[dict]) -> str
        Generates a response for a list of messages using the LlamaCpp LLM model.
    stream(messages: List[str])
        Placeholder method for streaming. Currently not implemented.
    """
    def __init__(self, model_name: str = DEFAULT_LLAMACPP_LLM_MODEL, stream: bool = True, temperature: float = 0.0, top_p: float = 0.95, top_k: int = 0, quantization: int = 8, max_tokens: int = 512, n_ctx: int = 8192, n_gpu_layers: int = -1, verbose: bool = False):
        self.model_name = model_name if model_name is not None and model_name in SUPPORTED_LLAMACPP_LLM_MODELS else DEFAULT_LLAMACPP_LLM_MODEL
        self.stream = stream
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.quantization = quantization
        self.max_tokens = max_tokens
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        self.client  = Llama.from_pretrained(
            repo_id=self.model_name,
            filename=f"*q{self.quantization}_0.gguf",
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            verbose=self.verbose
        )

    # TO DO: proper return type for generate
    def generate(self, messages: List[dict]) -> str:
        """
        Generates a response for a list of messages using the LlamaCpp LLM model.

        Parameters
        ----------
        messages : List[dict]
            A list of messages. Each message is a dictionary containing the necessary "roles" and "content" for the LLM.

        Returns
        -------
        str
            The generated response.

        Raises
        ------
        Exception
            If there is an error in creating the chat completion, an exception is raised.
        """
        try:
            return self.client.create_chat_completion(
            messages=messages,
            stream=self.stream,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_tokens)
        except Exception as e:
            raise e

    def stream(self, messages: List[str]):
        """
        Placeholder method for streaming.

        This method is currently not implemented.

        Parameters
        ----------
        messages : List[str]
            A list of messages. Each message is a string.
        """
