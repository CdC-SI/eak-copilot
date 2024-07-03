from typing import Any
import tiktoken

from components.config import SUPPORTED_OPENAI_EMBEDDING_MODELS

class TokenizerFactory:
    """
    Factory class for creating tokenizer instances based on the embedding model.

    Attributes
    ----------
    None

    Methods
    -------
    get_tokenizer_client(embedding_model: str) -> Any:
        Returns a tokenizer instance based on the embedding model.
    """

    @staticmethod
    def get_tokenizer_client(embedding_model: str) -> Any: # TO DO: Update Any to Tokenizer class
        """
        Returns a tokenizer instance based on the embedding model.

        Parameters
        ----------
        embedding_model : str
            The name of the embedding model.

        Returns
        -------
        tokenizer : Any
            The tokenizer instance. If the embedding model is supported,
            it returns a tiktoken tokenizer with max_input_tokens set to 8192.
        """
        if embedding_model in SUPPORTED_OPENAI_EMBEDDING_MODELS:
            tokenizer = tiktoken.get_encoding("cl100k_base")
            tokenizer.max_input_tokens = 8192
            return tokenizer