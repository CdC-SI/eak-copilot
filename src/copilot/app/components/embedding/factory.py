from components.embedding.base import Embedding
from components.embedding.implementations import *
from components.config import SUPPORTED_OPENAI_EMBEDDING_MODELS, SUPPORTED_ST_EMBEDDING_MODELS

class EmbeddingFactory:
    """
    Factory class for creating embedding clients.

    This class provides a static method to create instances of embedding clients based on a string identifier.

    Methods
    -------
    get_embedding_client(embedding_model: str) -> Embedding
        Factory method to instantiate embedding clients based on a string identifier.
    """

    @staticmethod
    def get_embedding_client(embedding_model: str) -> Embedding:
        """
        Factory method to instantiate embedding clients based on a string identifier.

        Parameters
        ----------
        embedding_model : str
            The name of the embedding model. Currently supported models are "text-embedding-ada-002" and
            "sentence-transformers/distiluse-base-multilingual-cased-v1".

        Returns
        -------
        Embedding
            An instance of the appropriate embedding client.

        Raises
        ------
        ValueError
            If the `embedding_model` is not supported.
        """
        if embedding_model in SUPPORTED_OPENAI_EMBEDDING_MODELS:
            return OpenAIEmbeddings()
        elif embedding_model in SUPPORTED_ST_EMBEDDING_MODELS:
            return SentenceTransformersEmbeddings()
        else:
            raise ValueError(f"Unsupported embedding model type: {embedding_model}")