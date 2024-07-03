import logging

from typing import List
from components.embedding.base import Embedding

from sentence_transformers import SentenceTransformer

# Import env vars
from components.config import SUPPORTED_ST_EMBEDDING_MODELS, DEFAULT_ST_EMBEDDING_MODEL

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SentenceTransformersEmbeddings(Embedding):
    """
    Class for embedding text documents using SentenceTransformers models.

    Attributes
    ----------
    model_name : str
        The name of the SentenceTransformers model to use for embedding.
    client : SentenceTransformer
        The SentenceTransformer client used to generate embeddings.

    Methods
    -------
    embed_documents(texts: List[str]) -> List[List[float]]
        Embeds a list of text documents using the SentenceTransformers model.
    embed_query(text: str) -> List[float]
        Embeds a single text query using the SentenceTransformers model.
    aembed_documents(texts: List[str]) -> List[List[float]]
        Asynchronously embeds a list of text documents using the SentenceTransformers model. Not implemented yet.
    aembed_query(text: str) -> List[float]
        Asynchronously embeds a single text query using the SentenceTransformers model. Not implemented yet.
    """
    def __init__(self, model_name: str = DEFAULT_ST_EMBEDDING_MODEL):
        self.model_name = model_name if model_name is not None and model_name in SUPPORTED_ST_EMBEDDING_MODELS else DEFAULT_ST_EMBEDDING_MODEL
        self.client = SentenceTransformer(self.model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Makes a call to the SentenceTransformers embedding model (running locally) to embed a list of text documents.

        Parameters
        ----------
        texts : list of str
            The list of texts to embed.

        Returns
        -------
        list of list of float
            List of embeddings, one for each text.

        Raises
        ------
        Exception
            If the embedding call fails.
        """
        try:
            response = self.client.encode(texts).tolist()
            return response
        except Exception as e:
            raise e

    def embed_query(self, text: str) -> List[float]:
        """
        Makes a call to the SentenceTransformers embedding model (running locally) to embed a single text query.

        Parameters
        ----------
        text : str
            The text to embed.

        Returns
        -------
        list of float
            Embedding for the text.
        """
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError("This method is not implemented yet.")

    async def aembed_query(self, text: str) -> List[float]:
        raise NotImplementedError("This method is not implemented yet.")
