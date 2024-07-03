import logging

from typing import List
from components.embedding.base import Embedding
from components.tokenizer.factory import TokenizerFactory

# Import env vars
from config.openai_config import openai
from components.config import SUPPORTED_OPENAI_EMBEDDING_MODELS, DEFAULT_OPENAI_EMBEDDING_MODEL

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OpenAIEmbeddings(Embedding):
    """
    Class for embedding text documents using OpenAI's models.

    Attributes
    ----------
    model_name : str
        The name of the OpenAI model to use for embedding.
    client : openai.OpenAI
        The OpenAI client.
    tokenizer : Tokenizer
        The tokenizer instance.

    Methods
    -------
    embed_documents(texts: List[str]) -> List[List[float]]:
        Embeds a list of text documents using the OpenAI API.
    embed_query(text: str) -> List[float]:
        Embeds a single text query using the OpenAI API.
    aembed_documents(texts: List[str]) -> List[List[float]]:
        Asynchronously embeds a list of text documents using the OpenAI API.
    aembed_query(text: str) -> List[float]:
        Asynchronously embeds a single text query using the OpenAI API.
    """
    def __init__(self, model_name: str = DEFAULT_OPENAI_EMBEDDING_MODEL):
        """
        Initializes the OpenAIEmbeddings instance.

        Parameters
        ----------
        model_name : str, optional
            The name of the OpenAI model to use for embedding. If not provided,
            the default model is used.
        """
        self.model_name = model_name if model_name is not None and model_name in SUPPORTED_OPENAI_EMBEDDING_MODELS else DEFAULT_OPENAI_EMBEDDING_MODEL
        self.client = openai.OpenAI()
        self.tokenizer = TokenizerFactory.get_tokenizer_client(self.model_name)
        super().__init__(self.tokenizer)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of text documents using the OpenAI API.

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
            If the API call fails.
        """
        texts = super().embed_documents(texts)
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts,
            )
            return response.data
        except Exception as e:
            raise e

    def embed_query(self, text: str) -> List[float]:
        """
        Embeds a single text query using the OpenAI API.

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