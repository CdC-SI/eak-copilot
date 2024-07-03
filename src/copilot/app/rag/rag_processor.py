from typing import List

from rag.prompts import OPENAI_RAG_SYSTEM_PROMPT_DE, MISTRAL_RAG_SYSTEM_PROMPT_DE
from rag.models import RAGRequest, EmbeddingRequest

from components.embedding.factory import EmbeddingFactory
from components.llm.factory import LLMFactory

from autocomplete.queries import semantic_similarity_match

import requests
from llama_cpp import Llama

class RAGProcessor:
    """
    Class implementing the RAG process

    Parameters
    ----------
    model : str
        The name of the model to be used.
    max_tokens : int
        The maximum number of tokens for the model.
    stream : bool
        Whether to stream the response.
    temperature : float
        The temperature for the model's output distribution.
    top_p : float
        The cumulative probability cutoff for the model's output distribution.
    top_k : int
        The number of top tokens to consider for the model's output distribution.
    embedding_model : str, optional
        The name of the embedding model to be used, by default "text-embedding-ada-002".
    llm_model_name : str, optional
        The name of the language model to be used, by default "gpt-3.5-turbo-0125".

    """
    def __init__(self, model: str, max_token: int, stream: bool, temperature: float,
                 top_p: float, top_k: int, embedding_model: str = "text-embedding-ada-002", llm_model: str = "gpt-3.5-turbo-0125"):
        self.model = model
        self.max_tokens = max_token
        self.stream = stream
        self.temperature = temperature
        self.top_p = top_p
        self.k_retrieve = top_k

        self.embedding_model = embedding_model
        self.embedding_client = self.init_embedding_client()

        self.llm_model = llm_model
        self.llm_client = self.init_llm_client()

    def init_embedding_client(self):
        """
        Initialize and return an embedding client based on `self.embedding_model_name`.

        Returns
        -------
        object
            An instance of the appropriate embedding client based on `self.embedding_model_name`.
        """
        return EmbeddingFactory.get_embedding_client(self.embedding_model)


    def init_llm_client(self):
        """
        Initialize and return a language model client based on `self.llm_model_name`.

        Returns
        -------
        object
            An instance of the appropriate language model client.
            If `self.llm_model_name` is "gpt-3.5-turbo-0125", it returns an OpenAI client.
            If `self.llm_model_name` is "Qwen/Qwen1.5-0.5B-Chat-GGUF", it returns a Llama client.
            If `self.llm_model_name` is "mlx-community/Nous-Hermes-2-Mistral-7B-DPO-4bit-MLX", it currently does nothing and returns None.
        """
        return LLMFactory.get_llm_client(self.llm_model)

    def stream_response(self, context_docs: List[str], query: str, source_url: str):
        """
        Generate a response stream based on the language model specified by `self.llm_model_name`.

        Parameters
        ----------
        context_docs : list
            The context documents to be used for generating the response.
        query : str
            The query to be used for generating the response.
        source_url : str
            The source URL to be used for generating the response.

        Returns
        -------
        str
            The generated response.
        """
        if self.llm_model == "gpt-3.5-turbo-0125":
            messages = self.create_openai_message(context_docs, query)
            stream = self.create_openai_stream(messages)
            return self.generate_openai(stream, source_url)

        elif self.llm_model == "Qwen/Qwen1.5-0.5B-Chat-GGUF":
            messages = self.create_openai_message(context_docs, query)
            stream = self.create_qwen_stream(messages)
            return self.generate_qwen(stream, source_url)

        elif self.llm_model == "mlx-community/Nous-Hermes-2-Mistral-7B-DPO-4bit-MLX":
            messages = self.create_mlx_message(context_docs, query)
            stream = self.create_mlx_stream(messages)
            return self.generate_mlx(stream, source_url)

    async def retrieve(self, request: RAGRequest, language: str = None, k: int = 0):
        """
        Retrieve context documents related to the user input question.

        Only supports retrieval of 1 document at the moment (set in /config/config.yaml).

        .. todo::
            multi-doc retrieval later

        Parameters
        ----------
        request : RAGRequest
            User input question
        language : str
            Question and context documents language
        k : int, default 0
            Number of context documents to return (need to be revised, current logic in the code is confusing)
        """
        k = self.k_retrieve if k is None else k

        rows = await semantic_similarity_match(request.query, db_name='embeddings', language=language, k=k)
        documents = [dict(row) for row in rows][0]

        return {"contextDocs": documents["text"], "sourceUrl": documents["url"], "cosineSimilarity": documents["similarity_metric"]}

    async def process(self, request: RAGRequest):
        """
        Execute the RAG process and query the LLM model.

        Parameters
        ----------
        request : RAGRequest
            User input question

        Returns
        -------
        str
            LLM generated answer to the question
        """
        documents = await self.retrieve(request)
        context_docs = documents['contextDocs']
        source_url = documents['sourceUrl']

        return self.stream_response(context_docs, request.query, source_url)

    async def embed(self, text_input: EmbeddingRequest):
        """
        Get the embedding of an embedding request.

        Parameters
        ----------
        text_input : EmbeddingRequest

        Returns
        -------
        dict
            The requested text embedding
        """
        embedding = self.embedding_client.embed_query(text_input.text)
        return {"data": embedding}

    def create_openai_message(self, context_docs, query):
        """
        Format the message to send to the OpenAI API.

        Parameters
        ----------
        context_docs : str
            Context matching the query according to the retrieval process
        query : str
            User input question

        Returns
        -------
        list of dict
            Contains the message in the correct format to send to the OpenAI API

        """
        openai_rag_system_prompt = OPENAI_RAG_SYSTEM_PROMPT_DE.format(context_docs=context_docs, query=query)
        return [{"role": "system", "content": openai_rag_system_prompt},]

    def create_mlx_message(self, context_docs, query):
        mlx_rag_system_prompt = MISTRAL_RAG_SYSTEM_PROMPT_DE.format(context_docs=context_docs, query=query)
        return mlx_rag_system_prompt

    def create_openai_stream(self, messages):
        """
        Create a stream to communicate with OpenAI.

        Parameters
        ----------
        messages : dict

        Returns
        -------
        chat.completion
        """
        return self.llm_client.generate(messages)

    def create_qwen_stream(self, messages):
        return self.llm_client.generate(messages)

    def create_mlx_stream(self, messages):
        return self.llm_client.generate(messages)

    def generate_openai(self, stream, source_url):
        """
        Generate the answer using LLM.

        Parameters
        ----------
        openai_stream
        source_url

        Returns
        -------

        """
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content.encode("utf-8")
            else:
                # Send a special token indicating the end of the response
                yield f"\n\n<a href='{source_url}' target='_blank' class='source-link'>{source_url}</a>".encode("utf-8")
                return

    def generate_qwen(self, stream, source_url):
        for chunk in stream:
            if chunk["choices"][0]["finish_reason"] != "stop":
                if "content" in chunk["choices"][0]["delta"].keys():
                    yield chunk["choices"][0]["delta"]["content"].encode("utf-8")
            else:
                # Send a special token indicating the end of the response
                yield f"\n\n<a href='{source_url}' target='_blank' class='source-link'>{source_url}</a>".encode("utf-8")
                return

    def generate_mlx(self, stream, source_url):
        for chunk in stream:
            if chunk is not None:
                yield chunk.encode("utf-8")

        # Send a special token indicating the end of the response
        yield f"\n\n<a href='{source_url}' target='_blank' class='source-link'>{source_url}</a>".encode("utf-8")
        return