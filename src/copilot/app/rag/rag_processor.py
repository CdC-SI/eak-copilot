from rag.prompts import OPENAI_RAG_SYSTEM_PROMPT_DE
from rag.models import RAGRequest, EmbeddingRequest

from autocomplete.queries import semantic_similarity_match
from utils.embedding import get_embedding


class RAGProcessor:
    """
    Class implementing the RAG process

    Parameters
    ----------
    model : str
        LLM Model used for generating chatbot answers
    max_token : int
    stream : bool
    temperature : float
    top_p : float
    top_k : int
    client
    """
    def __init__(self, model: str, max_token: int, stream: bool, temperature: float,
                 top_p: float, top_k: int, client):
        self.model = model
        self.max_tokens = max_token
        self.stream = stream
        self.temperature = temperature
        self.top_p = top_p
        self.k_retrieve = top_k

        self.client = client

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
        messages = self.create_openai_message(context_docs, request.query)
        openai_stream = self.create_openai_stream(messages)

        return self.generate(openai_stream, source_url)

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
        embedding = get_embedding(text_input.text)
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
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=self.stream,
            temperature=self.temperature,
            top_p=self.top_p)

    def generate(self, openai_stream, source_url):
        """
        Generate the answer using LLM.

        Parameters
        ----------
        openai_stream
        source_url

        Returns
        -------

        """
        for chunk in openai_stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content.encode("utf-8")
            else:
                # Send a special token indicating the end of the response
                yield f"\n\n<a href='{source_url}' target='_blank' class='source-link'>{source_url}</a>".encode("utf-8")
                return
