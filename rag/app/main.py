import logging

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# Load env variables
from config.base_config import rag_config, rag_app_config
from config.network_config import CORS_ALLOWED_ORIGINS
from config.pgvector_config import SIMILARITY_METRICS
from config.openai_config import clientAI

# Load utility functions
from utils.embedding import get_embedding
from utils.db import get_db_connection

# Load models
from rag.app.models import RAGRequest, EmbeddingRequest

# Load RAG system prompt
from rag.app.prompts import OPENAI_RAG_SYSTEM_PROMPT_DE

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create an instance of FastAPI
app = FastAPI(**rag_app_config)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[CORS_ALLOWED_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RAGProcessor:
    def __init__(self, rag_config):
        self.rag_config = rag_config
        self.client = clientAI

    async def fetch_context_docs(self, query):
        conn = await get_db_connection()

        try:
            # Get the query embedding vector
            query_embedding = get_embedding(query)[0].embedding

            # Only supports retrieval of 1 document at the moment (set in /config/config.yaml). Will implement multi-doc retrieval later
            top_k = rag_config["retrieval"]["top_k"]
            similarity_metric = rag_config["retrieval"]["metric"]
            similarity_metric_symbol = SIMILARITY_METRICS[similarity_metric]
            docs = await conn.fetch(f"""
                SELECT text, url,  1 - (embedding {similarity_metric_symbol} '{query_embedding}') AS similarity_score
                FROM embeddings
                ORDER BY similarity_score desc
                LIMIT {'NULL' if top_k==0 else top_k}
            """)

        except Exception as e:
            logger.error(f"Error fetching context docs: {e}")
            raise HTTPException(status_code=500, detail=str(e)) # Define what exceptions to catch

        finally:
            await conn.close()

        return [dict(row) for row in docs][0]

    def create_openai_message(self, context_docs, query):
        openai_rag_system_prompt = OPENAI_RAG_SYSTEM_PROMPT_DE.format(context_docs=context_docs, query=query)
        return [{"role": "system", "content": openai_rag_system_prompt},]

    def create_openai_stream(self, messages):
        return self.client.chat.completions.create(
            model=self.rag_config["llm"]["model"],
            messages=messages,
            max_tokens=self.rag_config["llm"]["max_output_tokens"],
            stream=self.rag_config["llm"]["stream"],
            temperature=self.rag_config["llm"]["temperature"],
            top_p=self.rag_config["llm"]["top_p"]
        )

    def generate(self, openai_stream, source_url):
        for chunk in openai_stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content.encode("utf-8")
            else:
                # Send a special token indicating the end of the response
                yield f"\n\n<a href='{source_url}' target='_blank' class='source-link'>{source_url}</a>".encode("utf-8")
                break

processor = RAGProcessor(rag_config)

@app.post("/rag/process", summary="Process RAG query endpoint", response_description="Return result from processing RAG query", status_code=200)
async def process_query(request: RAGRequest):
    docs = await processor.fetch_context_docs(request.query)
    context_docs = docs['text']
    source_url = docs['url']
    messages = processor.create_openai_message(context_docs, request.query)
    openai_stream = processor.create_openai_stream(messages)

    return StreamingResponse(processor.generate(openai_stream, source_url), media_type="text/event-stream")

@app.post("/rag/docs", summary="Retrieve context docs endpoint", response_description="Return context docs from semantic search", status_code=200)
async def docs(request: RAGRequest):
    docs = await processor.fetch_context_docs(request.query)
    return {"contextDocs": docs["text"], "sourceUrl": docs["url"], "similarityScore": docs["similarity_score"]}

@app.post("/rag/embed", summary="Embedding endpoint", response_description="A dictionary with embeddings for the input text")
async def embed(text_input: EmbeddingRequest):
    try:
        embedding = get_embedding(text_input.text)[0].embedding
        return {"data": embedding}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rag/rerank", summary="Reranking endpoint", response_description="Welcome Message")
async def rerank():
    """
    Dummy endpoint for retrieved docs reranking.
    """
    return Response(content="Not Implemented", status_code=status.HTTP_501_NOT_IMPLEMENTED)