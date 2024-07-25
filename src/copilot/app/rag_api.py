import logging

from fastapi import FastAPI, status
from fastapi.responses import Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from config.network_config import CORS_ALLOWED_ORIGINS

# Load env variables
from config.base_config import rag_app_config, rag_config
from config.openai_config import clientAI

# Load models
from rag.rag_processor import RAGProcessor
from rag.models import RAGRequest

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create required instances
processor = RAGProcessor(model=rag_config["llm"]["model"],
                         max_token=rag_config["llm"]["max_output_tokens"],
                         stream=rag_config["llm"]["stream"],
                         temperature=rag_config["llm"]["temperature"],
                         top_p=rag_config["llm"]["top_p"],
                         top_k=rag_config["retrieval"]["top_k"],
                         client=clientAI)

app = FastAPI(**rag_app_config)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[CORS_ALLOWED_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/model",
         status_code=200)
def model():
    return {"model": processor.model, "rag_config": rag_config}


@app.post("/query",
          summary="Process RAG query endpoint",
          response_description="Return result from processing RAG query",
          status_code=200)
async def process_query(request: RAGRequest):
    content = await processor.process(request)
    return StreamingResponse(content, media_type="text/event-stream")


@app.post("/context_docs",
          summary="Retrieve context docs endpoint",
          response_description="Return context docs from semantic search",
          status_code=200)
async def docs(request: RAGRequest, language: str = None):
    return await processor.retrieve(request, language)


@app.get("/rerank",
         summary="Reranking endpoint",
         response_description="Welcome Message")
async def rerank():
    """
    Dummy endpoint for retrieved docs reranking.
    """
    return Response(content="Not Implemented", status_code=status.HTTP_501_NOT_IMPLEMENTED)
