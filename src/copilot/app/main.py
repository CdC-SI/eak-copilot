from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config.network_config import CORS_ALLOWED_ORIGINS
from contextlib import asynccontextmanager

from indexing_api import app as indexing_app
from indexing_api import init_indexing
from autocomplete_api import app as autocomplete_app
from rag_api import app as rag_app


PREFIX = "/apy"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_indexing()
    yield

app = FastAPI(lifespan=lifespan)
# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[CORS_ALLOWED_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api = FastAPI()
app.mount(PREFIX, api)

api.mount("/indexing", indexing_app)
api.mount("/autocomplete", autocomplete_app)
api.mount("/rag", rag_app)


@api.post("/",
          summary="Hello",
          status_code=200)
async def welcome():
    return "Hello!"
