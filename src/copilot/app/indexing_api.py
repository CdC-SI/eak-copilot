import logging

from fastapi import FastAPI, status
from fastapi.responses import Response
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from config.network_config import CORS_ALLOWED_ORIGINS

# Load env variables
from config.base_config import indexing_config, indexing_app_config

# Load utility functions
from utils.db import check_db_connection
from indexing.implementations.admin import admin_indexer
from indexing.implementations.ahv import ahv_indexer
from indexing.scraper import Scraper
from indexing import dev_mode_data, queries

# Load models
from rag.models import ResponseBody
from indexing.models import FaqItem


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def init_indexing():
    await check_db_connection(retries=10, delay=10)

    if indexing_config["faq"]["auto_index"]:
        # With dev-mode, only index sample FAQ data
        if indexing_config["dev_mode"]:
            try:
                logger.info("Auto-indexing sample FAQ data")
                await index_faq_vectordb()
            except Exception as e:
                logger.error("Dev-mode: Failed to index sample FAQ data: %s", e)
        # If dev-mode is deactivated, scrap and index all bsv.admin.ch FAQ data
        else:
            try:
                logger.info("Auto-indexing bsv.admin.ch FAQ data")
                await index_faq_data()
            except Exception as e:
                logger.error("Failed to index bsv.admin.ch FAQ data: %s", e)

    if indexing_config["rag"]["auto_index"]:
        # With dev-mode, only index sample data
        if indexing_config["dev_mode"]:
            try:
                logger.info("Auto-indexing sample RAG data")
                await index_rag_vectordb()
            except Exception as e:
                logger.error("Failed to index sample RAG data: %s", e)
        # If dev-mode is deactivated, scrap and index all RAG data (NOTE: Will be implemented soon.)
        else:
            raise NotImplementedError("Feature is not implemented yet.")


# Create an instance of FastAPI
app = FastAPI(**indexing_app_config)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[CORS_ALLOWED_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/index_pdfs_from_sitemap",
          summary="Index memento PDFs from the https://www.ahv-iv.ch/de/Sitemap-DE sitemap",
          response_description="Confirmation message upon successful indexing",
          status_code=200,
          response_model=ResponseBody)
async def index_pdfs_from_sitemap(sitemap_url: str = "https://www.ahv-iv.ch/de/Sitemap-DE"):
    """
    Indexes PDFs from a given sitemap URL. The PDFs are scraped and their data is added to the
    embedding database. This function is specifically designed for the site "https://www.ahv-iv.ch".

    Parameters
    ----------
    sitemap_url : str, optional
        The URL of the sitemap to scrape PDFs from. Defaults to "https://www.ahv-iv.ch/de/Sitemap-DE".
    language : str, optional
        The language of the PDFs. Defaults to "de" (German).

    Returns
    -------
    ResponseBody
        A response body containing a confirmation message upon successful completion of the process.
    """
    return await ahv_indexer.index(sitemap_url)


@app.post("/index_html_from_sitemap",
          summary="Index HTML from a sitemap",
          response_description="Confirmation message upon successful indexing",
          status_code=200,
          response_model=ResponseBody)
async def index_html_from_sitemap(sitemap_url: str = "https://eak.admin.ch/eak/de/home.sitemap.xml"):
    """
    Indexes HTML from a given sitemap URL. The HTML pages are scraped and their data is added to the
    embedding database. This function is specifically designed for the site "https://eak.admin.ch".

    Parameters
    ----------
    sitemap_url : str, optional
        The URL of the sitemap to scrape HTML from. Defaults to "https://eak.admin.ch/eak/de/home.sitemap.xml".
    language : str, optional
        The language of the HTML pages. Defaults to "de" (German).

    Returns
    -------
    ResponseBody
        A response body containing a confirmation message upon successful completion of the process.
    """
    return await admin_indexer.index(sitemap_url)


@app.post("/index_rag_vectordb", summary="Insert Embedding data for RAG", response_description="Insert Embedding data for RAG", status_code=200, response_model=ResponseBody)
async def index_rag_vectordb():
    """
    Add and index test data for RAG to the embedding database.

    Returns
    -------
    str
        Confirmation message upon successful completion of the process
    """
    return await dev_mode_data.init_rag_vectordb()


@app.post("/index_faq_vectordb", summary="Insert Embedding data for FAQ autocomplete semantic similarity search", response_description="Insert Embedding data for FAQ semantic similarity search", status_code=200, response_model=ResponseBody)
async def index_faq_vectordb():
    """
    Add and index test data for Autocomplete to the FAQ database.

    Returns
    -------
    str
        Confirmation message upon successful completion of the process
    """
    return await dev_mode_data.init_faq_vectordb()


@app.get("/crawl_data", summary="Crawling endpoint", response_description="Welcome Message")
async def crawl_data():
    """
    Dummy endpoint for data crawling.
    """
    return Response(content="Not Implemented", status_code=status.HTTP_501_NOT_IMPLEMENTED)


@app.get("/scrap_data/", summary="Scraping Endpoint", response_description="Welcome Message")
async def scrap_data():
    """
    Dummy endpoint for data scraping.
    """
    return Response(content="Not Implemented", status_code=status.HTTP_501_NOT_IMPLEMENTED)


@app.get("/index_data", summary="Indexing Endpoint", response_description="Welcome Message")
async def index_data():
    """
    Dummy endpoint for data indexing.
    """
    return Response(content="Not Implemented", status_code=status.HTTP_501_NOT_IMPLEMENTED)


@app.get("/parse_faq_data", summary="FAQ Parsing Endpoint", response_description="Welcome Message")
async def parse_faq_data():
    """
    Dummy endpoint for FAQ data parsing.
    """
    return Response(content="Not Implemented", status_code=status.HTTP_501_NOT_IMPLEMENTED)


@app.get("/parse_rag_data", summary="Parsing Endpoint", response_description="Welcome Message")
async def parse_rag_data():
    """
    Dummy endpoint for data parsing (RAG).
    """
    return Response(content="Not Implemented", status_code=status.HTTP_501_NOT_IMPLEMENTED)


@app.get("/chunk_rag_data", summary="Chunking Endpoint", response_description="Welcome Message")
async def chunk_rag_data():
    """
    Dummy endpoint for data chunking (RAG).
    """
    return Response(content="Not Implemented", status_code=status.HTTP_501_NOT_IMPLEMENTED)


@app.put("/index_faq_data", summary="Insert Data from faq.bsv.admin.ch", response_description="Insert Data from faq.bsv.admin.ch")
async def index_faq_data(sitemap_url: str = 'https://faq.bsv.admin.ch/sitemap.xml', k: int = 0):
    """
    Add and index data for Autocomplete to the FAQ database. The data is obtained by scraping the website `sitemap_url`.

    Parameters
    ==========
    sitemap_url : str, default 'https://faq.bsv.admin.ch/sitemap.xml'
        the `sitemap.xml` URL of the website to scrap
    k : int, default 0
        Number of article to scrap and log to test the method.

    Returns
    -------
    str
        Confirmation message upon successful completion of the process
    """
    logging.basicConfig(level=logging.INFO)

    scraper = Scraper(sitemap_url)
    urls = await scraper.run(test=k)

    return {"message": f"Done! {len(urls)} wurden verarbeitet."}


@app.put("/data", summary="Update or Insert FAQ Data", response_description="Updated or Inserted Data")
async def index_data(item: FaqItem):
    """
    Upsert a single entry to the FAQ dataset.

    Parameters
    ----------
    item : FaqItem
        The FAQ item to insert or update :
            url : str
                URL where the entry article can be found
            question : str
                The FAQ question
            answer : str
                The question answer
            language : str
                The article language

    Returns
    -------
    dict
        The article id, url, question, answer and language upon successful completion of the process
    """
    #if the item has an id we update directly
    if item.id:
        await queries.update_data(item.url, item.question, item.answer, item.language, item.id)
        logger.info(f"Update item : {item.id} - {item.question}")
        return {"id": item.id, "url": item.url, "question": item.question, "answer": item.answer, "language": item.language}
    #if the item has no id we check if a similar question already exists anyway
    else :
        info, rid = await queries.update_or_insert(item.url, item.question, item.answer, item.language)
        logger.info(f"{info}: {item.question}")

    return {"id": rid, "url": item.url, "question": item.question, "answer": item.answer, "language": item.language}
