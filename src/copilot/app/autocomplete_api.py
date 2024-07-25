import logging

from autocomplete.autocompleter import autocompleter
from autocomplete.matching import exact_matcher, fuzzy_matcher, semantic_matcher

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config.network_config import CORS_ALLOWED_ORIGINS

# Load env variables
from config.base_config import autocomplete_app_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create required class instances
app = FastAPI(**autocomplete_app_config)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[CORS_ALLOWED_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/",
         summary="Facade for autocomplete",
         response_description="List of matching questions")
async def autocomplete(question: str, language: str = None):
    """
    If combined results of get_exact_match() and get_fuzzy_match() return less than 5 results,
    this method is called after every new "space" character in the question (user query) is
    added as well as when a "?" character is added at the end of the question.

    Parameters
    ----------
    question : str
       User input to match database entries
    language : str, optional
        Question and results language

    Return
    ------
    list of dict
    """
    return await autocompleter.get_autocomplete(question, language)


@app.get("/exact_match",
         summary="Search Questions with exact match",
         response_description="List of matching questions")
async def exact_match(question: str, language: str = None):
    """
    Return results from Exact matching

    Parameters
    ----------
    question : str
       User input to match database entries
    language : str, optional
        Question and results language

    Return
    ------
    list of dict
    """
    return await exact_matcher.match(question, language)


@app.get("/fuzzy_match",
         summary="Search Questions with fuzzy match",
         response_description="List of matching questions")
async def fuzzy_match(question: str, language: str = None):
    """
    Return results from Fuzzy matching

    Parameters
    ----------
    question : str
       User input to match database entries
    language : str, optional
        Question and results language

    Return
    ------
    list of dict
    """
    return await fuzzy_matcher.match(question, language)


@app.get("/semantic_similarity_match",
         summary="Search Questions with semantic similarity match",
         response_description="List of matching questions")
async def semantic_similarity_match(question: str, language: str = None):
    """
    Return results from Semantic Similarity matching

    Parameters
    ----------
    question : str
       User input to match database entries
    language : str, optional
        Question and results language

    Return
    ------
    list of dict
    """
    return await semantic_matcher.match(question, language)
