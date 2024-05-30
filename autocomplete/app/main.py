import logging

import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pyxdameraulevenshtein import damerau_levenshtein_distance

# Load env variables
from config.base_config import autocomplete_config, autocomplete_app_config
from config.network_config import CORS_ALLOWED_ORIGINS
from config.pgvector_config import SIMILARITY_METRICS

# Load utility functions
from utils.db import get_db_connection
from utils.embedding import get_embedding

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create an instance of FastAPI
app = FastAPI(**autocomplete_app_config)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[CORS_ALLOWED_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def get_exact_match(question: str):
    """
    Search for questions that contain the exact specified string, case-insensitive.

    - **question**: string to be searched within the questions.

    Returns a list of questions that exactly match the search criteria.
    """
    conn = await get_db_connection()
    try:
        # Convert both the 'question' column and the search string to lowercase to perform a case-insensitive search
        search_query = f"%{question.lower()}%"

        # Fetch results from the database
        max_results = autocomplete_config["exact_match"]["limit"]

        if max_results == -1:
            rows = await conn.fetch("SELECT * FROM data WHERE LOWER(question) LIKE $1", search_query)
        else:
            rows = await conn.fetch("SELECT * FROM data WHERE LOWER(question) LIKE $1 LIMIT $2", search_query, max_results)

        await conn.close()  # Close the database connection

        # Convert the results to a list of dictionaries
        matches = [dict(row) for row in rows]
        return matches
    except Exception as e:
        await conn.close()
        raise HTTPException(status_code=500, detail=str(e)) from e

async def get_fuzzy_match(question: str):
    """
    Search for questions with fuzzy match (levenstein-damerau distance) based on threshold, case-insensitive.

    - **question**: string to be searched within the questions.

    Returns a list of questions that match the search criteria if within the specified threshold.
    """
    conn = await get_db_connection()
    try:
        # Fetch all rows from the database
        rows = await conn.fetch("SELECT * FROM data")
        await conn.close()  # Close the database connection

        # Convert the question to lowercase
        question = question.lower()

        # Perform fuzzy matching
        matches = []
        for row in rows:
            # Convert the 'question' column to lowercase
            row_question = row['question'].lower()

            # Calculate the Levenshtein-Damerau distance
            distance = damerau_levenshtein_distance(question, row_question)

            # If the distance is above a certain threshold, add the row to the matches
            threshold = autocomplete_config["fuzzy_match"]["threshold"]
            if distance <= threshold:
                matches.append((distance, row))

        # Sort the matches by distance in ascending order
        matches = sorted(matches, key=lambda x: x[0])

        # Extract the rows from the matches
        matches = [match[1] for match in matches]

        max_results = autocomplete_config["fuzzy_match"]["limit"]
        matches = matches[:max_results] if max_results != -1 else matches

        # Convert the results to a list of dictionaries
        matches = [dict(row) for row in matches]
        return matches

    except Exception as e:
        await conn.close()
        raise HTTPException(status_code=500, detail=str(e)) from e

async def get_semantic_similarity_match(question: str):
    """
    Search for questions with cosine (semantic) similarity using an embedding model, case-insensitive.

    - **question**: string to be searched within the questions.

    Returns a list of 5 most similar questions based on cosine similarity.
    TO BE IMPLEMENTED: Returns a top_k list of questions that match the search criteria based on cosine similarity.
    """
    conn = await get_db_connection()

    try:
        # Get embedding vector for question
        question_embedding = get_embedding(question)[0]["embedding"]

        # Fetch the most similar questions based on cosine similarity
        similarity_metric = autocomplete_config["semantic_similarity_match"]["metric"]
        similarity_metric_symbol = SIMILARITY_METRICS[similarity_metric]
        max_results = autocomplete_config["semantic_similarity_match"]["limit"]

        if max_results == -1:
            matches = await conn.fetch(f"""
                SELECT question, answer, url,  1 - (embedding {similarity_metric_symbol} '{question_embedding}') AS {similarity_metric}
                FROM faq_embeddings
                ORDER BY {similarity_metric} desc
            """)
        else:
            matches = await conn.fetch(f"""
                SELECT question, answer, url,  1 - (embedding {similarity_metric_symbol} '{question_embedding}') AS {similarity_metric}
                FROM faq_embeddings
                ORDER BY {similarity_metric} desc
                LIMIT $1
            """, max_results)

        await conn.close() # Close the database connection

        # Convert the results to a list of dictionaries
        matches = [{"question": row[0],
                    "answer": row[1],
                    "url": row[2]} for row in matches]

        return matches

    except Exception as e:
        await conn.close()
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/autocomplete/", summary="Facade for autocomplete", response_description="List of matching questions")
async def autocomplete(question: str):
    """
     If combined results of get_exact_match() and get_fuzzy_match() return less than 5 results, this method is called after every new "space" character in the question (user query) is added as well as when a "?" character is added at the end of the question.
    """
    exact_match_results, fuzzy_match_results = await asyncio.gather(
        get_exact_match(question),
        get_fuzzy_match(question),
    )

    # Combine the results
    combined_matches = exact_match_results + fuzzy_match_results

    # If the combined results from exact match and fuzzy match are less than 5, get semantic similarity matches
    if len(combined_matches) < 5 and (question[-1] == " " or question[-1] == "?"):

        semantic_similarity_match_results = await get_semantic_similarity_match(question)

        # Combine the results
        combined_matches += semantic_similarity_match_results

    # Remove duplicates
    unique_matches = []
    [unique_matches.append(i) for i in combined_matches if i not in unique_matches]

    # Truncate the list to max_results
    max_results = autocomplete_config["results"]["limit"]
    if max_results != -1:
        unique_matches = unique_matches[:max_results]

    return unique_matches

@app.get("/autocomplete/exact_match/", summary="Search Questions with exact match", response_description="List of matching questions")
async def exact_match(question: str):
    return await get_exact_match(question)

@app.get("/autocomplete/fuzzy_match/", summary="Search Questions with fuzzy match", response_description="List of matching questions")
async def fuzzy_match(question: str):
    return await get_fuzzy_match(question)

@app.get("/autocomplete/semantic_similarity_match/", summary="Search Questions with semantic similarity match", response_description="List of matching questions")
async def semantic_similarity_match(question: str):
    return await get_semantic_similarity_match(question)
