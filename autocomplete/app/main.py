import asyncpg
import logging
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import httpx

from pyxdameraulevenshtein import damerau_levenshtein_distance

from autocomplete.app.web_scraper import WebScraper

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create an instance of FastAPI
app = FastAPI()

# Load env variables
from config import DB_PARAMS, CORS_ALLOWED_ORIGINS

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[CORS_ALLOWED_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def get_db_connection():
    """Establish a database connection."""
    conn = await asyncpg.connect(**DB_PARAMS)
    return conn

async def get_exact_match(question: str):
    """
    Search for questions that contain the exact specified string, case-insensitive.

    :param question: string to be searched within the questions.
    :type question: str

    :return: a list of questions that exactly match the search criteria.
    :rtype: list of str
    """
    conn = await get_db_connection()
    try:
        # Convert both the 'question' column and the search string to lowercase to perform a case-insensitive search
        search_query = f"%{question.lower()}%"
        rows = await conn.fetch("SELECT * FROM data WHERE LOWER(question) LIKE $1", search_query)
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

    :param question: string to be searched within the questions.
    :rtype: list of str

    :return: a list of questions that match the search criteria if within the specified threshold.
    :rtype: list of str
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
            if distance <= 5:
                matches.append(row)

        # Convert the results to a list of dictionaries
        matches = [dict(row) for row in matches]
        return matches

    except Exception as e:
        await conn.close()
        raise HTTPException(status_code=500, detail=str(e)) from e

async def get_semantic_similarity_match(question: str):
    """
    Search for questions with cosine (semantic) similarity using an embedding model, case-insensitive.

    :param question: string to be searched within the questions.
    :type question: str

    :return: a list of 5 most similar questions based on cosine similarity.
    :rtype: list of str

    .. todo::
        Returns a top_k list of questions that match the search criteria based on cosine similarity.
    """
    conn = await get_db_connection()

    try:
        # Make POST request to the /embed API endpoint to get the embedding
        async with httpx.AsyncClient() as client:
            response = await client.post("http://rag:8010/rag/embed", json={"text": question})

        # Ensure the request was successful
        response.raise_for_status()

        # Get the resulting embedding vector from the response
        question_embedding = response.json()["data"][0]["embedding"]

        matches = await conn.fetch(f"""
            SELECT question, answer, url,  1 - (embedding <=> '{question_embedding}') AS cosine_similarity
            FROM faq_embeddings
            ORDER BY cosine_similarity desc
            LIMIT 5
        """)

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
    Get matching results from exact matching and fuzzy matching.
    If combined results of get_exact_match() and get_fuzzy_match() return less than 5 results, this method is called after every new "space" character in the question (user query) is added as well as when a "?" character is added at the end of the question.

    :param question: string to be searched within the questions.
    :type question: str

    :return: a list of at least 5 most similar questions.
    :rtype: list of str
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

@app.put("/autocomplete/data/", summary="Update or Insert Data", response_description="Updated or Inserted Data")
async def update_or_insert_data(
    url: str,
    question: str,
    answer: str,
    language: str,
    id: int = Body(None)
):
    """
    Update an existing data record or insert a new one based on the presence of the question.

    :param url: The URL associated with the data.
    :type url: str
    :param question: The question.
    :type question: str
    :param answer: The answer.
    :type answer: str
    :param language: The language of the data.
    :type language: str

    :return: the updated or inserted data record.
    :rtype: str

    .. note::
        The operation now checks for the presence of a question in the database to decide between insert and update.
    """
    conn = await get_db_connection()
    try:
        # Convert the search question to lowercase to perform a case-insensitive search
        search_query = f"%{question.lower()}%"
        # Check if a record with the same question exists
        existing_row = await conn.fetchrow("SELECT * FROM data WHERE LOWER(question) LIKE $1", search_query)

        if existing_row:
            # Update the existing record
            try:
                await conn.execute(
                    "UPDATE data SET url = $1, question = $2, answer = $3, language = $4 WHERE id = $5",
                    url, question, answer, language, existing_row['id']
                )
                logger.info(f"Update: {url}")
                return {"id": existing_row['id'], "url": url, "question": question, "answer": answer, "language": language}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Update exception: {str(e)}") from e
        else:
            # Insert a new record
            try:
                row = await conn.fetchrow(
                    "INSERT INTO data (url, question, answer, language) VALUES ($1, $2, $3, $4) RETURNING id",
                    url, question, answer, language
                )
                logger.info(f"Insert: {url}")
                return {"id": row['id'], "url": url, "question": question, "answer": answer, "language": language}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Insert exception: {str(e)}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        await conn.close()

@app.put("/autocomplete/init_expert/", summary="Insert Data from faq.bsv.admin.ch", response_description="Insert Data from faq.bsv.admin.ch")
async def init_expert():
    """
    Asynchronously retrieves and processes FAQ data from 'https://faq.bsv.admin.ch' to insert into the database.

    The endpoint 'https://faq.bsv.admin.ch/sitemap.xml' is used to discover all relevant FAQ URLs. For each URL,
    the method extracts the primary question (denoted by the 'h1' tag) and its corresponding answer (within an 'article' tag).
    Unnecessary boilerplate text will be removed for clarity and conciseness.

    Each extracted FAQ entry is then upserted (inserted or updated if already exists) into the database, with detailed
    logging to track the operation's progress and identify any errors.

    :return: a confirmation message upon successful completion of the process.
    :rtype: str

    .. todo::
        - Consider implementing error handling at a more granular level to retry failed insertions or updates, enhancing the robustness of the data ingestion process.
        - Explore optimization opportunities in text extraction and processing to improve efficiency and reduce runtime, especially for large sitemaps.
    """
    logging.basicConfig(level=logging.INFO)

    sitemap_url = 'https://faq.bsv.admin.ch/sitemap.xml'
    scraper = WebScraper(sitemap_url)

    scraper.logger.info(f"Beginne Datenextraktion für: {sitemap_url}")
    urls = scraper.get_sitemap_urls()

    for url in urls:
        extracted_h1 = scraper.extract_text_from_tag(url, 'h1')
        extracted_article = scraper.extract_text_from_tag(url, 'article')
        language = scraper.detect_language(url)

        # Efficient text processing
        for term in ['Antwort\n', 'Rispondi\n', 'Réponse\n']:
            extracted_article = extracted_article.replace(term, '')

        if extracted_h1 and extracted_article:
            try:
                logger.info(f"extract: {url}")
                await update_or_insert_data(
                    url=url,
                    question=extracted_h1,
                    answer=extracted_article,
                    language=language,
                    id=None
                )
            except Exception as e:
                logger.error(f"Error: {e}")

    logger.info(f"Done! {len(urls)} wurden verarbeitet.")
    return {"message": f"Done! {len(urls)} wurden verarbeitet."}
