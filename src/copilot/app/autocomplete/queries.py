from typing import List
from utils.embedding import get_embedding
from utils.db import get_db_connection


async def fetch(db_name: str,
                select: List[str] = None,
                where: List[str] = None,
                language: str = None,
                order: str = None,
                k: int = 0):
    conn = await get_db_connection()

    selection = ', '.join(['id', 'question', 'answer', 'url', 'language'] + (select if select else [])) if db_name != 'embeddings' else ', '.join(['text, url'] + (select if select else []))
    conditions = []
    if language:
        conditions.append(f'language = {language}')
    if where:
        conditions += where
    where_clause = ('WHERE ' + ' AND '.join(conditions)) if conditions else ''
    order_clause = f'ORDER BY {order}' if order else ''
    limit_clause = f'LIMIT {k}' if k > 0 else ''

    query = f"""
            SELECT {selection}
            FROM {db_name}
            {where_clause}
            {order_clause}
            {limit_clause}
        """

    try:
        rows = await conn.fetch(query)

    finally:
        await conn.close()

    return rows


def exact_match(question: str, language: str = None, k: int = 0):
    return fetch(db_name='data',
                 where=[f"LOWER(question) LIKE '{question}'"],
                 language=language,
                 k=k)


def fuzzy_match(question: str, threshold, language: str = None, k: int = 0):
    return fetch(db_name='data',
                 where=[f"levenshtein_less_equal('{question}', question, {threshold}) < {threshold}"],
                 language=language,
                 order=f"levenshtein(question, '{question}') asc",
                 k=k)


def exact_or_fuzzy(question: str, threshold, language: str = None, k: int = 0):
    return fetch(db_name='data',
                 where=[f"LOWER(question) LIKE '{question}' OR levenshtein_less_equal('{question}', question, {threshold}) < {threshold}"],
                 language=language,
                 order=f"levenshtein(question, '{question}') asc",
                 k=k)


def semantic_similarity_match(question: str,
                              db_name: str = 'faq_embeddings',
                              language: str = None,
                              symbol: str = '<=>',
                              k: int = 0):

    question_embedding = get_embedding(question)

    return fetch(db_name=db_name,
                 select=[f"1 - (embedding {symbol} '{question_embedding}') AS similarity_metric"],
                 language=language,
                 order="similarity_metric desc",
                 k=k)


def semantic_similarity_match_l1(question: str, language: str = None, k: int = 0):
    return semantic_similarity_match(question, language, symbol='<+>', k=k)


def semantic_similarity_match_l2(question: str, language: str = None, k: int = 0):
    return semantic_similarity_match(question, language, symbol='<->', k=k)


def semantic_similarity_match_inner_prod(question: str, language: str = None, k: int = 0):
    return semantic_similarity_match(question, language, symbol='<#>', k=k)
