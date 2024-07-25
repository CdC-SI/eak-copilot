from utils.db import get_db_connection


async def fetchone(db_name: str, select: [str] = None, where: [str] = None):
    conn = await get_db_connection()

    selection = ', '.join(['id', 'question', 'answer', 'url'] + ([] if select is None else select))
    conditions = ' AND '.join(where)

    try:
        rows = await conn.fetchrow(f"""
            SELECT {selection}
            FROM {db_name}
            WHERE {conditions}
        """)

    finally:
        await conn.close()

    return rows


async def get_one(db_name: str, question: str):
    search_query = f"%{question.lower()}%"

    return await fetchone(db_name,
                          where=[f"LOWER(question) LIKE '{search_query}'"])


async def update_data(url: str,
                      question: str,
                      answer: str,
                      language: str,
                      id: int):
    conn = await get_db_connection()

    try:
        await conn.execute(f"""
            UPDATE data
            SET url = '{url}', question = '{question}', answer = '{answer}', language = '{language}' WHERE id = {id}
        """)

    finally:
        await conn.close()

    return id


async def insert_data(url: str,
                      question: str,
                      answer: str,
                      language: str):
    conn = await get_db_connection()

    try:
        return await conn.fetchval(f"""
            INSERT
            INTO data (url, question, answer, language)
            VALUES ('{url}', '{question}', '{answer}', '{language}')
            RETURNING id
        """)

    finally:
        await conn.close()


async def update_or_insert(url: str,
                           question: str,
                           answer: str,
                           language: str):
    existing_row = await get_one('data', question)

    if existing_row:
        # Update the existing record
        rid = await update_data(url, question, answer, language, existing_row['id'])
        return "Update", rid

    else:
        # Insert a new record
        rid = await insert_data(url, question, answer, language)
        return "Insert", rid


async def insert_faq(url: str,
                     question: str,
                     answer: str,
                     language: str,
                     embedding: str):
    conn = await get_db_connection()

    try:
        await conn.execute("""
            INSERT
            INTO faq_embeddings (url, question, answer, language, embedding)
            VALUES ($1, $2, $3, $4, $5)
        """, url, question, answer, language, embedding)

    finally:
        await conn.close()


async def insert_rag(embedding: str,
                     text: str,
                     url: str):
    conn = await get_db_connection()

    try:
        if embedding is None:
            await conn.execute("""
                INSERT
                INTO embeddings (text, url)
                VALUES ($1, $2)
            """, text, url)
        else:
            await conn.execute("""
                INSERT
                INTO embeddings (embedding, text, url)
                VALUES ($1, $2, $3)
            """, embedding, text, url)

    finally:
        await conn.close()
