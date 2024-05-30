import logging
import time
import asyncpg
from fastapi import HTTPException

# Import env vars
from config.db_config import DB_PARAMS

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Function to create a db connection
async def get_db_connection():
    """Establish a database connection."""
    conn = await asyncpg.connect(**DB_PARAMS)
    return conn

# Function to check if db is up
async def check_db_connection(retries: int = 5, delay: int = 5):
    """Check if database is up."""
    for _ in range(retries):
        try:
            conn = await get_db_connection()
            await conn.close()
            logger.info("Database connection successful.")
            return
        except Exception as e:
            logger.warning("Database connection failed: %s. Retrying in %s seconds...", e, delay)
            time.sleep(delay)
    raise HTTPException(status_code=500, detail="Failed to connect to the database after %s attempts." % retries)
