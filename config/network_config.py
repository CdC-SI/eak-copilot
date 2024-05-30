import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load CORS config
CORS_ALLOWED_ORIGINS = os.environ["CORS_ALLOWED_ORIGINS"]