import os
from dotenv import load_dotenv
import openai
import logging

# Load environment variables from .env file
load_dotenv()

# Load OpenAI API key
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
HTTP_PROXY = os.environ.get("HTTP_PROXY", None)
REQUESTS_CA_BUNDLE = os.environ.get("REQUESTS_CA_BUNDLE", None)

# if HTTP_PROXY then set the proxy in openai client
if HTTP_PROXY and REQUESTS_CA_BUNDLE:
    logger = logging.getLogger(__name__)
    logger.info(f"Setting up HTTP_PROXY: {HTTP_PROXY}")
    logger.info(f"Setting up REQUESTS_CA_BUNDLE: {REQUESTS_CA_BUNDLE}")

    import httpx
    clientAI = openai.OpenAI(
        http_client=httpx.Client(proxy=HTTP_PROXY, verify=REQUESTS_CA_BUNDLE),
        api_key=OPENAI_API_KEY
    )
else:
    clientAI = openai.OpenAI(
        api_key=OPENAI_API_KEY
    )
