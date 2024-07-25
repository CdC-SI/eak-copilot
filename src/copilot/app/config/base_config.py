import os
from dotenv import load_dotenv
from pyaml_env import parse_config
import yaml

# Load check_env_vars function
from utils.check_env import check_env_vars

# Load environment variables from .env file
load_dotenv()

# Path to the YAML configuration file
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')

# Load the YAML configuration file
config = parse_config(CONFIG_PATH)

# Check env variable values from config/config.yaml
check_env_vars(config)

# Access configuration values
autocomplete_config = config.get('autocomplete')
rag_config = config.get('rag')
indexing_config = config.get('indexing')

# Access fastapi app configuration values
# Path to the YAML configuration file
APP_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'app_config.yaml')

# Load the YAML configuration file
with open(APP_CONFIG_PATH, 'r') as file:
    app_config = yaml.safe_load(file)

autocomplete_app_config = app_config.get('autocomplete')
indexing_app_config = app_config.get('indexing')
rag_app_config = app_config.get('rag')
