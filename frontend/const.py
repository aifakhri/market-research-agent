import os
from dotenv import load_dotenv



load_dotenv()

# OPEN ROUTER
OPEN_ROUTER_MODEL = os.environ.get("OPEN_ROUTER_MODEL")
OPEN_ROUTER_API_KEY = os.environ.get("OPEN_ROUTER_API_KEY")
OPEN_ROUTER_URL = os.environ.get("OPEN_ROUTER_URL")

# OPENAI
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_EMBEDDING = os.environ.get("OPENAI_EMBEDDING")

# QDRANT
QDRANT_ADDRESS = os.environ.get("QDRANT_ADDRESS")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION")