import os
from dotenv import load_dotenv



load_dotenv()

# OPEN ROUTER
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_URL = os.environ.get("OPENROUTER_URL")

# OPENAI
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_EMBEDDING = os.environ.get("OPENAI_EMBEDDING")

# QDRANT
QDRANT_ADDRESS = os.environ.get("QDRANT_ADDRESS")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

# LANGFUSE
LANGFUSE_URL = os.environ.get("LANGFUSE_URL")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY")