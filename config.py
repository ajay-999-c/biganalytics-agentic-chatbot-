import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LLM Settings
USE_OPENSOURCE_LLM = False
USE_HYBRID_LLM = True  # True: Gemini+Cloudflare, False: Gemini only

# Ollama Model Configuration (for local testing)
LLM_MODEL_NAME = "llama3.2"  # Default Ollama model

# Cloudflare API (loaded from .env file)
CLOUDFLARE_ACCOUNT_ID = os.getenv('CLOUDFLARE_ACCOUNT_ID')
CLOUDFLARE_API_TOKEN = os.getenv('CLOUDFLARE_API_TOKEN')


EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # Upgraded to larger model

# Reranker Model Configuration (for enhanced RAG)
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-12-v2"

# Version tracking for automatic rebuilds
EMBEDDING_MODEL_VERSION = "v3.0-chromadb-mpnet"  # Updated for ChromaDB migration

# Vector Database Configuration
# FAISS (old) - keeping for backward compatibility
VECTOR_DB_PATH = "faiss_index"

# ChromaDB (new enhanced system)
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "bignalytics_advanced"

KNOWLEDGE_BASE_FILE = "knowledge_base.txt"