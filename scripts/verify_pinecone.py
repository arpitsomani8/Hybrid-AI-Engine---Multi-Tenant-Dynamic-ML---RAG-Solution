import os
import sys
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")
print(f"Loaded Pinecone API key: {api_key[:12]}... (length: {len(api_key)})")

try:
    from pinecone import Pinecone, ServerlessSpec
    pc = Pinecone(api_key=api_key)
    
    # List active indexes
    indexes = pc.list_indexes()
    print("Pinecone connection successful!")
    print(f"Existing indexes: {[idx.name for idx in indexes]}")
    
except Exception as e:
    print(f"Pinecone verification error: {e}", file=sys.stderr)
    sys.exit(1)
