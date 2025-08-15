#!/usr/bin/env python3
"""Debug script for Shadowrun RAG system issues."""

import os
import sys
from pathlib import Path
import chromadb
from chromadb.config import Settings
import ollama
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_file_structure():
    """Check if files exist in expected locations."""
    print("=== FILE STRUCTURE CHECK ===")

    paths_to_check = [
        "data/raw_pdfs",
        "data/processed_markdown",
        "data/chroma_db"
    ]

    for path in paths_to_check:
        p = Path(path)
        exists = p.exists()
        print(f"📁 {path}: {'✅' if exists else '❌'}")

        if exists and p.is_dir():
            files = list(p.rglob("*"))
            print(f"   └── Files: {len(files)}")
            if files:
                for f in files[:5]:  # Show first 5
                    print(f"       - {f.name}")
                if len(files) > 5:
                    print(f"       ... and {len(files) - 5} more")


def check_chromadb():
    """Check ChromaDB collection status."""
    print("\n=== CHROMADB CHECK ===")

    try:
        chroma_path = os.getenv("CHROMA_DB_PATH", "data/chroma_db")
        collection_name = os.getenv("COLLECTION_NAME", "shadowrun_docs")

        client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False)
        )

        print(f"📊 ChromaDB path: {chroma_path}")
        print(f"📊 Collection name: {collection_name}")

        # List collections
        collections = client.list_collections()
        print(f"📊 Available collections: {[c.name for c in collections]}")

        if collections:
            collection = client.get_collection(collection_name)
            count = collection.count()
            print(f"📊 Documents in '{collection_name}': {count}")

            if count > 0:
                # Sample some data
                sample = collection.get(limit=3, include=['documents', 'metadatas'])
                print("📊 Sample documents:")
                for i, (doc, meta) in enumerate(zip(sample['documents'], sample['metadatas'])):
                    print(f"   {i + 1}. Source: {meta.get('source', 'Unknown')}")
                    print(f"      Content: {doc[:100]}...")
                    print(f"      Metadata: {meta}")
            else:
                print("❌ Collection is empty!")
        else:
            print("❌ No collections found!")

    except Exception as e:
        print(f"❌ ChromaDB error: {e}")


def check_ollama():
    """Check Ollama models and connectivity."""
    print("\n=== OLLAMA CHECK ===")

    try:
        # List models
        models = ollama.list()
        print(f"🤖 Available models: {[m['name'] for m in models.get('models', [])]}")

        # Test embedding model
        embedding_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
        print(f"🤖 Testing embedding model: {embedding_model}")

        test_text = "This is a test sentence for embeddings."
        response = ollama.embeddings(model=embedding_model, prompt=test_text)

        if 'embedding' in response:
            embedding_dim = len(response['embedding'])
            print(f"✅ Embedding test successful! Dimension: {embedding_dim}")
        else:
            print(f"❌ Embedding response invalid: {response}")

    except Exception as e:
        print(f"❌ Ollama error: {e}")
        print("   Make sure Ollama container is running and models are pulled")


def test_search():
    """Test a simple search query."""
    print("\n=== SEARCH TEST ===")

    try:
        chroma_path = os.getenv("CHROMA_DB_PATH", "data/chroma_db")
        collection_name = os.getenv("COLLECTION_NAME", "shadowrun_docs")
        embedding_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

        client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False)
        )
        collection = client.get_collection(collection_name)

        test_query = "combat rules"
        print(f"🔍 Testing query: '{test_query}'")

        # Get embedding
        response = ollama.embeddings(model=embedding_model, prompt=test_query)
        query_embedding = response['embedding']

        # Search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )

        print(f"🔍 Search results: {len(results['documents'][0])} documents found")

        if results['documents'][0]:
            for i, (doc, dist) in enumerate(zip(results['documents'][0], results['distances'][0])):
                print(f"   {i + 1}. Distance: {dist:.3f}")
                print(f"      Content: {doc[:150]}...")
        else:
            print("❌ No search results!")

    except Exception as e:
        print(f"❌ Search test failed: {e}")


def check_env_vars():
    """Check environment variables."""
    print("\n=== ENVIRONMENT VARIABLES ===")

    env_vars = [
        "OLLAMA_HOST",
        "EMBEDDING_MODEL",
        "LLM_MODEL",
        "CHROMA_DB_PATH",
        "COLLECTION_NAME"
    ]

    for var in env_vars:
        value = os.getenv(var, "NOT SET")
        print(f"🔧 {var}: {value}")


if __name__ == "__main__":
    print("🎲 Shadowrun RAG Debug Tool\n")

    check_env_vars()
    check_file_structure()
    check_ollama()
    check_chromadb()
    test_search()

    print("\n=== RECOMMENDATIONS ===")
    print(
        "1. If ChromaDB is empty, run: docker exec shadowrun-backend python -c 'from backend.indexer import IncrementalIndexer; IncrementalIndexer().index_directory(\"data/processed_markdown\", force_reindex=True)'")
    print("2. If Ollama fails, check: docker logs shadowrun-ollama")
    print("3. If processed_markdown is empty, check PDF processing logs")
    print("4. For embedding issues, try: docker exec shadowrun-ollama ollama pull nomic-embed-text")