#!/usr/bin/env python3
"""Test query directly through the retriever."""

import sys

sys.path.append('/app')

from backend.retriever import Retriever
import json


def test_direct_query():
    """Test the retriever directly."""
    print("=== DIRECT RETRIEVER TEST ===")

    retriever = Retriever()

    # Test simple query
    test_queries = [
        "combat rules",
        "how to roll dice",
        "initiative",
        "what is shadowrun"
    ]

    for query in test_queries:
        print(f"\n🔍 Testing: '{query}'")

        try:
            result = retriever.query(
                question=query,
                n_results=3,
                query_type="general"
            )

            print(f"✅ Answer length: {len(result.get('answer', ''))}")
            print(f"✅ Sources: {len(result.get('sources', []))}")
            print(f"✅ Chunks: {len(result.get('chunks', []))}")

            # Show first part of answer
            answer = result.get('answer', '')
            if answer:
                print(f"📝 Answer preview: {answer[:200]}...")
            else:
                print("❌ Empty answer!")

        except Exception as e:
            print(f"❌ Query failed: {e}")


def test_search_only():
    """Test just the search component."""
    print("\n=== SEARCH ONLY TEST ===")

    retriever = Retriever()

    search_result = retriever.search(
        question="combat rules",
        n_results=3
    )

    print(f"Documents found: {len(search_result['documents'])}")
    print(f"Metadata entries: {len(search_result['metadatas'])}")

    if search_result['documents']:
        print("✅ Search working!")
        for i, doc in enumerate(search_result['documents'][:2]):
            print(f"  {i + 1}. {doc[:100]}...")
    else:
        print("❌ Search returned no results!")


if __name__ == "__main__":
    test_search_only()
    test_direct_query()