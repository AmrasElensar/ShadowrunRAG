#!/usr/bin/env python3
"""
Debug script to understand actual content patterns in your chunks
Save as: tools/debug_content.py
"""

import chromadb
from chromadb.config import Settings
import re
from collections import Counter


def debug_content_patterns():
    """Analyze actual content to understand classification patterns."""

    # Connect to ChromaDB
    client = chromadb.PersistentClient(
        path="data/chroma_db",
        settings=Settings(anonymized_telemetry=False)
    )

    collection = client.get_collection("shadowrun_docs")

    # Get sample chunks
    sample = collection.get(limit=100, include=['documents', 'metadatas'])

    print("🔍 ANALYZING ACTUAL CONTENT PATTERNS\n")

    # 1. Look for combat-related content
    combat_keywords = []
    for doc in sample['documents']:
        doc_lower = doc.lower()

        # Find combat indicators
        combat_patterns = [
            'initiative', 'damage', 'armor', 'weapon', 'attack', 'defense',
            'combat', 'melee', 'ranged', 'firearms', 'body + armor',
            'wound', 'stun', 'physical damage'
        ]

        for pattern in combat_patterns:
            if pattern in doc_lower:
                combat_keywords.append(pattern)

    print("🗡️  COMBAT KEYWORDS FOUND:")
    combat_counts = Counter(combat_keywords)
    for keyword, count in combat_counts.most_common(10):
        print(f"   {keyword}: {count} chunks")

    # 2. Look for magic-related content
    magic_keywords = []
    for doc in sample['documents']:
        doc_lower = doc.lower()

        magic_patterns = [
            'spell', 'magic', 'mage', 'astral', 'spirit', 'summoning',
            'drain', 'force', 'tradition', 'adept', 'enchanting'
        ]

        for pattern in magic_patterns:
            if pattern in doc_lower:
                magic_keywords.append(pattern)

    print("\n🪄 MAGIC KEYWORDS FOUND:")
    magic_counts = Counter(magic_keywords)
    for keyword, count in magic_counts.most_common(10):
        print(f"   {keyword}: {count} chunks")

    # 3. Look for matrix/hacking content
    matrix_keywords = []
    for doc in sample['documents']:
        doc_lower = doc.lower()

        matrix_patterns = [
            'matrix', 'hacking', 'decker', 'cyberdeck', 'program', 'ic',
            'host', 'firewall', 'data processing', 'attack rating', 'sleaze'
        ]

        for pattern in matrix_patterns:
            if pattern in doc_lower:
                matrix_keywords.append(pattern)

    print("\n💻 MATRIX KEYWORDS FOUND:")
    matrix_counts = Counter(matrix_keywords)
    for keyword, count in matrix_counts.most_common(10):
        print(f"   {keyword}: {count} chunks")

    # 4. Look for specific formatting patterns
    print("\n📄 CONTENT STRUCTURE ANALYSIS:")

    table_chunks = sum(1 for doc in sample['documents'] if '|' in doc and doc.count('|') > 5)
    print(f"   Chunks with tables: {table_chunks}")

    header_chunks = sum(1 for doc in sample['documents'] if re.search(r'^#+', doc, re.MULTILINE))
    print(f"   Chunks with headers: {header_chunks}")

    link_chunks = sum(1 for doc in sample['documents'] if '[' in doc and '](' in doc)
    print(f"   Chunks with links: {link_chunks}")

    dice_pool_chunks = sum(1 for doc in sample['documents'] if 'dice pool' in doc.lower())
    print(f"   Chunks with 'dice pool': {dice_pool_chunks}")

    test_chunks = sum(1 for doc in sample['documents'] if re.search(r'\btest\b', doc.lower()))
    print(f"   Chunks with 'test': {test_chunks}")

    # 5. Sample a few chunks to understand the structure
    print("\n📝 SAMPLE CHUNKS FOR ANALYSIS:")

    # Find a chunk with taser/stun content
    taser_chunk = None
    for i, doc in enumerate(sample['documents']):
        if any(word in doc.lower() for word in ['taser', 'stun', 'electrical']):
            taser_chunk = (i, doc)
            break

    if taser_chunk:
        idx, content = taser_chunk
        print(f"\n   TASER CHUNK #{idx} (first 500 chars):")
        print(f"   {repr(content[:500])}")

    # Find a chunk with combat content
    combat_chunk = None
    for i, doc in enumerate(sample['documents']):
        if any(word in doc.lower() for word in ['initiative', 'damage', 'attack']):
            combat_chunk = (i, doc)
            break

    if combat_chunk:
        idx, content = combat_chunk
        print(f"\n   COMBAT CHUNK #{idx} (first 500 chars):")
        print(f"   {repr(content[:500])}")

    # Find a chunk with magic content
    magic_chunk = None
    for i, doc in enumerate(sample['documents']):
        if any(word in doc.lower() for word in ['spell', 'magic', 'astral']):
            magic_chunk = (i, doc)
            break

    if magic_chunk:
        idx, content = magic_chunk
        print(f"\n   MAGIC CHUNK #{idx} (first 500 chars):")
        print(f"   {repr(content[:500])}")

    print("\n" + "=" * 80)
    print("📊 RECOMMENDATIONS BASED ON ANALYSIS:")

    if combat_counts:
        print("✅ Combat content detected - patterns need adjustment")
    else:
        print("❌ No combat content found with current patterns")

    if magic_counts:
        print("✅ Magic content detected - patterns need adjustment")
    else:
        print("❌ No magic content found with current patterns")

    if matrix_counts:
        print("✅ Matrix content detected - patterns need adjustment")
    else:
        print("❌ No matrix content found with current patterns")

    print("\n💡 Next step: Update classification patterns based on this analysis")


if __name__ == "__main__":
    debug_content_patterns()