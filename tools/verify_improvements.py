# ========================================
# VERIFICATION SCRIPT
# ========================================

def verify_improvements():
    """
    Run this after implementing the changes to verify they work.
    Add this to tools/verify_improvements.py
    """

    import chromadb
    from chromadb.config import Settings
    from collections import Counter

    # Connect to updated ChromaDB
    client = chromadb.PersistentClient(
        path="data/chroma_db",
        settings=Settings(anonymized_telemetry=False)
    )

    collection = client.get_collection("shadowrun_docs")

    # Get sample of chunks
    sample = collection.get(limit=50, include=['metadatas'])

    if not sample['metadatas']:
        print("❌ No chunks found - re-index your documents first")
        return

    # Analyze improvements
    primary_sections = [m.get('primary_section', 'Unknown') for m in sample['metadatas']]
    content_types = [m.get('content_type', 'unknown') for m in sample['metadatas']]
    multi_label_count = sum(1 for m in sample['metadatas']
                            if len(str(m.get('sections', '')).split(',')) > 1)

    print("🔍 VERIFICATION RESULTS:")
    print(f"   Primary sections: {dict(Counter(primary_sections))}")
    print(f"   Content types: {dict(Counter(content_types))}")
    print(f"   Multi-label chunks: {multi_label_count}/{len(sample['metadatas'])}")
    print(f"   Classification diversity: {len(set(primary_sections)) / len(primary_sections) * 100:.1f}%")

    # Check for taser-specific improvements
    taser_chunks = [m for m in sample['metadatas']
                    if any(word in str(m.get('mechanical_keywords', '')).lower()
                           for word in ['taser', 'stun', 'electrical'])]

    print(f"   Taser-related chunks found: {len(taser_chunks)}")

    if len(set(primary_sections)) > 3:
        print("✅ IMPROVEMENT SUCCESS: Classification diversity improved!")
    else:
        print("❌ ISSUE: Still low classification diversity")

    if multi_label_count > len(sample['metadatas']) * 0.2:
        print("✅ IMPROVEMENT SUCCESS: Multi-label classification working!")
    else:
        print("❌ ISSUE: Multi-label classification not working")


if __name__ == "__main__":
    print("Save this as tools/verify_improvements.py and run after implementing changes")
    verify_improvements()