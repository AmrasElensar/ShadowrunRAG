"""
Complete integration patch for adding entity registry to existing IncrementalIndexer.
This patch adds entity processing without breaking existing functionality.

USAGE:
    from backend.indexer import IncrementalIndexer
    from tools.indexer_integration_patch import EntityEnhancedIndexer

    # Your existing indexer
    base_indexer = IncrementalIndexer(...)

    # Wrap with entity capabilities
    indexer = EntityEnhancedIndexer(base_indexer)

    # Use exactly as before + new entity features
    indexer.index_documents("docs/")
    results = indexer.search("Ares Predator burst fire")
"""

import logging
from typing import Dict, List, Any, Optional

# Import the new entity components
from tools.enhanced_retrieval import create_entity_integration_layer, EntityIntegrationLayer

logger = logging.getLogger(__name__)


class EntityEnhancedIndexer:
    """Wrapper that adds entity processing to existing IncrementalIndexer."""

    def __init__(self, base_indexer, entity_db_path: str = "data/entity_registry.db"):
        """
        Initialize with existing indexer instance.

        Args:
            base_indexer: Your existing IncrementalIndexer instance
            entity_db_path: Path to entity registry database
        """
        self.base_indexer = base_indexer
        self.entity_layer = create_entity_integration_layer(entity_db_path)

        logger.info("Entity-Enhanced Indexer initialized")

    def index_documents(self, documents_path: str, file_patterns: List[str] = None) -> Dict[str, Any]:
        """
        Enhanced document indexing with entity extraction.
        Delegates to base indexer then adds entity processing.
        """

        # Step 1: Use existing indexer functionality
        logger.info("Running base indexing...")
        base_result = self.base_indexer.index_documents(documents_path, file_patterns)

        # Step 2: Process chunks for entity extraction
        logger.info("Processing chunks for entity extraction...")
        chunks_processed = 0
        entities_extracted = 0

        # Get all chunks from the collection
        collection = self.base_indexer.collection
        all_chunks = collection.get()

        if all_chunks and 'documents' in all_chunks:
            for i, (chunk_id, chunk_text, metadata) in enumerate(zip(
                all_chunks['ids'],
                all_chunks['documents'],
                all_chunks['metadatas']
            )):
                # Create chunk dictionary for entity processing
                chunk_dict = {
                    "id": chunk_id,
                    "text": chunk_text,
                    "source": metadata.get("source", "unknown"),
                    "metadata": metadata
                }

                # Process chunk for entities (non-breaking)
                try:
                    enhanced_chunk = self.entity_layer.process_chunk_for_entities(chunk_dict)

                    # Check if entities were extracted
                    extracted_counts = enhanced_chunk.get("metadata", {}).get("extracted_entities", {})
                    if any(extracted_counts.values()):
                        entities_extracted += sum(extracted_counts.values())

                    chunks_processed += 1

                    if chunks_processed % 10 == 0:
                        logger.info(f"Processed {chunks_processed} chunks for entities...")

                except Exception as e:
                    logger.warning(f"Entity extraction failed for chunk {chunk_id}: {e}")
                    # Continue processing - don't break on entity extraction errors

        # Step 3: Get registry statistics
        registry_stats = self.entity_layer.storage.get_registry_stats()

        # Step 4: Enhance base result with entity information
        enhanced_result = base_result.copy()
        enhanced_result.update({
            "entity_processing": {
                "chunks_processed": chunks_processed,
                "entities_extracted": entities_extracted,
                "registry_stats": registry_stats
            }
        })

        logger.info(f"Entity processing complete: {entities_extracted} entities extracted from {chunks_processed} chunks")
        logger.info(f"Registry now contains: {registry_stats}")

        return enhanced_result

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Enhanced search with entity awareness.
        Uses base search then adds entity enhancements.
        """

        # Step 1: Get base search results
        base_results = self.base_indexer.search(query, n_results * 2)  # Get more for reranking

        # Step 2: Convert base results to list format for entity processing
        base_chunks = []
        if base_results.get("documents"):
            for i, (doc, metadata, distance) in enumerate(zip(
                base_results["documents"][0],
                base_results["metadatas"][0],
                base_results["distances"][0]
            )):
                base_chunks.append({
                    "id": base_results["ids"][0][i],
                    "text": doc,
                    "metadata": metadata,
                    "score": 1.0 - distance,  # Convert distance to score
                    "source": metadata.get("source", "unknown")
                })

        # Step 3: Enhance with entity awareness
        try:
            enhanced_results, entity_info = self.entity_layer.enhance_search_results(query, base_chunks)

            # Step 4: Convert back to expected format
            enhanced_response = {
                "documents": [[r.chunk_text for r in enhanced_results[:n_results]]],
                "metadatas": [[r.metadata for r in enhanced_results[:n_results]]],
                "ids": [[r.chunk_id for r in enhanced_results[:n_results]]],
                "distances": [[1.0 - r.score for r in enhanced_results[:n_results]]],
                "entity_info": entity_info
            }

            # Log entity detection
            if entity_info.get("entities_detected"):
                detected = ", ".join([f"{e['name']} ({e['type']})" for e in entity_info["entities_detected"]])
                logger.info(f"Entities detected in query: {detected}")

            if entity_info.get("validation_warnings"):
                for warning in entity_info["validation_warnings"]:
                    logger.warning(f"Entity validation: {warning}")

            return enhanced_response

        except Exception as e:
            logger.error(f"Entity enhancement failed: {e}")
            # Fallback to base results
            return base_results

    def get_entity_stats(self, entity_name: str, entity_type: str = None) -> Optional[Dict[str, Any]]:
        """Get detailed statistics for a specific entity."""
        return self.entity_layer.retriever.get_entity_stats(entity_name, entity_type)

    def validate_weapon_mode(self, weapon_name: str, mode: str) -> Dict[str, Any]:
        """Validate weapon firing mode capability."""
        return self.entity_layer.storage.validate_weapon_mode(weapon_name, mode)

    # Delegate all other methods to base indexer
    def __getattr__(self, name):
        """Delegate unknown attributes to base indexer."""
        return getattr(self.base_indexer, name)


# Complete integration functions
def integrate_with_existing_indexer(base_indexer) -> 'EntityEnhancedIndexer':
    """
    Complete integration function that wraps your existing indexer.

    Usage in your backend/indexer.py or main application:

    ```python
    from backend.indexer import IncrementalIndexer
    from tools.indexer_integration_patch import integrate_with_existing_indexer

    # Your existing indexer initialization
    indexer = IncrementalIndexer(
        chroma_path="data/chroma_db",
        collection_name="shadowrun_docs",
        embedding_model="nomic-embed-text"
    )

    # Add entity capabilities (this line adds everything)
    enhanced_indexer = integrate_with_existing_indexer(indexer)

    # Use exactly as before
    enhanced_indexer.index_documents("docs/")
    results = enhanced_indexer.search("Ares Predator burst fire")

    # Plus new entity features
    weapon_stats = enhanced_indexer.get_entity_stats("Ares Predator", "weapon")
    validation = enhanced_indexer.validate_weapon_mode("Ares Predator", "burst fire")
    ```

    Args:
        base_indexer: Your existing IncrementalIndexer instance

    Returns:
        EntityEnhancedIndexer: Enhanced indexer with entity capabilities
    """
    return EntityEnhancedIndexer(base_indexer)


def create_enhanced_indexer(chroma_path: str = "data/chroma_db",
                           collection_name: str = "shadowrun_docs",
                           embedding_model: str = "nomic-embed-text",
                           **kwargs) -> 'EntityEnhancedIndexer':
    """
    Alternative: Create new enhanced indexer from scratch.

    Usage:
    ```python
    from tools.indexer_integration_patch import create_enhanced_indexer

    # Creates IncrementalIndexer + entity capabilities in one step
    indexer = create_enhanced_indexer(
        chroma_path="data/chroma_db",
        collection_name="shadowrun_docs"
    )
    ```
    """
    from backend.indexer import IncrementalIndexer

    base_indexer = IncrementalIndexer(
        chroma_path=chroma_path,
        collection_name=collection_name,
        embedding_model=embedding_model,
        **kwargs
    )

    return EntityEnhancedIndexer(base_indexer)


# Example integration for your specific setup
def setup_shadowrun_indexer() -> 'EntityEnhancedIndexer':
    """
    Complete setup function for Shadowrun indexer with entity capabilities.

    This is a ready-to-use function you can copy to your main application.
    """
    from backend.indexer import IncrementalIndexer

    # Initialize base indexer with your settings
    base_indexer = IncrementalIndexer(
        chroma_path="data/chroma_db",
        collection_name="shadowrun_docs",
        embedding_model="nomic-embed-text",
        chunk_size=800,
        chunk_overlap=150
    )

    # Add entity capabilities
    enhanced_indexer = EntityEnhancedIndexer(base_indexer)

    logger.info("Shadowrun indexer with entity capabilities ready")
    return enhanced_indexer


# Test function to verify integration
def test_entity_integration():
    """Test the entity integration with sample data."""

    # This would test with your actual indexer
    print("Entity Integration Test")
    print("======================")

    # Mock test data
    sample_weapon_chunk = {
        "id": "test_chunk_1",
        "text": """
        | Weapon | ACC | Damage | AP | Mode | RC | Ammo | Avail | Cost |
        | Ares Predator V | 5 (7) | 8P | –1 | SA | — | 15 (c) | 5R | 725¥ |
        | Ares Viper Slivergun | 4 | 9P (f) | +4 | SA/BF | — | 30 (c) | 8F | 380¥ |
        
        **Ares Predator V:** The newest iteration of the most popular handgun in the world.
        """,
        "source": "test_data",
        "metadata": {}
    }

    # Test entity extraction
    entity_layer = create_entity_integration_layer()
    enhanced_chunk = entity_layer.process_chunk_for_entities(sample_weapon_chunk)

    print(f"Entities extracted: {enhanced_chunk.get('metadata', {}).get('extracted_entities', {})}")

    # Test validation
    validation = entity_layer.storage.validate_weapon_mode("Ares Predator", "burst fire")
    print(f"Burst fire validation: {validation}")

    validation2 = entity_layer.storage.validate_weapon_mode("Ares Viper", "burst fire")
    print(f"Viper burst validation: {validation2}")

    # Test registry stats
    stats = entity_layer.storage.get_registry_stats()
    print(f"Registry stats: {stats}")


if __name__ == "__main__":
    test_entity_integration()