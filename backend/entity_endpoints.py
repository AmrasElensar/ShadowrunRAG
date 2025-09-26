"""
API endpoint for entity re-extraction from existing chunks.
Add this to your main FastAPI application or create a new endpoint file.
"""

from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
from pathlib import Path

# Import your indexer
from backend.indexer import create_enhanced_indexer

logger = logging.getLogger(__name__)

# Router for entity endpoints
entity_router = APIRouter(prefix="/api/entities", tags=["Entity Management"])


class EntityReextractionRequest(BaseModel):
    """Request model for entity re-extraction."""
    entity_types: Optional[List[str]] = Field(
        default=None,
        description="Specific entity types to extract: ['weapons', 'spells', 'ic_programs']. None = all types"
    )
    chunk_limit: Optional[int] = Field(
        default=None,
        description="Limit processing to N chunks for testing. None = all chunks"
    )
    force_update: bool = Field(
        default=False,
        description="Force re-extraction on all chunks, even those with existing entities"
    )
    chroma_path: Optional[str] = Field(
        default="data/chroma_db",
        description="Path to ChromaDB collection"
    )
    collection_name: Optional[str] = Field(
        default="shadowrun_docs",
        description="ChromaDB collection name"
    )


class EntityReextractionResponse(BaseModel):
    """Response model for entity re-extraction."""
    success: bool
    message: str
    extraction_stats: Dict[str, Any]
    final_registry_stats: Dict[str, Any]
    processing_time: float
    chunks_processed: int
    entities_found: int


class EntityStatsResponse(BaseModel):
    """Response model for entity statistics."""
    total_weapons: int
    total_spells: int
    total_ic_programs: int
    total_entities: int
    registry_details: Dict[str, Any]


# Initialize indexer (you might want to make this a dependency)
def get_indexer(chroma_path: str = "data/chroma_db", collection_name: str = "shadowrun_docs"):
    """Get indexer instance."""
    try:
        return create_enhanced_indexer(
            chroma_path=chroma_path,
            collection_name=collection_name
        )
    except Exception as e:
        logger.error(f"Failed to initialize indexer: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize indexer: {str(e)}")


@entity_router.post("/reextract", response_model=EntityReextractionResponse)
async def reextract_entities(request: EntityReextractionRequest):
    """
    Re-extract entities from existing chunks without full reindexing.

    This endpoint allows you to:
    - Test improved entity patterns on a subset of chunks
    - Process all chunks that don't have entities yet
    - Force re-extraction on all chunks with updated patterns
    """
    try:
        logger.info(f"Starting entity re-extraction with parameters: {request.dict()}")

        # Validate entity types if provided
        valid_types = {"weapons", "spells", "ic_programs"}
        if request.entity_types:
            invalid_types = set(request.entity_types) - valid_types
            if invalid_types:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid entity types: {invalid_types}. Valid types: {valid_types}"
                )

        # Initialize indexer
        indexer = get_indexer(request.chroma_path, request.collection_name)

        # Check if collection exists and has chunks
        collection_stats = indexer.get_collection_stats()
        if collection_stats.get("total_chunks", 0) == 0:
            raise HTTPException(
                status_code=400,
                detail="No chunks found in collection. Please index documents first."
            )

        logger.info(f"Collection has {collection_stats['total_chunks']} chunks")

        # Perform re-extraction
        result = indexer.reextract_entities_from_existing(
            entity_types=request.entity_types,
            chunk_limit=request.chunk_limit,
            force_update=request.force_update
        )

        if not result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail="Entity re-extraction failed"
            )

        # Build response
        extraction_stats = result["extraction_stats"]

        response = EntityReextractionResponse(
            success=True,
            message=f"Re-extracted entities from {extraction_stats['chunks_processed']} chunks",
            extraction_stats=extraction_stats,
            final_registry_stats=result["final_registry_stats"],
            processing_time=result["processing_time"],
            chunks_processed=extraction_stats["chunks_processed"],
            entities_found=extraction_stats["total_entities_found"]
        )

        logger.info(f"Entity re-extraction completed successfully: {extraction_stats}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Entity re-extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Entity re-extraction failed: {str(e)}")


@entity_router.get("/stats", response_model=EntityStatsResponse)
async def get_entity_stats(
        chroma_path: str = Query(default="data/chroma_db", description="Path to ChromaDB collection"),
        collection_name: str = Query(default="shadowrun_docs", description="ChromaDB collection name")
):
    """
    Get current entity registry statistics.
    """
    try:
        indexer = get_indexer(chroma_path, collection_name)

        # Get registry stats
        registry_stats = indexer.get_registry_stats()

        # Extract counts (adjust based on your registry stats format)
        total_weapons = registry_stats.get("weapons", 0)
        total_spells = registry_stats.get("spells", 0)
        total_ic = registry_stats.get("ic_programs", 0)

        response = EntityStatsResponse(
            total_weapons=total_weapons,
            total_spells=total_spells,
            total_ic_programs=total_ic,
            total_entities=total_weapons + total_spells + total_ic,
            registry_details=registry_stats
        )

        logger.info(f"Entity stats retrieved: {registry_stats}")
        return response

    except Exception as e:
        logger.error(f"Failed to get entity stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get entity stats: {str(e)}")


@entity_router.get("/lookup/{entity_name}")
async def lookup_entity(
        entity_name: str,
        entity_type: Optional[str] = Query(default=None, description="Entity type: weapon, spell, or ic"),
        chroma_path: str = Query(default="data/chroma_db", description="Path to ChromaDB collection"),
        collection_name: str = Query(default="shadowrun_docs", description="ChromaDB collection name")
):
    """
    Look up detailed information about a specific entity.
    """
    try:
        indexer = get_indexer(chroma_path, collection_name)

        # Get entity stats
        entity_info = indexer.get_entity_stats(entity_name, entity_type)

        if not entity_info:
            raise HTTPException(
                status_code=404,
                detail=f"Entity '{entity_name}' not found"
            )

        logger.info(f"Entity lookup successful for: {entity_name}")
        return entity_info

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Entity lookup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Entity lookup failed: {str(e)}")


@entity_router.post("/validate/weapon-mode")
async def validate_weapon_mode(
        weapon_name: str = Body(..., description="Name of the weapon"),
        firing_mode: str = Body(..., description="Firing mode to validate"),
        chroma_path: str = Query(default="data/chroma_db", description="Path to ChromaDB collection"),
        collection_name: str = Query(default="shadowrun_docs", description="ChromaDB collection name")
):
    """
    Validate if a weapon supports a specific firing mode.
    """
    try:
        indexer = get_indexer(chroma_path, collection_name)

        # Validate weapon mode
        validation_result = indexer.validate_weapon_mode(weapon_name, firing_mode)

        logger.info(f"Weapon mode validation: {weapon_name} -> {firing_mode} = {validation_result}")
        return validation_result

    except Exception as e:
        logger.error(f"Weapon mode validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@entity_router.get("/test/patterns")
async def test_entity_patterns(
        test_text: str = Query(..., description="Text to test entity patterns against"),
        chroma_path: str = Query(default="data/chroma_db", description="Path to ChromaDB collection"),
        collection_name: str = Query(default="shadowrun_docs", description="ChromaDB collection name")
):
    """
    Test entity extraction patterns against provided text.
    Useful for debugging and improving patterns.
    """
    try:
        indexer = get_indexer(chroma_path, collection_name)

        # Create a test chunk
        test_chunk = {
            "id": "test_chunk",
            "text": test_text,
            "source": "test",
            "metadata": {}
        }

        # Extract entities
        entities = indexer.entity_builder.extract_entities_from_chunk(test_chunk)

        # Count results
        entity_counts = {
            "weapons": len(entities.get("weapons", [])),
            "spells": len(entities.get("spells", [])),
            "ic_programs": len(entities.get("ic_programs", []))
        }

        result = {
            "test_text": test_text[:200] + "..." if len(test_text) > 200 else test_text,
            "entities_found": entities,
            "entity_counts": entity_counts,
            "total_entities": sum(entity_counts.values())
        }

        logger.info(f"Pattern test completed: {entity_counts}")
        return result

    except Exception as e:
        logger.error(f"Pattern test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pattern test failed: {str(e)}")

# Include this router in your main FastAPI app:
# from your_entity_endpoints import entity_router
# app.include_router(entity_router)