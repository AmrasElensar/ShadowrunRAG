"""
Enhanced Entity-Aware Retrieval System
Layers entity awareness on top of existing semantic search.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from tools.entity_registry_builder import EntityRegistryBuilder
from tools.registry_storage import EntityRegistryStorage

logger = logging.getLogger(__name__)


@dataclass
class EntityMatch:
    """Represents a matched entity in a query."""
    entity_type: str  # "weapon", "spell", "ic"
    entity_name: str
    confidence: float
    position: int  # Position in query text


@dataclass
class EnhancedSearchResult:
    """Enhanced search result with entity context."""
    chunk_id: str
    chunk_text: str
    score: float
    source: str
    metadata: Dict[str, Any]
    entity_matches: List[EntityMatch] = None
    validation_warnings: List[str] = None


class EntityAwareRetriever:
    """Adds entity awareness to existing retrieval system."""

    def __init__(self, registry_storage: EntityRegistryStorage):
        self.registry = registry_storage

        # Entity detection patterns
        self.weapon_indicators = [
            r'\b(ares|colt|ruger|browning|remington)\s+[\w\s-]+',
            r'\b[\w\s-]*(?:pistol|rifle|shotgun|smg)\b',
            r'\bpredator\b|\bgovernment\b|\bultra-power\b'
        ]

        self.spell_indicators = [
            r'\b(fireball|manabolt|lightning\s+bolt|death\s+touch)\b',
            r'\b[\w\s]+(?:bolt|ball|touch|barrier)\b',
            r'\bspell\b|\bmagic\b|\bdrain\b'
        ]

        self.ic_indicators = [
            r'\b(black\s+ic|blaster|acid|binder|killer)\b',
            r'\b[\w\s]*\s+ic\b',
            r'\bhost\b|\bmatrix\b|\bhacking\b'
        ]

        logger.info("Entity-Aware Retriever initialized")

    def enhance_search(self, query: str, base_results: List[Dict], max_results: int = 10) -> List[EnhancedSearchResult]:
        """Enhance base search results with entity awareness."""

        # Step 1: Extract entities from query
        entity_matches = self._extract_entities_from_query(query)

        # Step 2: Get entity-related chunks
        entity_chunks = self._get_entity_related_chunks(entity_matches)

        # Step 3: Validate entity combinations
        validation_warnings = self._validate_entity_combinations(query, entity_matches)

        # Step 4: Merge and rank results
        enhanced_results = self._merge_and_rank_results(
            base_results, entity_chunks, entity_matches, validation_warnings
        )

        return enhanced_results[:max_results]

    def _extract_entities_from_query(self, query: str) -> List[EntityMatch]:
        """Extract entity references from query text."""
        entities = []
        query_lower = query.lower()

        # Extract weapon references
        for pattern in self.weapon_indicators:
            matches = re.finditer(pattern, query_lower, re.IGNORECASE)
            for match in matches:
                entity_name = match.group().strip()
                # Clean up entity name
                entity_name = self._clean_entity_name(entity_name, "weapon")

                entities.append(EntityMatch(
                    entity_type="weapon",
                    entity_name=entity_name,
                    confidence=0.8,
                    position=match.start()
                ))

        # Extract spell references
        for pattern in self.spell_indicators:
            matches = re.finditer(pattern, query_lower, re.IGNORECASE)
            for match in matches:
                entity_name = match.group().strip()
                entity_name = self._clean_entity_name(entity_name, "spell")

                entities.append(EntityMatch(
                    entity_type="spell",
                    entity_name=entity_name,
                    confidence=0.7,
                    position=match.start()
                ))

        # Extract IC references
        for pattern in self.ic_indicators:
            matches = re.finditer(pattern, query_lower, re.IGNORECASE)
            for match in matches:
                entity_name = match.group().strip()
                entity_name = self._clean_entity_name(entity_name, "ic")

                entities.append(EntityMatch(
                    entity_type="ic",
                    entity_name=entity_name,
                    confidence=0.7,
                    position=match.start()
                ))

        # Remove duplicates and sort by position
        entities = self._deduplicate_entities(entities)
        entities.sort(key=lambda x: x.position)

        return entities

    def _clean_entity_name(self, raw_name: str, entity_type: str) -> str:
        """Clean and normalize entity names."""
        # Remove common noise words
        noise_words = ["the", "a", "an", "with", "using", "in", "on", "at"]
        words = raw_name.split()
        cleaned_words = [w for w in words if w.lower() not in noise_words]

        return " ".join(cleaned_words).strip()

    def _deduplicate_entities(self, entities: List[EntityMatch]) -> List[EntityMatch]:
        """Remove duplicate entity matches."""
        seen = set()
        unique_entities = []

        for entity in entities:
            key = (entity.entity_type, entity.entity_name.lower())
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)

        return unique_entities

    def _get_entity_related_chunks(self, entities: List[EntityMatch]) -> List[str]:
        """Get chunk IDs related to detected entities."""
        related_chunks = []

        for entity in entities:
            chunks = self.registry.get_related_chunks(entity.entity_type, entity.entity_name)
            related_chunks.extend(chunks)

        # Remove duplicates while preserving order
        seen = set()
        unique_chunks = []
        for chunk_id in related_chunks:
            if chunk_id not in seen:
                seen.add(chunk_id)
                unique_chunks.append(chunk_id)

        return unique_chunks

    def _validate_entity_combinations(self, query: str, entities: List[EntityMatch]) -> List[str]:
        """Validate entity combinations and detect impossible scenarios."""
        warnings = []

        # Check for weapon + firing mode combinations
        weapons = [e for e in entities if e.entity_type == "weapon"]

        # Look for firing mode indicators in query
        firing_modes = self._extract_firing_modes(query)

        for weapon in weapons:
            for mode in firing_modes:
                validation = self.registry.validate_weapon_mode(weapon.entity_name, mode)
                if not validation.get("valid", True):
                    warnings.append(validation.get("error", "Invalid weapon/mode combination"))

                    # Suggest alternatives
                    alternatives = self.registry.search_weapons_by_mode(mode)
                    if alternatives:
                        alt_names = [w.name for w in alternatives[:3]]
                        warnings.append(f"Weapons that support {mode}: {', '.join(alt_names)}")

        return warnings

    def _extract_firing_modes(self, query: str) -> List[str]:
        """Extract firing mode references from query."""
        mode_patterns = {
            "burst": r'\bburst\s*(?:fire|mode)?\b',
            "semi-auto": r'\bsemi[-\s]*auto(?:matic)?\b',
            "full auto": r'\bfull[-\s]*auto(?:matic)?\b',
            "single shot": r'\bsingle[-\s]*shot\b'
        }

        modes = []
        query_lower = query.lower()

        for mode_name, pattern in mode_patterns.items():
            if re.search(pattern, query_lower):
                modes.append(mode_name)

        return modes

    def _merge_and_rank_results(self,
                                base_results: List[Dict],
                                entity_chunks: List[str],
                                entity_matches: List[EntityMatch],
                                validation_warnings: List[str]) -> List[EnhancedSearchResult]:
        """Merge semantic results with entity-specific chunks."""

        enhanced_results = []
        seen_chunks = set()

        # Convert base results to enhanced format
        for result in base_results:
            chunk_id = result.get("id", "")
            if chunk_id in seen_chunks:
                continue

            enhanced_result = EnhancedSearchResult(
                chunk_id=chunk_id,
                chunk_text=result.get("text", ""),
                score=result.get("score", 0.5),
                source=result.get("source", ""),
                metadata=result.get("metadata", {}),
                entity_matches=entity_matches,
                validation_warnings=validation_warnings
            )

            # Boost score if chunk contains entities
            if chunk_id in entity_chunks:
                enhanced_result.score *= 1.5  # Boost entity-related chunks

            enhanced_results.append(enhanced_result)
            seen_chunks.add(chunk_id)

        # Add entity-specific chunks that weren't in semantic results
        for chunk_id in entity_chunks:
            if chunk_id not in seen_chunks:
                # We would need to fetch chunk text from the main storage here
                # For now, create placeholder
                enhanced_result = EnhancedSearchResult(
                    chunk_id=chunk_id,
                    chunk_text="[Entity-related chunk]",
                    score=0.7,  # High score for entity matches
                    source="entity_registry",
                    metadata={"entity_related": True},
                    entity_matches=entity_matches,
                    validation_warnings=validation_warnings
                )
                enhanced_results.append(enhanced_result)
                seen_chunks.add(chunk_id)

        # Sort by score (descending)
        enhanced_results.sort(key=lambda x: x.score, reverse=True)

        return enhanced_results

    def get_entity_stats(self, entity_name: str, entity_type: str = None) -> Optional[Dict[str, Any]]:
        """Get detailed stats for a specific entity."""

        if entity_type == "weapon" or entity_type is None:
            weapon = self.registry.get_weapon(entity_name)
            if weapon:
                return {
                    "type": "weapon",
                    "name": weapon.name,
                    "stats": {
                        "accuracy": weapon.accuracy,
                        "damage": weapon.damage,
                        "ap": weapon.ap,
                        "mode": weapon.mode,
                        "rc": weapon.rc,
                        "ammo": weapon.ammo,
                        "avail": weapon.avail,
                        "cost": weapon.cost
                    },
                    "category": weapon.category,
                    "manufacturer": weapon.manufacturer,
                    "description": weapon.description
                }

        if entity_type == "spell" or entity_type is None:
            spell = self.registry.get_spell(entity_name)
            if spell:
                return {
                    "type": "spell",
                    "name": spell.name,
                    "stats": {
                        "spell_type": spell.spell_type,
                        "range": spell.range,
                        "damage": spell.damage,
                        "duration": spell.duration,
                        "drain": spell.drain
                    },
                    "keywords": spell.keywords,
                    "category": spell.category,
                    "description": spell.description
                }

        if entity_type == "ic" or entity_type is None:
            ic = self.registry.get_ic_program(entity_name)
            if ic:
                return {
                    "type": "ic",
                    "name": ic.name,
                    "stats": {
                        "attack_pattern": ic.attack_pattern,
                        "resistance_test": ic.resistance_test
                    },
                    "description": ic.effect_description,
                    "category": ic.category
                }

        return None


class EntityIntegrationLayer:
    """Integration layer for adding entity awareness to existing RAG system."""

    def __init__(self, registry_builder: EntityRegistryBuilder, registry_storage: EntityRegistryStorage):
        self.builder = registry_builder
        self.storage = registry_storage
        self.retriever = EntityAwareRetriever(registry_storage)

        logger.info("Entity Integration Layer initialized")

    def process_chunk_for_entities(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Process a chunk during indexing to extract entities (non-breaking)."""

        # Extract entities
        entities = self.builder.extract_entities_from_chunk(chunk)

        # Store entities if any found
        if any(entities.values()):
            chunk_id = chunk.get("id", "")
            source = chunk.get("source", "")
            self.storage.store_entities(entities, chunk_id, source)

            # Add entity metadata to chunk (optional)
            if "metadata" not in chunk:
                chunk["metadata"] = {}

            chunk["metadata"]["extracted_entities"] = {
                "weapons": len(entities.get("weapons", [])),
                "spells": len(entities.get("spells", [])),
                "ic_programs": len(entities.get("ic_programs", []))
            }

            logger.info(f"Extracted entities from {chunk_id}: {chunk['metadata']['extracted_entities']}")

        return chunk

    def enhance_search_results(self, query: str, base_results: List[Dict]) -> Tuple[
        List[EnhancedSearchResult], Dict[str, Any]]:
        """Enhance search results with entity awareness."""

        enhanced_results = self.retriever.enhance_search(query, base_results)

        # Compile entity information for response
        entity_info = {
            "entities_detected": [],
            "validation_warnings": [],
            "registry_stats": self.storage.get_registry_stats()
        }

        if enhanced_results:
            first_result = enhanced_results[0]
            if first_result.entity_matches:
                entity_info["entities_detected"] = [
                    {"type": e.entity_type, "name": e.entity_name, "confidence": e.confidence}
                    for e in first_result.entity_matches
                ]

            if first_result.validation_warnings:
                entity_info["validation_warnings"] = first_result.validation_warnings

        return enhanced_results, entity_info


# Factory functions for integration
def create_entity_integration_layer(db_path: str = "data/entity_registry.db") -> EntityIntegrationLayer:
    """Create complete entity integration layer."""
    builder = EntityRegistryBuilder()
    storage = EntityRegistryStorage(db_path)
    return EntityIntegrationLayer(builder, storage)


def create_entity_aware_retriever(db_path: str = "data/entity_registry.db") -> EntityAwareRetriever:
    """Create entity-aware retriever only."""
    storage = EntityRegistryStorage(db_path)
    return EntityAwareRetriever(storage)