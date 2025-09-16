"""
Enhanced query processing system that improves retrieval targeting.
Consolidates all query processing logic.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QueryAnalysis:
    """Structured query analysis results."""
    intent: str  # weapon, matrix, magic, etc.
    entities: List[str]  # specific items mentioned
    query_type: str  # lookup, comparison, explanation
    filters: Dict[str, Any]  # ChromaDB filters to apply
    boost_terms: List[str]  # Terms to boost in search
    confidence: float  # Confidence in analysis


class EnhancedQueryProcessor:
    """Enhanced query processor for better content targeting."""

    def __init__(self):
        # Import verified patterns
        try:
            from .verified_shadowrun_patterns import VERIFIED_SHADOWRUN_PATTERNS
            self.patterns = VERIFIED_SHADOWRUN_PATTERNS
        except ImportError:
            logger.warning("Verified patterns not found, using basic patterns")
            self.patterns = self._get_basic_patterns()

        # Extract pattern categories for easier access
        self.weapon_patterns = self.patterns.get("gear", {})
        self.matrix_patterns = self.patterns.get("matrix", {})
        self.magic_patterns = self.patterns.get("magic", {})
        self.rigger_patterns = self.patterns.get("riggers", {})
        self.combat_patterns = self.patterns.get("combat", {})
        self.skills_patterns = self.patterns.get("skills", {})
        self.chargen_patterns = self.patterns.get("character_creation", {})
        self.social_patterns = self.patterns.get("social", {})

        # Query intent patterns
        self.intent_patterns = {
            "stats_lookup": ["stats", "statistics", "details", "information about", "tell me about"],
            "comparison": ["vs", "versus", "compare", "better than", "difference between"],
            "rules_question": ["how to", "rules for", "how does", "can i", "is it possible"],
            "cost_availability": ["cost", "price", "how much", "availability", "where to buy"]
        }

    def _get_basic_patterns(self) -> Dict:
        """Fallback patterns if verified patterns not available."""
        return {
            "gear": {
                "manufacturers": ["ares", "colt", "ruger", "browning"],
                "weapon_types": ["pistol", "rifle", "shotgun"],
                "specific_weapons": ["predator", "government 2066"]
            },
            "matrix": {
                "ic_programs": ["black ic", "white ic"],
                "matrix_damage": ["matrix damage", "biofeedback"]
            },
            "magic": {
                "specific_spells": ["fireball", "manabolt"],
                "magic_skills": ["spellcasting", "summoning"]
            }
        }

    def analyze_query(self, query: str) -> QueryAnalysis:
        """Comprehensive query analysis for better retrieval targeting."""

        query_lower = query.lower()

        # Determine intent
        intent = self._determine_intent(query_lower)

        # Extract entities
        entities = self._extract_entities(query_lower)

        # Determine query type
        query_type = self._determine_query_type(query_lower)

        # Generate filters and boost terms
        filters, boost_terms = self._generate_search_parameters(intent, entities, query_lower)

        # Calculate confidence
        confidence = self._calculate_confidence(intent, entities, query_lower)

        return QueryAnalysis(
            intent=intent,
            entities=entities,
            query_type=query_type,
            filters=filters,
            boost_terms=boost_terms,
            confidence=confidence
        )

    def _determine_intent(self, query: str) -> str:
        """Determine the primary intent of the query."""

        # Check for weapon-related queries
        weapon_score = 0
        for category, terms in self.weapon_patterns.items():
            for term in terms:
                if term in query:
                    weapon_score += 1

        # Check for matrix-related queries
        matrix_score = 0
        for category, terms in self.matrix_patterns.items():
            for term in terms:
                if term in query:
                    matrix_score += 1

        # Check for magic-related queries
        magic_score = 0
        for category, terms in self.magic_patterns.items():
            for term in terms:
                if term in query:
                    magic_score += 1

        # Check for rigger-related queries
        rigger_score = 0
        for category, terms in self.rigger_patterns.items():
            for term in terms:
                if term in query:
                    rigger_score += 1

        # Check for combat-related queries
        combat_score = 0
        for category, terms in self.combat_patterns.items():
            for term in terms:
                if term in query:
                    combat_score += 1

        # Check for skills-related queries
        skills_score = 0
        for category, terms in self.skills_patterns.items():
            for term in terms:
                if term in query:
                    skills_score += 1

        # Check for character creation queries
        chargen_score = 0
        for category, terms in self.chargen_patterns.items():
            for term in terms:
                if term in query:
                    chargen_score += 1

        # Check for social queries
        social_score = 0
        for category, terms in self.social_patterns.items():
            for term in terms:
                if term in query:
                    social_score += 1

        # Determine highest scoring intent
        scores = [
            ("weapon", weapon_score),
            ("matrix", matrix_score),
            ("magic", magic_score),
            ("rigger", rigger_score),
            ("combat", combat_score),
            ("skills", skills_score),
            ("character_creation", chargen_score),
            ("social", social_score)
        ]

        intent, max_score = max(scores, key=lambda x: x[1])

        # Default to general if no strong signals
        if max_score == 0:
            return "general"

        return intent

    def _extract_entities(self, query: str) -> List[str]:
        """Extract specific entities mentioned in the query."""

        entities = []

        # Extract from all pattern categories
        for section_patterns in [self.weapon_patterns, self.matrix_patterns, self.magic_patterns,
                                self.rigger_patterns, self.combat_patterns, self.skills_patterns,
                                self.chargen_patterns, self.social_patterns]:
            for category, terms in section_patterns.items():
                if isinstance(terms, list):
                    for term in terms:
                        if term in query:
                            entities.append(term)

        return list(set(entities))  # Remove duplicates

    def _determine_query_type(self, query: str) -> str:
        """Determine the type of query (lookup, comparison, etc.)."""

        for query_type, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if pattern in query:
                    return query_type

        # Default based on query structure
        if "?" in query:
            return "question"
        elif len(query.split()) <= 3:
            return "lookup"
        else:
            return "explanation"

    def _generate_search_parameters(self, intent: str, entities: List[str], query: str) -> Tuple[Dict, List[str]]:
        """Generate ChromaDB filters and boost terms based on analysis."""

        filters = {}
        boost_terms = []

        # Intent-based filters
        if intent == "weapon":
            filters["primary_section"] = "Gear"
            boost_terms.extend(["weapon", "damage", "accuracy", "cost"])

        elif intent == "matrix":
            filters["primary_section"] = "Matrix"
            boost_terms.extend(["matrix", "hacking", "cyberdeck", "ic"])

        elif intent == "magic":
            filters["primary_section"] = "Magic"
            boost_terms.extend(["spell", "magic", "force", "drain"])

        elif intent == "rigger":
            filters["primary_section"] = "Riggers"
            boost_terms.extend(["drone", "vehicle", "rigger", "pilot"])

        elif intent == "combat":
            filters["primary_section"] = "Combat"
            boost_terms.extend(["combat", "initiative", "damage", "test"])

        elif intent == "skills":
            filters["primary_section"] = "Skills"
            boost_terms.extend(["skill", "dice pool", "test", "threshold"])

        elif intent == "character_creation":
            filters["primary_section"] = "Character_Creation"
            boost_terms.extend(["priority", "attribute", "metatype", "karma"])

        elif intent == "social":
            filters["primary_section"] = "Social"
            boost_terms.extend(["contact", "etiquette", "negotiation", "loyalty"])

        # Add entities as boost terms
        boost_terms.extend(entities)

        # Query type specific filters
        if any(term in query for term in ["table", "list", "stats"]):
            filters["content_type"] = "table"

        elif any(term in query for term in ["rule", "mechanic", "how to"]):
            filters["contains_rules"] = True

        return filters, boost_terms

    def _calculate_confidence(self, intent: str, entities: List[str], query: str) -> float:
        """Calculate confidence in the query analysis."""

        confidence = 0.5  # Base confidence

        # Boost confidence for specific entities found
        if entities:
            confidence += min(0.3, len(entities) * 0.1)

        # Boost confidence for clear intent signals
        if intent != "general":
            confidence += 0.2

        # Boost confidence for clear query structure
        if any(term in query for term in ["what is", "how to", "tell me about"]):
            confidence += 0.1

        return min(1.0, confidence)

    def build_enhanced_query(self, original_query: str, analysis: QueryAnalysis) -> str:
        """Build an enhanced query string for better retrieval."""

        enhanced_parts = [original_query]

        # Add boost terms if they're not already in the query
        for term in analysis.boost_terms:
            if term not in original_query.lower():
                enhanced_parts.append(term)

        # Add entity synonyms for better matching
        entity_synonyms = self._get_entity_synonyms(analysis.entities)
        enhanced_parts.extend(entity_synonyms)

        return " ".join(enhanced_parts)

    def _get_entity_synonyms(self, entities: List[str]) -> List[str]:
        """Get synonyms for entities to improve matching."""

        synonyms = []

        for entity in entities:
            # Add common synonyms for weapons
            if entity in ["pistol", "handgun"]:
                synonyms.append("sidearm")
            elif entity in ["rifle", "assault rifle"]:
                synonyms.append("longarm")
            elif entity in ["shotgun"]:
                synonyms.append("scattergun")

            # Add common synonyms for matrix terms
            elif entity in ["hacking", "matrix"]:
                synonyms.append("cybercombat")
            elif entity in ["cyberdeck"]:
                synonyms.append("deck")

            # Add common synonyms for magic terms
            elif entity in ["spell", "magic"]:
                synonyms.append("magical")
            elif entity in ["summoning"]:
                synonyms.append("spirit")

        return synonyms


# Factory function for integration
def create_enhanced_query_processor() -> EnhancedQueryProcessor:
    """Create enhanced query processor instance."""
    return EnhancedQueryProcessor()