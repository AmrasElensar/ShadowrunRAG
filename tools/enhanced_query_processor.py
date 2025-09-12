"""
Fixed Enhanced query processing system that improves retrieval targeting.
This fixes the implementation issues in the current version.
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


class ShadowrunQueryProcessor:
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
                "manufacturers": ["ares", "colt", "ruger", "browning", "remington"],
                "weapon_types": ["pistol", "rifle", "shotgun", "smg"],
                "specific_weapons": ["predator", "government 2066", "ultra-power"],
                "weapon_stats": ["accuracy", "damage", "ap", "mode", "cost"]
            },
            "matrix": {
                "ic_programs": ["black ic", "white ic", "killer ic"],
                "matrix_actions": ["hack on the fly", "brute force", "data spike"],
                "matrix_damage": ["matrix damage", "biofeedback", "dumpshock"]
            },
            "magic": {
                "specific_spells": ["fireball", "manabolt", "lightning bolt"],
                "spell_mechanics": ["force", "drain", "spellcasting"],
                "awakened_types": ["mage", "shaman", "adept"]
            }
        }

    def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze user query to determine intent and target content."""

        query_lower = query.lower().strip()

        # Determine primary intent
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
        """Determine the primary intent/domain of the query."""

        intent_scores = {
            "weapon": 0,
            "matrix": 0,
            "magic": 0,
            "rigger": 0,
            "combat": 0,
            "skill": 0,
            "character": 0,
            "social": 0,
            "setting": 0
        }

        # Score weapon intent
        for category, terms in self.weapon_patterns.items():
            if isinstance(terms, list):
                for term in terms:
                    if term in query:
                        intent_scores["weapon"] += 2

        # Score matrix intent
        for category, terms in self.matrix_patterns.items():
            if isinstance(terms, list):
                for term in terms:
                    if term in query:
                        intent_scores["matrix"] += 2

        # Score magic intent
        for category, terms in self.magic_patterns.items():
            if isinstance(terms, list):
                for term in terms:
                    if term in query:
                        intent_scores["magic"] += 2

        # Score rigger intent
        for category, terms in self.rigger_patterns.items():
            if isinstance(terms, list):
                for term in terms:
                    if term in query:
                        intent_scores["rigger"] += 2

        # Additional context scoring
        context_clues = {
            "weapon": ["gun", "weapon", "pistol", "rifle", "blade", "stats", "damage"],
            "matrix": ["hack", "cyberdeck", "ic", "matrix", "program", "firewall"],
            "magic": ["spell", "magic", "spirit", "mage", "cast", "force", "drain"],
            "rigger": ["drone", "vehicle", "pilot", "rig", "jumped in"],
            "combat": ["attack", "defense", "initiative", "combat", "fight"],
            "skill": ["test", "dice pool", "skill", "attribute", "roll"],
            "character": ["build", "creation", "priority", "karma", "chargen"],
            "social": ["contact", "lifestyle", "reputation", "negotiation"],
            "setting": ["lore", "background", "corp", "history", "timeline"]
        }

        for intent, clues in context_clues.items():
            for clue in clues:
                if clue in query:
                    intent_scores[intent] += 1

        # Return highest scoring intent
        if max(intent_scores.values()) == 0:
            return "general"

        return max(intent_scores, key=intent_scores.get)

    def _extract_entities(self, query: str) -> List[str]:
        """Extract specific entities mentioned in the query."""

        entities = []

        # Extract from all pattern categories
        for section_patterns in [self.weapon_patterns, self.matrix_patterns, self.magic_patterns,
                                self.rigger_patterns, self.combat_patterns, self.skills_patterns]:
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

        # Map intents to sections
        section_mapping = {
            "weapon": "Gear",
            "magic": "Magic",
            "matrix": "Matrix",
            "rigger": "Riggers",
            "combat": "Combat",
            "skill": "Skills",
            "character": "Character_Creation",
            "social": "Social",
            "setting": "Setting"
        }

        if intent in section_mapping:
            target_section = section_mapping[intent]

            filters = {
                "primary_section": {"$eq": target_section},
                "$or": [
                    {"content_type": {"$eq": "table"}},
                    {"contains_rules": {"$eq": True}},
                    {"content_type": {"$eq": "explicit_rule"}}
                ]
            }

            # Section-specific boost terms
            section_boost_terms = {
                "Gear": ["weapon", "damage", "accuracy", "ap", "cost", "gear", "equipment"],
                "Magic": ["spell", "spirit", "magic", "force", "drain", "astral", "mana"],
                "Matrix": ["matrix", "ic", "hack", "cyberdeck", "firewall", "program"],
                "Riggers": ["drone", "vehicle", "pilot", "rigger", "jumped in", "rcc"],
                "Combat": ["combat", "attack", "defense", "initiative", "damage", "armor"],
                "Skills": ["skill", "test", "dice pool", "attribute", "threshold"],
                "Character_Creation": ["priority", "karma", "metatype", "quality", "creation"],
                "Social": ["contact", "lifestyle", "reputation", "social", "etiquette"],
                "Setting": ["corporation", "history", "timeline", "location", "lore"]
            }

            boost_terms.extend(section_boost_terms.get(target_section, []))
            boost_terms.extend(entities)

        else:
            # General query - no strict filters but still boost with entities
            boost_terms.extend(entities)

        return filters, boost_terms

    def _calculate_confidence(self, intent: str, entities: List[str], query: str) -> float:
        """Calculate confidence in the query analysis."""

        confidence = 0.5  # Base confidence

        # Boost confidence for clear intent indicators
        if intent != "general":
            confidence += 0.2

        # Boost confidence for specific entities
        if entities:
            confidence += min(0.3, len(entities) * 0.1)

        # Boost confidence for specific weapon/matrix/magic terms
        specific_terms = 0
        for term in ["predator", "ares", "matrix", "ic", "spell", "spirit"]:
            if term in query:
                specific_terms += 1

        confidence += min(0.2, specific_terms * 0.05)

        return min(1.0, confidence)

    def process_query_for_retrieval(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """Process query and return parameters for ChromaDB retrieval."""

        analysis = self.analyze_query(query)

        # Build search parameters
        search_params = {
            "query_texts": [query],
            "n_results": max_results,
            "include": ["documents", "metadatas", "distances"]
        }

        # Add filters if high confidence
        if analysis.confidence > 0.7 and analysis.filters:
            search_params["where"] = analysis.filters

        # Create enhanced query with boost terms
        if analysis.boost_terms:
            # Create expanded query with boost terms
            expanded_query = f"{query} {' '.join(analysis.boost_terms[:5])}"
            search_params["query_texts"] = [expanded_query]

        return {
            "search_params": search_params,
            "analysis": analysis,
            "debug_info": {
                "original_query": query,
                "intent": analysis.intent,
                "entities": analysis.entities,
                "confidence": analysis.confidence,
                "filters_applied": bool(analysis.filters),
                "boost_terms": analysis.boost_terms[:5]
            }
        }

    def explain_query_processing(self, query: str) -> str:
        """Provide human-readable explanation of query processing."""

        analysis = self.analyze_query(query)

        explanation = f"Query Analysis for: '{query}'\n\n"
        explanation += f"Detected Intent: {analysis.intent.title()}\n"
        explanation += f"Confidence: {analysis.confidence:.2f}\n"

        if analysis.entities:
            explanation += f"Entities Found: {', '.join(analysis.entities)}\n"

        explanation += f"Query Type: {analysis.query_type}\n"

        if analysis.filters:
            explanation += f"Applied Filters: {analysis.filters}\n"

        if analysis.boost_terms:
            explanation += f"Boost Terms: {', '.join(analysis.boost_terms[:5])}\n"

        return explanation


# Integration helper for existing system
def create_enhanced_query_processor() -> ShadowrunQueryProcessor:
    """Create enhanced query processor instance."""
    return ShadowrunQueryProcessor()


# Example usage and testing
if __name__ == "__main__":
    processor = ShadowrunQueryProcessor()

    # Test queries
    test_queries = [
        "Give me all the details and statistics about the Ares Predator gun",
        "What are the Matrix damage rules for IC attacks?",
        "How do I cast a fireball spell?",
        "Compare Ares Predator vs Colt Government 2066",
        "What is biofeedback damage?"
    ]

    for query in test_queries:
        print(processor.explain_query_processing(query))
        print("-" * 50)