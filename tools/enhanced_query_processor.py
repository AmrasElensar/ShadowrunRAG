"""
Enhanced query processing system that improves retrieval targeting.
This addresses the core issue where "Ares Predator gun" returned Matrix content.
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
        # Weapon-related patterns
        self.weapon_patterns = {
            "manufacturers": {
                "ares": ["ares", "ares macrotechnology", "ares arms"],
                "colt": ["colt", "colt government", "colt america"],
                "ruger": ["ruger", "ruger super warhawk"],
                "browning": ["browning", "browning ultra-power"],
                "remington": ["remington", "remington roomsweeper"],
                "defiance": ["defiance"],
                "yamaha": ["yamaha", "yamaha raiden"],
                "fichetti": ["fichetti"],
                "beretta": ["beretta"],
                "steyr": ["steyr"],
                "hk": ["hk", "heckler", "koch"]
            },
            "weapon_types": {
                "pistols": ["pistol", "sidearm", "handgun", "heavy pistol", "light pistol"],
                "rifles": ["rifle", "assault rifle", "sniper rifle", "hunting rifle"],
                "shotguns": ["shotgun", "scattergun"],
                "smgs": ["smg", "submachine gun", "machine pistol"],
                "melee": ["sword", "blade", "knife", "katana", "club", "staff", "axe"]
            },
            "specific_weapons": {
                "predator": ["predator", "ares predator", "predator v"],
                "government": ["government 2066", "colt government"],
                "ultra_power": ["ultra-power", "browning ultra-power"],
                "roomsweeper": ["roomsweeper", "remington roomsweeper"],
                "warhawk": ["warhawk", "super warhawk", "ruger super warhawk"],
                "crusader": ["crusader", "ares crusader"],
                "ak97": ["ak-97", "ak97", "kalashnikov"],
                "alpha": ["ares alpha", "alpha assault rifle"]
            },
            "weapon_stats": ["accuracy", "damage", "ap", "armor penetration", "mode", "recoil", "ammo", "availability",
                             "cost"]
        }

        # Matrix-related patterns
        self.matrix_patterns = {
            "ic_types": ["black ic", "white ic", "gray ic", "killer ic", "marker ic", "patrol ic"],
            "hacking_terms": ["hack", "matrix", "cyberdeck", "decker", "spider", "host"],
            "matrix_attributes": ["firewall", "sleaze", "attack", "data processing"],
            "matrix_actions": ["hack on the fly", "brute force", "data spike", "crash program"],
            "matrix_damage": ["matrix damage", "biofeedback", "dumpshock", "link-lock"]
        }

        # Magic-related patterns
        self.magic_patterns = {
            "spell_types": ["spell", "cantrip", "ritual", "enchantment"],
            "spirit_types": ["spirit", "elemental", "nature spirit", "man spirit"],
            "magic_terms": ["mage", "shaman", "adept", "mystic adept", "awakened"],
            "astral": ["astral", "astral projection", "astral space", "astral sight"]
        }

        # Query intent patterns
        self.intent_patterns = {
            "stats_lookup": ["stats", "statistics", "details", "information about", "tell me about"],
            "comparison": ["vs", "versus", "compare", "better than", "difference between"],
            "rules_question": ["how to", "rules for", "how does", "can i", "is it possible"],
            "cost_availability": ["cost", "price", "how much", "availability", "where to buy"]
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
            "skill": 0,
            "character": 0,
            "setting": 0
        }

        # Score weapon intent
        for category, patterns in self.weapon_patterns.items():
            if isinstance(patterns, dict):
                for subcategory, terms in patterns.items():
                    for term in terms:
                        if term in query:
                            intent_scores["weapon"] += 2
            else:
                for term in patterns:
                    if term in query:
                        intent_scores["weapon"] += 1

        # Score matrix intent
        for category, patterns in self.matrix_patterns.items():
            for term in patterns:
                if term in query:
                    intent_scores["matrix"] += 2

        # Score magic intent
        for category, patterns in self.magic_patterns.items():
            for term in patterns:
                if term in query:
                    intent_scores["magic"] += 2

        # Additional scoring for context clues
        if any(word in query for word in ["gun", "weapon", "pistol", "rifle", "blade"]):
            intent_scores["weapon"] += 1

        if any(word in query for word in ["hack", "cyberdeck", "matrix"]):
            intent_scores["matrix"] += 1

        if any(word in query for word in ["spell", "magic", "spirit"]):
            intent_scores["magic"] += 1

        # Return highest scoring intent
        if max(intent_scores.values()) == 0:
            return "general"

        return max(intent_scores, key=intent_scores.get)

    def _extract_entities(self, query: str) -> List[str]:
        """Extract specific entities mentioned in the query."""

        entities = []

        # Extract weapon entities
        for category, patterns in self.weapon_patterns.items():
            if isinstance(patterns, dict):
                for subcategory, terms in patterns.items():
                    for term in terms:
                        if term in query:
                            entities.append(term)
            else:
                for term in patterns:
                    if term in query:
                        entities.append(term)

        # Extract matrix entities
        for category, patterns in self.matrix_patterns.items():
            for term in patterns:
                if term in query:
                    entities.append(term)

        # Extract magic entities
        for category, patterns in self.magic_patterns.items():
            for term in patterns:
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

        if intent == "weapon":
            filters = {
                "primary_section": {"$eq": "Gear"},
                "$or": [
                    {"content_type": {"$eq": "table"}},
                    {"contains_rules": {"$eq": True}}
                ]
            }

            # Add weapon-specific boost terms
            boost_terms.extend([
                "weapon", "damage", "accuracy", "ap", "mode", "cost", "availability"
            ])

            # Add specific entities as high-priority boost terms
            boost_terms.extend(entities)

        elif intent == "matrix":
            filters = {
                "primary_section": {"$eq": "Matrix"},
                "$or": [
                    {"contains_rules": {"$eq": True}},
                    {"content_type": {"$eq": "explicit_rule"}}
                ]
            }

            boost_terms.extend([
                "matrix", "ic", "hack", "cyberdeck", "firewall", "sleaze"
            ])
            boost_terms.extend(entities)

        elif intent == "magic":
            filters = {
                "primary_section": {"$eq": "Magic"},
                "$or": [
                    {"contains_rules": {"$eq": True}},
                    {"content_type": {"$eq": "explicit_rule"}}
                ]
            }

            boost_terms.extend([
                "magic", "spell", "spirit", "astral", "mage", "drain"
            ])
            boost_terms.extend(entities)

        else:
            # General query - no strict filters
            filters = {}
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