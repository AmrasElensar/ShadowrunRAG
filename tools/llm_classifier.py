"""
Unified LLM-based classifier using qwen2.5:14b for consistent Shadowrun content classification.
Replaces tools/llm_classifier.py
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import ollama

logger = logging.getLogger(__name__)


class UnifiedShadowrunClassifier:
    """Unified classifier using qwen2.5:14b for all classification tasks."""

    def __init__(self, model_name: str = "qwen2.5:14b-instruct-q6_K", timeout: int = 45):
        self.model_name = model_name
        self.timeout = timeout
        self.fallback_classifier = PatternBasedFallback()

        # Test if model is available
        self._ensure_model_available()

        # Enhanced classification prompt for qwen2.5:14b
        self.classification_prompt = """You are an expert Shadowrun tabletop RPG content analyzer. Classify this content with high precision.

Content to classify:
{text}

CLASSIFICATION REQUIREMENTS:
1. PRIMARY SECTION: Choose the single most relevant category
2. CONTENT TYPE: Determine the exact type of content
3. RULE ASSESSMENT: Does this contain actual game mechanics?

PRIMARY SECTIONS (choose ONE most relevant):
- Matrix: Hacking, cybercombat, IC (including Black IC), programs, hosts, data processing, cyberdecks, Matrix damage, resistance tests
- Combat: Physical combat, damage, armor, weapons, initiative, condition monitors, wound modifiers
- Magic: Spells, spirits, astral projection, magical traditions, drain, force ratings, summoning
- Skills: Skill tests, dice pools, thresholds, specializations, defaulting, extended tests
- Gear: Equipment, weapons, armor, electronics, availability, costs, modifications
- Character_Creation: Priority system, attributes, skill points, metatypes, qualities, karma
- Riggers: Drones, vehicles, jumped-in control, vehicle combat, pilot programs
- Game_Mechanics: Core rules, Edge, limits, initiative, general test procedures
- Social: Contacts, reputation, etiquette, negotiation, lifestyle, social tests
- Setting: Background, corporations, locations, history, world lore

CONTENT TYPES:
- explicit_rule: Contains specific game mechanics, dice pools, test procedures, formulas
- example: Character stories, scenarios, sample gameplay situations
- table: Equipment lists, reference tables, stat blocks, costs
- narrative: Background text, flavor text, setting descriptions
- reference: Page numbers, cross-references, table of contents

CRITICAL MATRIX CONTENT DETECTION:
- If content mentions IC, Black IC, Matrix damage, resistance tests, or hacking mechanics → Matrix section
- If content contains dice pool formulas for Matrix actions → explicit_rule type
- Pay special attention to IC attack mechanics and Matrix damage resistance

Respond ONLY with valid JSON:
{{
  "primary_section": "Matrix",
  "content_type": "explicit_rule",
  "contains_rules": true,
  "contains_dice_pools": true,
  "contains_examples": false,
  "confidence": 0.95,
  "reasoning": "Contains IC attack mechanics with dice pool formulas",
  "mechanical_keywords": ["dice pool", "test", "resistance", "damage"],
  "specific_topics": ["Black IC", "Matrix damage", "resistance test"]
}}

RESPOND ONLY WITH THE JSON OBJECT. NO OTHER TEXT."""

    def _ensure_model_available(self):
        """Check if qwen2.5:14b is available."""
        try:
            test_response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": "Test classification"}],
                options={"num_predict": 10, "temperature": 0.1}
            )
            logger.info(f"Unified classifier using model: {self.model_name}")

        except Exception as e:
            logger.error(f"Model {self.model_name} not available: {e}")

            # Try qwen2.5 variants
            alternatives = [
                "qwen2.5:14b-instruct",
                "qwen2.5:7b-instruct",
                "qwen2.5:3b-instruct"
            ]

            for alt_model in alternatives:
                try:
                    ollama.chat(
                        model=alt_model,
                        messages=[{"role": "user", "content": "Test"}],
                        options={"num_predict": 5}
                    )
                    self.model_name = alt_model
                    logger.info(f"Using alternative model: {alt_model}")
                    return
                except:
                    continue

            logger.error("No suitable qwen2.5 model found. Using pattern fallback only.")
            self.model_name = None

    def classify_content(self, text: str, source: str) -> Dict[str, Any]:
        """Main classification method with enhanced Matrix detection."""

        # Quick filter for obvious index/reference content
        if self._is_index_content(text):
            return self._create_index_metadata(source)

        # Try LLM classification with qwen2.5:14b
        if self.model_name:
            llm_result = self._classify_with_llm(text)
            if llm_result and llm_result.get("confidence", 0) > 0.5:
                return self._create_metadata_from_llm_result(llm_result, text, source)

        # Enhanced pattern fallback with better Matrix detection
        logger.warning(f"Using enhanced pattern fallback for {source}")
        return self._enhanced_pattern_classification(text, source)

    def _classify_with_llm(self, text: str, max_retries: int = 2) -> Optional[Dict]:
        """Classify using qwen2.5:14b with retry logic."""

        # Truncate very long text to first 1500 chars for classification
        sample_text = text[:1500] if len(text) > 1500 else text

        for attempt in range(max_retries + 1):
            try:
                prompt = self.classification_prompt.format(text=sample_text)

                response = ollama.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    options={
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "num_predict": 300,
                        "stop": ["\n\n", "```"]
                    }
                )

                response_text = response["message"]["content"].strip()

                logger.info(f"LLM raw response attempt {attempt + 1}: {response_text[:200]}...")
                logger.info(f"Response length: {len(response_text)} chars")

                # Clean response - remove any markdown formatting
                response_text = re.sub(r'```json\s*', '', response_text)
                response_text = re.sub(r'```\s*$', '', response_text)

                logger.info(f"Cleaned response: {response_text[:200]}...")

                result = json.loads(response_text)

                # Validate required fields
                required_fields = ["primary_section", "content_type", "contains_rules", "confidence"]
                if all(field in result for field in required_fields):
                    return result
                else:
                    logger.warning(f"LLM result missing required fields: {result}")

            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error (attempt {attempt + 1}): {e}")
                logger.debug(f"Raw response: {response_text[:200]}")

            except Exception as e:
                logger.warning(f"LLM classification error (attempt {attempt + 1}): {e}")

            if attempt < max_retries:
                time.sleep(1)  # Brief pause before retry

        return None

    def _enhanced_pattern_classification(self, text: str, source: str) -> Dict[str, Any]:
        """Enhanced pattern-based fallback with better Matrix detection."""

        text_lower = text.lower()

        # Enhanced Matrix detection patterns
        matrix_patterns = {
            "ic_specific": ["black ic", "white ic", "gray ic", "ic attack", "types of ic"],
            "hacking": ["hack", "matrix action", "cyberdeck", "data processing"],
            "matrix_damage": ["matrix damage", "biofeedback", "dumpshock", "resistance test"],
            "matrix_tests": ["firewall", "sleaze", "attack rating", "matrix attributes"]
        }

        combat_patterns = ["damage", "armor", "weapon", "initiative", "attack roll"]
        magic_patterns = ["spell", "magic", "astral", "spirit", "drain"]
        skills_patterns = ["dice pool", "threshold", "skill test", "extended test"]

        # Score sections with enhanced Matrix detection
        section_scores = {}

        # Matrix scoring (enhanced for IC detection)
        matrix_score = 0
        for category, patterns in matrix_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    matrix_score += 3 if category == "ic_specific" else 1

        if matrix_score > 0:
            section_scores["Matrix"] = matrix_score

        # Other sections
        for pattern in combat_patterns:
            if pattern in text_lower:
                section_scores["Combat"] = section_scores.get("Combat", 0) + 1

        for pattern in magic_patterns:
            if pattern in text_lower:
                section_scores["Magic"] = section_scores.get("Magic", 0) + 1

        for pattern in skills_patterns:
            if pattern in text_lower:
                section_scores["Game_Mechanics"] = section_scores.get("Game_Mechanics", 0) + 1

        # Determine primary section
        if section_scores:
            primary_section = max(section_scores, key=section_scores.get)
        else:
            primary_section = "Setting"

        # Content type determination
        content_type = "explicit_rule" if any(term in text_lower for term in
                                            ["dice pool", "test:", "roll", "resistance"]) else "narrative"

        # Rule detection
        contains_rules = any(term in text_lower for term in
                           ["dice pool", "test", "roll", "damage", "threshold"])

        return {
            "source": source,
            "document_type": "rulebook",
            "edition": "SR5",
            "primary_section": primary_section,
            "content_type": content_type,
            "contains_rules": contains_rules,
            "is_rule_definition": contains_rules,
            "contains_dice_pools": "dice pool" in text_lower,
            "contains_examples": any(term in text_lower for term in ["example", "suppose", "let's say"]),
            "confidence": max(section_scores.values()) / 5.0 if section_scores else 0.3,
            "confidence_scores": section_scores,
            "mechanical_keywords": self._extract_keywords(text_lower),
            "specific_topics": self._extract_topics(text_lower),
            "classification_method": "enhanced_pattern"
        }

    def _create_metadata_from_llm_result(self, llm_result: Dict, text: str, source: str) -> Dict[str, Any]:
        """Convert LLM result to full metadata structure."""

        text_lower = text.lower()

        return {
            "source": source,
            "document_type": "rulebook",
            "edition": "SR5",
            "primary_section": llm_result["primary_section"],
            "content_type": llm_result["content_type"],
            "contains_rules": llm_result["contains_rules"],
            "is_rule_definition": llm_result["contains_rules"],
            "contains_dice_pools": llm_result.get("contains_dice_pools", "dice pool" in text_lower),
            "contains_examples": llm_result.get("contains_examples", False),
            "confidence": llm_result["confidence"],
            "mechanical_keywords": llm_result.get("mechanical_keywords", []),
            "specific_topics": llm_result.get("specific_topics", []),
            "classification_method": "llm_qwen2.5"
        }

    def _extract_keywords(self, text_lower: str) -> List[str]:
        """Extract mechanical keywords from text."""
        keywords = []
        patterns = ["dice pool", "test", "threshold", "modifier", "resistance", "damage", "armor"]
        for pattern in patterns:
            if pattern in text_lower:
                keywords.append(pattern)
        return keywords

    def _extract_topics(self, text_lower: str) -> List[str]:
        """Extract specific topics from text."""
        topics = []
        topic_patterns = ["black ic", "white ic", "matrix damage", "biofeedback", "firewall", "sleaze"]
        for topic in topic_patterns:
            if topic in text_lower:
                topics.append(topic)
        return topics

    def _is_index_content(self, text: str) -> bool:
        """Check if content is index/reference material."""
        text_lower = text.lower()
        index_indicators = ["table of contents", "index", "page", "see also", "refer to"]

        # Short content that's mostly page references
        if len(text) < 200 and any(indicator in text_lower for indicator in index_indicators):
            return True

        # High density of page references
        page_refs = len(re.findall(r'\b(?:page|p\.)\s*\d+', text_lower))
        if page_refs > 3 and len(text) < 500:
            return True

        return False

    def _create_index_metadata(self, source: str) -> Dict[str, Any]:
        """Create metadata for index content."""
        return {
            "source": source,
            "document_type": "reference",
            "edition": "SR5",
            "primary_section": "Reference",
            "content_type": "reference",
            "contains_rules": False,
            "is_rule_definition": False,
            "contains_dice_pools": False,
            "contains_examples": False,
            "confidence": 0.9,
            "classification_method": "index_filter"
        }


class PatternBasedFallback:
    """Basic pattern fallback for when LLM fails completely."""

    def __init__(self):
        self.section_patterns = {
            "Matrix": ["matrix", "ic", "hack", "cyberdeck", "firewall", "sleaze", "data processing"],
            "Combat": ["damage", "armor", "weapon", "initiative", "attack", "condition monitor"],
            "Magic": ["spell", "magic", "astral", "spirit", "mage", "drain", "force"],
            "Skills": ["test", "dice pool", "threshold", "skill", "specialization"],
            "Gear": ["gear", "equipment", "cost", "availability", "rating"]
        }

    def classify_content(self, text: str, source: str) -> Dict[str, Any]:
        """Emergency fallback classification."""

        text_lower = text.lower()

        section_scores = {}
        for section, patterns in self.section_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text_lower)
            if score > 0:
                section_scores[section] = score

        primary_section = max(section_scores, key=section_scores.get) if section_scores else "Setting"

        return {
            "source": source,
            "document_type": "rulebook",
            "edition": "SR5",
            "primary_section": primary_section,
            "content_type": "explicit_rule" if "dice pool" in text_lower else "narrative",
            "contains_rules": "dice pool" in text_lower or "test" in text_lower,
            "is_rule_definition": "dice pool" in text_lower or "test" in text_lower,
            "contains_dice_pools": "dice pool" in text_lower,
            "contains_examples": False,
            "confidence": 0.4,
            "classification_method": "emergency_fallback"
        }


# Factory function for indexer.py integration
def create_llm_classifier(model_name: str = "qwen2.5:14b-instruct-q6_K") -> UnifiedShadowrunClassifier:
    """Create unified classifier instance."""
    return UnifiedShadowrunClassifier(model_name=model_name)