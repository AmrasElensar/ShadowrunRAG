"""
Unified two-tier Shadowrun content classifier.
Consolidates all classification logic from scattered files.
"""

import json
import logging
import re
import time
from typing import Dict, List, Any, Optional
import ollama

logger = logging.getLogger(__name__)

ENHANCED_CLASSIFICATION_PROMPT = """You are an expert Shadowrun tabletop RPG content analyzer. Classify this content with high precision.

Content to classify:
{text}

CLASSIFICATION REQUIREMENTS:
1. PRIMARY SECTION: Choose the single most relevant category
2. CONTENT TYPE: Determine the exact type of content  
3. RULE ASSESSMENT: Does this contain actual game mechanics?

PRIMARY SECTIONS (choose ONE most relevant):
- Gear: **WEAPONS (Ares, Colt, Ruger, Browning, pistols, rifles, shotguns, melee weapons), armor, equipment, electronics, modifications, costs, availability, weapon statistics tables**
- Combat: Physical combat mechanics, damage resolution, initiative, condition monitors, wound modifiers, melee combat rules
- Matrix: Hacking, cybercombat, IC programs, hosts, data processing, cyberdecks, Matrix damage, biofeedback, resistance tests
- Magic: Spells, spirits, astral projection, magical traditions, drain, force ratings, summoning, magical combat
- Skills: Skill tests, dice pools, thresholds, specializations, defaulting, extended tests, skill descriptions
- Character_Creation: Priority system, attributes, skill points, metatypes, qualities, karma, character building
- Riggers: Drones, vehicles, jumped-in control, vehicle combat, pilot programs, rigger gear
- Game_Mechanics: Core rules, Edge, limits, initiative, general test procedures, basic game concepts
- Social: Contacts, reputation, etiquette, negotiation, lifestyle, social tests, networking
- Setting: Background, corporations, locations, history, world lore, timeline, NPCs

CONTENT TYPES:
- table: **Weapon statistics (ACC/Damage/AP/Mode/RC/Ammo/Avail/Cost), equipment lists, reference tables, stat blocks**
- explicit_rule: Contains specific game mechanics, dice pools, test procedures, formulas, rules explanations
- example: Character stories, scenarios, sample gameplay situations, "for example" content
- narrative: Background text, flavor text, setting descriptions, corporate information
- reference: Page numbers, cross-references, table of contents, indices

CRITICAL WEAPON DETECTION (HIGHEST PRIORITY):
- If content contains weapon statistics tables with columns like ACC, Damage, AP, Mode → Gear section, table type
- If content mentions specific weapons (Ares Predator, Colt Government, Browning Ultra-Power, etc.) → Gear section
- If content describes weapon capabilities, modifications, or accessories → Gear section
- Weapon manufacturers: Ares, Colt, Ruger, Browning, HK, Remington, Defiance, Yamaha, etc.

Respond with valid JSON only:
{{
    "primary_section": "section_name",
    "content_type": "type_name", 
    "contains_rules": true/false,
    "is_rule_definition": true/false,
    "contains_dice_pools": true/false,
    "contains_examples": true/false,
    "confidence": 0.0-1.0,
    "mechanical_keywords": ["keyword1", "keyword2"],
    "specific_topics": ["topic1", "topic2"]
}}"""


class EnhancedLLMClassifier:
    """LLM-based semantic classifier with retry logic."""

    def __init__(self, model_name: str = "qwen2.5:14b-instruct-q6_K"):
        self.model_name = model_name
        self.max_retries = 2

    def classify_with_llm(self, text: str) -> Optional[Dict[str, Any]]:
        """Classify content using LLM with structured output."""

        if len(text.strip()) < 20:
            return None

        # Truncate very long text to prevent token limits
        if len(text) > 4000:
            text = text[:4000] + "..."

        prompt = ENHANCED_CLASSIFICATION_PROMPT.format(text=text)

        for attempt in range(self.max_retries):
            try:
                response = ollama.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    options={
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "num_predict": 300
                    }
                )

                content = response['message']['content'].strip()

                # Extract JSON from response
                content = content.replace('```json', '').replace('```', '').strip()

                result = json.loads(content)

                # Validate required fields
                required_fields = ["primary_section", "content_type", "contains_rules"]
                if all(field in result for field in required_fields):
                    result["classification_method"] = "llm_semantic"
                    return result
                else:
                    logger.warning(f"LLM response missing required fields: {result}")

            except json.JSONDecodeError as e:
                logger.warning(f"LLM JSON parse error (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to parse LLM response after {self.max_retries} attempts")
            except Exception as e:
                logger.warning(f"LLM classification error (attempt {attempt + 1}): {e}")
                time.sleep(1)  # Brief pause before retry

        return None


class PatternBasedCorrector:
    """Pattern-based corrections using verified Shadowrun patterns."""

    def __init__(self):
        # Try to import verified patterns, fallback to basic patterns
        try:
            from .verified_shadowrun_patterns import VERIFIED_SHADOWRUN_PATTERNS, create_verified_detector_set
            self.verified_patterns = VERIFIED_SHADOWRUN_PATTERNS
            self.verified_detectors = create_verified_detector_set()
        except ImportError:
            logger.warning("Verified patterns not found, using basic patterns")
            self.verified_patterns = self._get_basic_patterns()
            self.verified_detectors = {}

        # Extract patterns for direct access
        self.weapon_patterns = self.verified_patterns.get("gear", {})
        self.matrix_patterns = self.verified_patterns.get("matrix", {})
        self.magic_patterns = self.verified_patterns.get("magic", {})

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

    def apply_corrections(self, text: str, llm_result: Dict, source: str) -> Dict[str, Any]:
        """Apply pattern-based corrections to LLM classification."""

        # Start with LLM result
        corrected_result = llm_result.copy()
        corrected_result["source"] = source
        corrected_result["document_type"] = "rulebook"
        corrected_result["edition"] = "SR5"

        # Run pattern detection
        weapon_indicators = self._detect_weapon_content(text)
        matrix_indicators = self._detect_matrix_content(text)
        magic_indicators = self._detect_magic_content(text)

        # Pattern-based corrections (high confidence overrides)
        if weapon_indicators["confidence_score"] > 0.5:
            corrected_result["primary_section"] = "Gear"
            corrected_result["pattern_weapon_confidence"] = weapon_indicators["confidence_score"]

            # Weapon table detection
            if weapon_indicators.get("weapon_stats_found") and len(weapon_indicators["weapon_stats_found"]) >= 3:
                corrected_result["content_type"] = "table"
                corrected_result["is_weapon_table"] = True

        elif matrix_indicators["confidence_score"] > 0.4:
            corrected_result["primary_section"] = "Matrix"
            corrected_result["pattern_matrix_confidence"] = matrix_indicators["confidence_score"]

        elif magic_indicators["confidence_score"] > 0.4:
            corrected_result["primary_section"] = "Magic"
            corrected_result["pattern_magic_confidence"] = magic_indicators["confidence_score"]

        # Enhanced metadata
        corrected_result.update({
            "pattern_detections": {
                "weapon_score": weapon_indicators["confidence_score"],
                "matrix_score": matrix_indicators["confidence_score"],
                "magic_score": magic_indicators["confidence_score"]
            },
            "specific_items_found": (
                    weapon_indicators.get("specific_weapons_found", []) +
                    matrix_indicators.get("ic_programs_found", []) +
                    magic_indicators.get("spells_found", [])
            )
        })

        return corrected_result

    def _detect_weapon_content(self, text: str) -> Dict[str, Any]:
        """Detect weapon-specific content patterns."""
        text_lower = text.lower()
        weapon_indicators = {
            "manufacturers_found": [],
            "weapon_types_found": [],
            "weapon_stats_found": [],
            "specific_weapons_found": [],
            "damage_patterns_found": [],
            "confidence_score": 0.0
        }

        # Check manufacturers
        for manufacturer in self.weapon_patterns.get("manufacturers", []):
            if manufacturer in text_lower:
                weapon_indicators["manufacturers_found"].append(manufacturer)
                weapon_indicators["confidence_score"] += 0.2

        # Check weapon types
        for weapon_type in self.weapon_patterns.get("weapon_types", []):
            if weapon_type in text_lower:
                weapon_indicators["weapon_types_found"].append(weapon_type)
                weapon_indicators["confidence_score"] += 0.15

        # Check weapon statistics
        for stat in self.weapon_patterns.get("weapon_stats", []):
            if stat in text_lower:
                weapon_indicators["weapon_stats_found"].append(stat)
                weapon_indicators["confidence_score"] += 0.1

        # Check specific weapons
        for weapon in self.weapon_patterns.get("specific_weapons", []):
            if weapon in text_lower:
                weapon_indicators["specific_weapons_found"].append(weapon)
                weapon_indicators["confidence_score"] += 0.25

        # Check damage patterns
        damage_patterns = [r'\d+P\s*(?:\(f\))?', r'AP\s*-\d+', r'\d+S\s*\(e\)']
        for pattern in damage_patterns:
            matches = re.findall(pattern, text)
            if matches:
                weapon_indicators["damage_patterns_found"].extend(matches)
                weapon_indicators["confidence_score"] += 0.2

        return weapon_indicators

    def _detect_matrix_content(self, text: str) -> Dict[str, Any]:
        """Detect Matrix-specific content patterns."""
        text_lower = text.lower()
        matrix_indicators = {
            "ic_programs_found": [],
            "matrix_damage_found": [],
            "hacking_terms_found": [],
            "matrix_actions_found": [],
            "confidence_score": 0.0
        }

        # Check each pattern category
        for category, patterns in self.matrix_patterns.items():
            found_key = f"{category}_found"
            if found_key in matrix_indicators:
                for pattern in patterns:
                    if pattern in text_lower:
                        matrix_indicators[found_key].append(pattern)
                        matrix_indicators["confidence_score"] += 0.2

        return matrix_indicators

    def _detect_magic_content(self, text: str) -> Dict[str, Any]:
        """Detect magic-specific content patterns."""
        text_lower = text.lower()
        magic_indicators = {
            "spells_found": [],
            "mechanics_found": [],
            "skills_found": [],
            "confidence_score": 0.0
        }

        # Check specific spells
        for spell in self.magic_patterns.get("specific_spells", []):
            if spell in text_lower:
                magic_indicators["spells_found"].append(spell)
                magic_indicators["confidence_score"] += 0.3

        # Check magic mechanics
        for mechanic in self.magic_patterns.get("spell_mechanics", []):
            if mechanic in text_lower:
                magic_indicators["mechanics_found"].append(mechanic)
                magic_indicators["confidence_score"] += 0.2

        # Check magic skills
        for skill in self.magic_patterns.get("magic_skills", []):
            if skill in text_lower:
                magic_indicators["skills_found"].append(skill)
                magic_indicators["confidence_score"] += 0.25

        return magic_indicators

    def classify_with_patterns(self, text: str, source: str) -> Dict[str, Any]:
        """Emergency fallback: pure pattern classification."""

        weapon_indicators = self._detect_weapon_content(text)
        matrix_indicators = self._detect_matrix_content(text)
        magic_indicators = self._detect_magic_content(text)

        # Determine primary section based on highest confidence
        scores = [
            ("Gear", weapon_indicators["confidence_score"]),
            ("Matrix", matrix_indicators["confidence_score"]),
            ("Magic", magic_indicators["confidence_score"])
        ]

        primary_section, confidence = max(scores, key=lambda x: x[1])

        if confidence < 0.3:
            primary_section = "Setting"
            confidence = 0.3

        return {
            "source": source,
            "document_type": "rulebook",
            "edition": "SR5",
            "primary_section": primary_section,
            "content_type": "explicit_rule" if "dice pool" in text.lower() else "narrative",
            "contains_rules": "dice pool" in text.lower() or "test" in text.lower(),
            "is_rule_definition": "dice pool" in text.lower(),
            "contains_dice_pools": "dice pool" in text.lower(),
            "contains_examples": "example" in text.lower(),
            "confidence": confidence,
            "classification_method": "pattern_fallback",
            "mechanical_keywords": [],
            "specific_topics": []
        }


class UnifiedShadowrunClassifier:
    """Unified two-tier classification: LLM reasoning + pattern-based corrections."""

    def __init__(self, model_name: str = "qwen2.5:14b-instruct-q6_K"):
        self.model_name = model_name
        self.llm_classifier = EnhancedLLMClassifier(model_name)
        self.pattern_corrector = PatternBasedCorrector()

    def classify_content(self, text: str, source: str) -> Dict[str, Any]:
        """Main classification method with two-tier approach."""

        # Tier 1: LLM semantic understanding
        llm_result = self.llm_classifier.classify_with_llm(text)

        if not llm_result:
            logger.warning("LLM classification failed, using pattern fallback")
            return self.pattern_corrector.classify_with_patterns(text, source)

        # Tier 2: Pattern-based corrections and enhancements
        corrected_result = self.pattern_corrector.apply_corrections(text, llm_result, source)

        return corrected_result


# Factory functions for integration
def create_unified_classifier(model_name: str = "qwen2.5:14b-instruct-q6_K") -> UnifiedShadowrunClassifier:
    """Create unified classifier instance."""
    return UnifiedShadowrunClassifier(model_name=model_name)


def create_two_tier_classifier(model_name: str = "qwen2.5:14b-instruct-q6_K") -> UnifiedShadowrunClassifier:
    """Backward compatibility function."""
    return create_unified_classifier(model_name)


def create_llm_classifier(model_name: str = "qwen2.5:14b-instruct-q6_K") -> UnifiedShadowrunClassifier:
    """Backward compatibility function."""
    return create_unified_classifier(model_name)