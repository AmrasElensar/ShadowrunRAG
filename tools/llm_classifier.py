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

from .verified_shadowrun_patterns import (
    VERIFIED_SHADOWRUN_PATTERNS,
    detect_verified_magic_content,
    detect_verified_rigger_content,
    detect_verified_combat_content,
    detect_verified_social_content,
    detect_verified_character_creation_content,
    detect_verified_skills_content,
    create_verified_detector_set
)

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
- Weapon types: pistol, rifle, shotgun, SMG, assault rifle, sniper rifle, machine gun, melee weapons

CRITICAL MATRIX CONTENT DETECTION:
- If content mentions IC programs, Matrix damage, biofeedback, or hacking mechanics → Matrix section
- If content contains cyberdeck specifications or Matrix attribute arrays → Matrix section
- Matrix-specific terms: IC, biofeedback, dumpshock, firewall, sleaze, data processing

WEAPON vs MATRIX DISAMBIGUATION:
- Weapon damage (8P, 9P, etc.) with AP values → Gear section
- Matrix damage with resistance tests → Matrix section
- Physical weapon descriptions → Gear section  
- Virtual/electronic combat → Matrix section

Respond ONLY with valid JSON:
{{
  "primary_section": "Gear",
  "content_type": "table", 
  "contains_rules": true,
  "contains_dice_pools": false,
  "contains_examples": false,
  "confidence": 0.95,
  "reasoning": "Contains weapon statistics table with ACC/Damage/AP values",
  "mechanical_keywords": ["weapon", "damage", "accuracy", "armor penetration"],
  "specific_topics": ["Ares Predator", "heavy pistols", "weapon statistics"]
}}

RESPOND ONLY WITH THE JSON OBJECT. NO OTHER TEXT."""


class TwoTierShadowrunClassifier:
    """Two-tier classification: LLM reasoning + pattern-based corrections."""

    def __init__(self, model_name: str = "qwen2.5:14b-instruct-q6_K"):
        self.model_name = model_name
        self.llm_classifier = EnhancedLLMClassifier(model_name)
        self.pattern_corrector = PatternBasedCorrector()

        # Weapon detection patterns (high confidence)
        self.weapon_patterns = {
            "manufacturers": ["ares", "colt", "ruger", "browning", "remington", "defiance",
                              "yamaha", "hk", "fichetti", "beretta", "taurus", "steyr"],
            "weapon_types": ["pistol", "rifle", "shotgun", "smg", "assault rifle", "sniper rifle",
                             "machine gun", "crossbow", "sword", "blade", "knife", "club"],
            "weapon_stats": ["acc", "damage", "ap", "mode", "rc", "ammo", "avail", "cost"],
            "specific_weapons": ["predator", "government 2066", "ultra-power", "roomsweeper",
                                 "super warhawk", "crusader", "black scorpion", "ak-97"],
            "weapon_accessories": ["smartgun", "laser sight", "silencer", "scope", "foregrip"],
            "damage_patterns": [r'\d+P\s*(?:\(f\))?', r'AP\s*-\d+', r'\d+S\s*\(e\)']
        }

        # Matrix detection patterns
        self.matrix_patterns = {
            "ic_programs": ["black ic", "white ic", "gray ic", "ic attack", "killer ic", "marker ic"],
            "matrix_damage": ["matrix damage", "biofeedback", "dumpshock", "link-lock"],
            "hacking_terms": ["cyberdeck", "firewall", "sleaze", "data processing", "matrix attributes"],
            "matrix_actions": ["hack on the fly", "brute force", "data spike", "crash program"]
        }

        # Add comprehensive patterns
        self.verified_patterns = VERIFIED_SHADOWRUN_PATTERNS
        self.verified_detectors = create_verified_detector_set()

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

    def _detect_weapon_content(self, text: str) -> Dict[str, Any]:
        """Detect weapon-specific content with high confidence."""
        text_lower = text.lower()
        weapon_indicators = {
            "manufacturers_found": [],
            "weapon_types_found": [],
            "weapon_stats_found": [],
            "specific_weapons_found": [],
            "damage_patterns_found": [],
            "confidence_score": 0.0
        }

        # Check for manufacturers
        for manufacturer in self.weapon_patterns["manufacturers"]:
            if manufacturer in text_lower:
                weapon_indicators["manufacturers_found"].append(manufacturer)
                weapon_indicators["confidence_score"] += 0.2

        # Check for weapon types
        for weapon_type in self.weapon_patterns["weapon_types"]:
            if weapon_type in text_lower:
                weapon_indicators["weapon_types_found"].append(weapon_type)
                weapon_indicators["confidence_score"] += 0.15

        # Check for weapon statistics table columns
        for stat in self.weapon_patterns["weapon_stats"]:
            if stat in text_lower:
                weapon_indicators["weapon_stats_found"].append(stat)
                weapon_indicators["confidence_score"] += 0.1

        # Check for specific weapon names
        for weapon in self.weapon_patterns["specific_weapons"]:
            if weapon in text_lower:
                weapon_indicators["specific_weapons_found"].append(weapon)
                weapon_indicators["confidence_score"] += 0.25

        # Check for damage patterns (weapon damage codes)
        for pattern in self.weapon_patterns["damage_patterns"]:
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
            for pattern in patterns:
                if pattern in text_lower:
                    matrix_indicators[found_key].append(pattern)
                    matrix_indicators["confidence_score"] += 0.2

        return matrix_indicators


class EnhancedLLMClassifier:
    """LLM classifier with improved prompt."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        # Use the enhanced prompt from the previous artifact
        self.classification_prompt = ENHANCED_CLASSIFICATION_PROMPT  # From previous artifact

    def classify_with_llm(self, text: str, max_retries: int = 2) -> Optional[Dict]:
        """Classify using enhanced LLM prompt."""

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

                # Clean response - remove any markdown formatting
                response_text = re.sub(r'```json\s*', '', response_text)
                response_text = re.sub(r'```\s*$', '', response_text)

                result = json.loads(response_text)

                # Validate required fields
                required_fields = ["primary_section", "content_type", "contains_rules", "confidence"]
                if all(field in result for field in required_fields):
                    return result
                else:
                    logger.warning(f"LLM result missing required fields: {result}")

            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error (attempt {attempt + 1}): {e}")

            except Exception as e:
                logger.warning(f"LLM classification error (attempt {attempt + 1}): {e}")

            if attempt < max_retries:
                time.sleep(1)

        return None


class PatternBasedCorrector:
    """Pattern-based corrections and enhancements to LLM results."""

    def __init__(self):
        self.classifier = TwoTierShadowrunClassifier.__new__(TwoTierShadowrunClassifier)
        self.classifier.__init__()  # Get weapon/matrix patterns

    def apply_corrections(self, text: str, llm_result: Dict, source: str) -> Dict[str, Any]:
        """Apply pattern-based corrections to LLM classification."""

        # Check for weapon content override
        weapon_indicators = self.classifier._detect_weapon_content(text)
        matrix_indicators = self.classifier._detect_matrix_content(text)

        # High-confidence weapon override
        if weapon_indicators["confidence_score"] >= 0.4:
            if llm_result["primary_section"] != "Gear":
                logger.info(
                    f"Pattern override: Weapon content detected → Gear (confidence: {weapon_indicators['confidence_score']:.2f})")
                llm_result["primary_section"] = "Gear"
                llm_result["confidence"] = min(0.9, weapon_indicators["confidence_score"])
                llm_result["classification_method"] = "pattern_corrected"

                # Update content type if weapon table detected
                if len(weapon_indicators["weapon_stats_found"]) >= 3:
                    llm_result["content_type"] = "table"

        # High-confidence Matrix override
        elif matrix_indicators["confidence_score"] >= 0.5:
            if llm_result["primary_section"] != "Matrix":
                logger.info(
                    f"Pattern override: Matrix content detected → Matrix (confidence: {matrix_indicators['confidence_score']:.2f})")
                llm_result["primary_section"] = "Matrix"
                llm_result["confidence"] = min(0.9, matrix_indicators["confidence_score"])
                llm_result["classification_method"] = "pattern_corrected"

        # Enhance keywords with pattern findings
        enhanced_keywords = list(llm_result.get("mechanical_keywords", []))
        enhanced_topics = list(llm_result.get("specific_topics", []))

        # Add weapon-specific keywords
        if weapon_indicators["confidence_score"] > 0.2:
            enhanced_keywords.extend(weapon_indicators["weapon_types_found"])
            enhanced_topics.extend(weapon_indicators["specific_weapons_found"])
            enhanced_topics.extend(weapon_indicators["manufacturers_found"])

        # Add matrix-specific keywords
        if matrix_indicators["confidence_score"] > 0.2:
            enhanced_keywords.extend(matrix_indicators["hacking_terms_found"])
            enhanced_topics.extend(matrix_indicators["ic_programs_found"])

        llm_result["mechanical_keywords"] = list(set(enhanced_keywords))
        llm_result["specific_topics"] = list(set(enhanced_topics))

        # Add source and document metadata
        llm_result.update({
            "source": source,
            "document_type": "rulebook",
            "edition": "SR5",
            "is_rule_definition": llm_result["contains_rules"],
            "contains_dice_pools": llm_result.get("contains_dice_pools", False),
            "contains_examples": llm_result.get("contains_examples", False)
        })

        return llm_result

    def classify_with_patterns(self, text: str, source: str) -> Dict[str, Any]:
        """Emergency fallback: pure pattern classification."""

        weapon_indicators = self.classifier._detect_weapon_content(text)
        matrix_indicators = self.classifier._detect_matrix_content(text)

        if weapon_indicators["confidence_score"] > matrix_indicators["confidence_score"]:
            primary_section = "Gear"
            confidence = weapon_indicators["confidence_score"]
        elif matrix_indicators["confidence_score"] > 0.3:
            primary_section = "Matrix"
            confidence = matrix_indicators["confidence_score"]
        else:
            primary_section = "Setting"
            confidence = 0.3

        return {
            "source": source,
            "document_type": "rulebook",
            "edition": "SR5",
            "primary_section": primary_section,
            "content_type": "explicit_rule" if "test" in text.lower() else "narrative",
            "contains_rules": "dice pool" in text.lower() or "test" in text.lower(),
            "is_rule_definition": "dice pool" in text.lower(),
            "contains_dice_pools": "dice pool" in text.lower(),
            "contains_examples": "example" in text.lower(),
            "confidence": confidence,
            "classification_method": "pattern_fallback",
            "mechanical_keywords": [],
            "specific_topics": []
        }


# Factory function for compatibility
def create_two_tier_classifier(model_name: str = "qwen2.5:14b-instruct-q6_K") -> TwoTierShadowrunClassifier:
    """Create two-tier classifier instance."""
    return TwoTierShadowrunClassifier(model_name=model_name)