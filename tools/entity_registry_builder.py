"""
Updated Entity Registry Builder for Shadowrun RAG System
Now uses improved patterns for better entity extraction.
"""

import re
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

# Import the improved patterns
from tools.verified_shadowrun_patterns import IMPROVED_SHADOWRUN_PATTERNS, create_improved_detector_set

logger = logging.getLogger(__name__)


@dataclass
class WeaponStats:
    """Structured weapon statistics."""
    name: str
    accuracy: str
    damage: str
    ap: str
    mode: str
    rc: str
    ammo: str
    avail: str
    cost: str
    category: str  # "Heavy Pistols", "Assault Rifles", etc.
    description: str = ""
    manufacturer: str = ""


@dataclass
class SpellStats:
    """Structured spell statistics."""
    name: str
    spell_type: str  # M or P
    range: str
    damage: str
    duration: str
    drain: str
    keywords: List[str]
    category: str  # "Combat", "Detection", etc.
    description: str = ""


@dataclass
class ICStats:
    """Structured IC program statistics."""
    name: str
    attack_pattern: str
    resistance_test: str
    effect_description: str
    category: str = "IC"


class EntityRegistryBuilder:
    """Builds entity registries from Shadowrun content using improved patterns."""

    def __init__(self):
        # Load improved detection patterns
        self.patterns = IMPROVED_SHADOWRUN_PATTERNS
        self.detectors = create_improved_detector_set()

        # Weapon table pattern - matches the exact table format
        self.weapon_table_pattern = re.compile(
            r'\|\s*([^|]+?)\s*\|\s*(\d+(?:\s*\(\d+\))?)\s*\|\s*(\d+[PS](?:\s*\([^)]+\))?)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|',
            re.MULTILINE
        )

        # Enhanced spell pattern using improved patterns
        self.spell_pattern = re.compile(
            r'([A-Z][A-Z\s]+?)\s*(?:\([^)]+\))?\s*\*\*Type:\*\*\s*([MP])\s*;\s*\*\*Range:\*\*\s*([^;]+?)\s*;(?:\s*\*\*Damage:\*\*\s*([^;]+?)\s*;)?\s*\*\*Duration:\*\*\s*([^;]+?)\s*;\s*\*\*Drain:\*\*\s*([^\n]+)',
            re.MULTILINE | re.IGNORECASE
        )

        # Enhanced IC pattern using improved patterns
        self.ic_pattern = re.compile(
            r'([A-Z][A-Z\s]+?)\s*\*\*Attack:\*\*\s*([^v]+?)\s*v\.\s*([^\n]+)',
            re.MULTILINE
        )

        # Weapon description pattern
        self.weapon_desc_pattern = re.compile(
            r'\*\*([^:]+?):\*\*\s*([^*]+?)(?=\*\*|$)',
            re.MULTILINE | re.DOTALL
        )

        # Enhanced patterns from improved detection
        self.weapon_categories = [
            "LIGHT PISTOLS", "HEAVY PISTOLS", "MACHINE PISTOLS",
            "SUBMACHINE GUNS", "ASSAULT RIFLES", "SNIPER RIFLES",
            "SHOTGUNS", "SPECIAL WEAPONS"
        ]

        self.spell_categories = [
            "Combat", "Detection", "Health", "Illusion", "Manipulation", "Environmental"
        ]

        logger.info("Enhanced Entity Registry Builder initialized with improved patterns")

    def extract_entities_from_chunk(self, chunk: Dict[str, Any]) -> Dict[str, List]:
        """Extract all entities from a single chunk using improved detection."""
        text = chunk.get("text", "")
        source = chunk.get("source", "")

        entities = {
            "weapons": [],
            "spells": [],
            "ic_programs": []
        }

        # Use improved content detection
        weapon_detection = self.detectors["weapon_detector"](text)
        spell_detection = self.detectors["spell_detector"](text)
        ic_detection = self.detectors["ic_detector"](text)

        # Extract weapons if detected
        if weapon_detection["is_weapon_content"]:
            logger.debug(f"Weapon content detected (confidence: {weapon_detection['confidence']:.2f})")
            entities["weapons"] = self._extract_weapons(text, source)

        # Extract spells if detected (improved detection)
        if spell_detection["is_spell_content"]:
            logger.debug(f"Spell content detected (confidence: {spell_detection['confidence']:.2f})")
            entities["spells"] = self._extract_spells(text, source, spell_detection)

        # Extract IC if detected (improved detection)
        if ic_detection["is_ic_content"]:
            logger.debug(f"IC content detected (confidence: {ic_detection['confidence']:.2f})")
            entities["ic_programs"] = self._extract_ic_programs(text, source, ic_detection)

        return entities

    def _extract_weapons(self, text: str, source: str) -> List[WeaponStats]:
        """Extract weapon statistics from text (existing logic)."""
        weapons = []

        # Find current weapon category
        current_category = "Unknown"
        for category in self.weapon_categories:
            if category in text.upper():
                current_category = category.title()
                break

        # Extract from tables
        table_matches = self.weapon_table_pattern.findall(text)
        for match in table_matches:
            if len(match) >= 8:
                # Skip header rows
                if "WEAPON" in match[0].upper() or "FIREARM" in match[0].upper():
                    continue

                weapon = WeaponStats(
                    name=match[0].strip(),
                    accuracy=match[1].strip(),
                    damage=match[2].strip(),
                    ap=match[3].strip(),
                    mode=match[4].strip(),
                    rc=match[5].strip(),
                    ammo=match[6].strip(),
                    avail=match[7].strip(),
                    cost=match[8].strip() if len(match) > 8 else "",
                    category=current_category
                )

                # Extract manufacturer from name
                weapon.manufacturer = self._extract_manufacturer(weapon.name)
                weapons.append(weapon)

        # Extract weapon descriptions
        desc_matches = self.weapon_desc_pattern.findall(text)
        for weapon in weapons:
            for desc_name, desc_text in desc_matches:
                if weapon.name.lower() in desc_name.lower():
                    weapon.description = desc_text.strip()
                    break

        return weapons

    def _extract_spells(self, text: str, source: str, detection_info: Dict = None) -> List[SpellStats]:
        """Extract spell statistics using improved patterns."""
        spells = []

        # Method 1: Use formal spell patterns (existing)
        formal_matches = self.spell_pattern.findall(text)
        for match in formal_matches:
            if len(match) >= 6:
                spell_name = match[0].strip().title()
                spell_type = match[1]
                range_val = match[2].strip()
                damage = match[3].strip() if match[3] else ""
                duration = match[4].strip()
                drain = match[5].strip()

                spell = SpellStats(
                    name=spell_name,
                    spell_type=spell_type,
                    range=range_val,
                    damage=damage,
                    duration=duration,
                    drain=drain,
                    keywords=[],
                    category=self._determine_spell_category(spell_name, text),
                    description=self._extract_spell_description(spell_name, text)
                )
                spells.append(spell)

        # Method 2: Use improved pattern matching for spell names
        if detection_info and detection_info.get("found_spells"):
            known_spells = self.patterns["magic"]["specific_spells"]
            text_upper = text.upper()

            for spell_name in detection_info["found_spells"]:
                # Skip if already found by formal pattern
                if any(s.name.lower() == spell_name.lower() for s in spells):
                    continue

                # Look for spell name in text and try to extract basic info
                if spell_name.upper() in text_upper:
                    # Try to find spell format around this name
                    spell_context = self._extract_spell_context(spell_name, text)
                    if spell_context:
                        spell = SpellStats(
                            name=spell_name.title(),
                            spell_type=spell_context.get("type", "Unknown"),
                            range=spell_context.get("range", "Unknown"),
                            damage=spell_context.get("damage", ""),
                            duration=spell_context.get("duration", "Unknown"),
                            drain=spell_context.get("drain", "Unknown"),
                            keywords=spell_context.get("keywords", []),
                            category=self._determine_spell_category(spell_name, text),
                            description=spell_context.get("description", "")
                        )
                        spells.append(spell)

        return spells

    def _extract_ic_programs(self, text: str, source: str, detection_info: Dict = None) -> List[ICStats]:
        """Extract IC program statistics using improved patterns."""
        ic_programs = []

        # Method 1: Use formal IC attack patterns (existing)
        formal_matches = self.ic_pattern.findall(text)
        for match in formal_matches:
            if len(match) >= 3:
                ic_name = match[0].strip()
                attack_pattern = match[1].strip()
                resistance_test = match[2].strip()

                ic = ICStats(
                    name=ic_name,
                    attack_pattern=attack_pattern,
                    resistance_test=resistance_test,
                    effect_description=self._extract_ic_description(ic_name, text),
                    category="IC"
                )
                ic_programs.append(ic)

        # Method 2: Use improved pattern matching for known IC types
        if detection_info and detection_info.get("found_programs"):
            text_upper = text.upper()

            for ic_name in detection_info["found_programs"]:
                # Skip if already found by formal pattern
                if any(ic.name.lower() == ic_name.lower() for ic in ic_programs):
                    continue

                # Look for IC name and try to extract attack info
                if ic_name.upper() in text_upper:
                    ic_context = self._extract_ic_context(ic_name, text)
                    if ic_context:
                        ic = ICStats(
                            name=ic_name.title(),
                            attack_pattern=ic_context.get("attack_pattern", "Unknown"),
                            resistance_test=ic_context.get("resistance_test", "Unknown"),
                            effect_description=ic_context.get("description", ""),
                            category="IC"
                        )
                        ic_programs.append(ic)

        return ic_programs

    def _extract_spell_context(self, spell_name: str, text: str) -> Optional[Dict[str, str]]:
        """Extract spell context from surrounding text."""
        # Look for spell format patterns around the spell name
        spell_patterns = self.patterns["magic"]["spell_format_patterns"]

        for pattern in spell_patterns:
            # Search for pattern near the spell name
            context_pattern = f"{re.escape(spell_name)}.*?{pattern}"
            match = re.search(context_pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                context_text = match.group()
                return self._parse_spell_format(context_text)

        return None

    def _extract_ic_context(self, ic_name: str, text: str) -> Optional[Dict[str, str]]:
        """Extract IC context from surrounding text."""
        # Look for attack patterns around the IC name
        attack_patterns = self.patterns["matrix"]["ic_attack_patterns"]

        for pattern in attack_patterns:
            # Search for pattern near the IC name
            context_pattern = f"{re.escape(ic_name)}.*?{pattern}"
            match = re.search(context_pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                context_text = match.group()
                return self._parse_ic_format(context_text)

        return None

    def _parse_spell_format(self, context_text: str) -> Dict[str, str]:
        """Parse spell format from context text."""
        result = {}

        # Extract Type
        type_match = re.search(r'Type:\*\*\s*([MP])', context_text, re.IGNORECASE)
        if type_match:
            result["type"] = type_match.group(1)

        # Extract Range
        range_match = re.search(r'Range:\*\*\s*([^;]+)', context_text, re.IGNORECASE)
        if range_match:
            result["range"] = range_match.group(1).strip()

        # Extract Duration
        duration_match = re.search(r'Duration:\*\*\s*([^;]+)', context_text, re.IGNORECASE)
        if duration_match:
            result["duration"] = duration_match.group(1).strip()

        # Extract Drain
        drain_match = re.search(r'Drain:\*\*\s*([^\n]+)', context_text, re.IGNORECASE)
        if drain_match:
            result["drain"] = drain_match.group(1).strip()

        return result

    def _parse_ic_format(self, context_text: str) -> Dict[str, str]:
        """Parse IC format from context text."""
        result = {}

        # Extract attack pattern
        attack_match = re.search(r'Attack:\*\*\s*([^v]+?)\s*v\.', context_text, re.IGNORECASE)
        if attack_match:
            result["attack_pattern"] = attack_match.group(1).strip()

        # Extract resistance test
        resistance_match = re.search(r'v\.\s*([^\n]+)', context_text, re.IGNORECASE)
        if resistance_match:
            result["resistance_test"] = resistance_match.group(1).strip()

        return result

    def _determine_spell_category(self, spell_name: str, text: str) -> str:
        """Determine spell category from context."""
        spell_lower = spell_name.lower()
        text_lower = text.lower()

        # Combat spells
        if any(term in spell_lower for term in ["bolt", "ball", "touch", "shatter"]):
            return "Combat"

        # Detection spells
        if any(term in spell_lower for term in ["detect", "analyze", "sense"]):
            return "Detection"

        # Manipulation spells
        if any(term in spell_lower for term in ["control", "levitate", "fingers", "armor"]):
            return "Manipulation"

        # Environmental spells
        if any(term in spell_lower for term in ["barrier", "light", "sheet"]):
            return "Environmental"

        # Health spells
        if any(term in spell_lower for term in ["heal", "cure", "increase", "decrease"]):
            return "Health"

        return "Unknown"

    def _extract_manufacturer(self, weapon_name: str) -> str:
        """Extract manufacturer from weapon name."""
        manufacturers = self.patterns["gear"]["manufacturers"]

        for mfg in manufacturers:
            if mfg.lower() in weapon_name.lower():
                return mfg.title()

        return ""

    def _extract_spell_description(self, spell_name: str, text: str) -> str:
        """Extract spell description from text."""
        # Look for description paragraph after spell name
        pattern = re.compile(f"{re.escape(spell_name)}.*?\n\n([^*]+?)(?=\n\n|\*\*|$)", re.DOTALL | re.IGNORECASE)
        match = pattern.search(text)

        if match:
            return match.group(1).strip()

        return ""

    def _extract_ic_description(self, ic_name: str, text: str) -> str:
        """Extract IC description from text."""
        # Look for description paragraph after IC name
        pattern = re.compile(f"{re.escape(ic_name)}.*?\n\n([^*]+?)(?=\n\n|\*\*|$)", re.DOTALL | re.IGNORECASE)
        match = pattern.search(text)

        if match:
            return match.group(1).strip()

        return ""

    def validate_weapon_capability(self, weapon_name: str, requested_mode: str, weapons_registry: List[WeaponStats]) -> Dict[str, Any]:
        """Validate if a weapon supports a requested firing mode."""
        for weapon in weapons_registry:
            if weapon_name.lower() in weapon.name.lower():
                available_modes = [mode.strip() for mode in weapon.mode.split('/')]

                # Map common aliases
                mode_aliases = {
                    "burst": ["BF", "SA/BF", "BURST"],
                    "burst fire": ["BF", "SA/BF", "BURST"],
                    "semi-auto": ["SA", "SEMI"],
                    "full auto": ["FA", "FULL"],
                    "single shot": ["SS", "SINGLE"]
                }

                requested_aliases = mode_aliases.get(requested_mode.lower(), [requested_mode.upper()])

                for alias in requested_aliases:
                    if any(alias in mode for mode in available_modes):
                        return {"valid": True, "weapon": weapon}

                return {
                    "valid": False,
                    "error": f"{weapon.name} cannot fire in {requested_mode} mode",
                    "available_modes": available_modes,
                    "weapon": weapon
                }

        return {"valid": False, "error": f"Weapon '{weapon_name}' not found in registry"}


# Factory function for integration
def create_entity_registry_builder() -> EntityRegistryBuilder:
    """Create enhanced entity registry builder instance."""
    return EntityRegistryBuilder()


# Test function for validation
def test_enhanced_extraction():
    """Test the enhanced extraction on sample content."""
    builder = create_entity_registry_builder()

    # Test spell extraction
    spell_text = """
FIREBALL (INDIRECT, ELEMENTAL)

**Type:** P ; **Range:** LOS (A) ; **Damage:** P ; **Duration:** I ; **Drain:** F – 1

These spells create an explosion of flames that flash into existence and scorch the target(s).

MANABOLT (DIRECT)

**Type:** M ; **Range:** LOS ; **Damage:** P ; **Duration:** I ; **Drain:** F – 3

Manabolt channels destructive magical power into the target, doing Physical damage.
    """

    test_chunk = {"text": spell_text, "source": "test"}
    entities = builder.extract_entities_from_chunk(test_chunk)

    print("Enhanced Extraction Test:")
    print(f"Extracted {len(entities['spells'])} spells:")
    for spell in entities["spells"]:
        print(f"  {spell.name}: {spell.spell_type} type, {spell.drain} drain")

    print(f"Extracted {len(entities['weapons'])} weapons")
    print(f"Extracted {len(entities['ic_programs'])} IC programs")


if __name__ == "__main__":
    test_enhanced_extraction()