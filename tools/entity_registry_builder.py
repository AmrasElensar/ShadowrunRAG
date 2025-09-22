"""
Entity Registry Builder for Shadowrun RAG System
Extracts weapons, spells, and IC entities during indexing without breaking existing functionality.
"""

import re
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

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
    """Builds entity registries from Shadowrun content during indexing."""

    def __init__(self):
        # Weapon table pattern - matches the exact table format
        self.weapon_table_pattern = re.compile(
            r'\|\s*([^|]+?)\s*\|\s*(\d+(?:\s*\(\d+\))?)\s*\|\s*(\d+[PS](?:\s*\([^)]+\))?)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|',
            re.MULTILINE
        )

        # Spell pattern - matches the standardized spell format
        self.spell_pattern = re.compile(
            r'([A-Z\s]+?)\s*\(([^)]+)\)\s*\*\*Type:\*\*\s*([MP])\s*;\s*\*\*Range:\*\*\s*([^;]+?)\s*;\s*\*\*(?:Damage:\*\*\s*([^;]+?)\s*;\s*)?\*\*Duration:\*\*\s*([^;]+?)\s*;\s*\*\*Drain:\*\*\s*([^\n]+)',
            re.MULTILINE | re.IGNORECASE
        )

        # IC pattern - matches IC attack descriptions
        self.ic_pattern = re.compile(
            r'([A-Z\s]+?)\s*\*\*Attack:\*\*\s*([^v]+?)\s*v\.\s*([^\n]+)',
            re.MULTILINE
        )

        # Weapon description pattern
        self.weapon_desc_pattern = re.compile(
            r'\*\*([^:]+?):\*\*\s*([^*]+?)(?=\*\*|$)',
            re.MULTILINE | re.DOTALL
        )

        # Known weapon categories for context
        self.weapon_categories = [
            "LIGHT PISTOLS", "HEAVY PISTOLS", "MACHINE PISTOLS",
            "SUBMACHINE GUNS", "ASSAULT RIFLES", "SNIPER RIFLES",
            "SHOTGUNS", "SPECIAL WEAPONS"
        ]

        # Known spell categories
        self.spell_categories = [
            "Combat", "Detection", "Health", "Illusion", "Manipulation"
        ]

        logger.info("Entity Registry Builder initialized")

    def extract_entities_from_chunk(self, chunk: Dict[str, Any]) -> Dict[str, List]:
        """Extract all entities from a single chunk."""
        text = chunk.get("text", "")
        source = chunk.get("source", "")

        entities = {
            "weapons": [],
            "spells": [],
            "ic_programs": []
        }

        # Extract weapons if this looks like weapon content
        if self._is_weapon_content(text):
            entities["weapons"] = self._extract_weapons(text, source)

        # Extract spells if this looks like magic content
        if self._is_spell_content(text):
            entities["spells"] = self._extract_spells(text, source)

        # Extract IC if this looks like Matrix content
        if self._is_ic_content(text):
            entities["ic_programs"] = self._extract_ic_programs(text, source)

        return entities

    def _is_weapon_content(self, text: str) -> bool:
        """Check if text contains weapon statistics."""
        # Check for weapon table headers
        if re.search(r'\|\s*(?:WEAPON|FIREARM|ACC|Damage|AP|Mode)\s*\|', text, re.IGNORECASE):
            return True

        # Check for weapon categories
        for category in self.weapon_categories:
            if category in text.upper():
                return True

        # Check for weapon manufacturers
        manufacturers = ["ARES", "COLT", "RUGER", "BROWNING", "REMINGTON"]
        for mfg in manufacturers:
            if mfg in text.upper():
                return True

        return False

    def _is_spell_content(self, text: str) -> bool:
        """Check if text contains spell definitions."""
        # Look for spell format patterns
        if re.search(r'\*\*Type:\*\*\s*[MP]\s*;\s*\*\*Range:\*\*', text):
            return True

        # Look for spell keywords
        spell_keywords = ["DRAIN", "FORCE", "MANA", "PHYSICAL", "ASTRAL"]
        count = sum(1 for keyword in spell_keywords if keyword in text.upper())
        return count >= 2

    def _is_ic_content(self, text: str) -> bool:
        """Check if text contains IC program definitions."""
        # Look for IC attack patterns
        if re.search(r'\*\*Attack:\*\*.*?v\.', text):
            return True

        # Look for known IC types
        ic_types = ["BLACK IC", "BLASTER", "ACID", "BINDER", "KILLER", "PATROL"]
        for ic_type in ic_types:
            if ic_type in text.upper():
                return True

        return False

    def _extract_weapons(self, text: str, source: str) -> List[WeaponStats]:
        """Extract weapon statistics from text."""
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

    def _extract_spells(self, text: str, source: str) -> List[SpellStats]:
        """Extract spell statistics from text."""
        spells = []

        # Determine spell category from context
        current_category = "Unknown"
        for category in self.spell_categories:
            if category.lower() in text.lower():
                current_category = category
                break

        matches = self.spell_pattern.findall(text)
        for match in matches:
            if len(match) >= 6:
                spell_name = match[0].strip()
                keywords = [k.strip() for k in match[1].split(',')]
                spell_type = match[2].strip()
                range_val = match[3].strip()
                damage = match[4].strip() if match[4] else ""
                duration = match[5].strip()
                drain = match[6].strip()

                spell = SpellStats(
                    name=spell_name,
                    spell_type=spell_type,
                    range=range_val,
                    damage=damage,
                    duration=duration,
                    drain=drain,
                    keywords=keywords,
                    category=current_category
                )

                spells.append(spell)

        return spells

    def _extract_ic_programs(self, text: str, source: str) -> List[ICStats]:
        """Extract IC program statistics from text."""
        ic_programs = []

        matches = self.ic_pattern.findall(text)
        for match in matches:
            if len(match) >= 3:
                ic_name = match[0].strip()
                attack_pattern = match[1].strip()
                resistance_test = match[2].strip()

                # Extract effect description from following text
                effect_desc = self._extract_ic_description(text, ic_name)

                ic = ICStats(
                    name=ic_name,
                    attack_pattern=attack_pattern,
                    resistance_test=resistance_test,
                    effect_description=effect_desc
                )

                ic_programs.append(ic)

        return ic_programs

    def _extract_manufacturer(self, weapon_name: str) -> str:
        """Extract manufacturer from weapon name."""
        manufacturers = {
            "ares": "Ares",
            "colt": "Colt",
            "ruger": "Ruger",
            "browning": "Browning",
            "remington": "Remington",
            "defiance": "Defiance",
            "fichetti": "Fichetti",
            "beretta": "Beretta",
            "yamaha": "Yamaha"
        }

        weapon_lower = weapon_name.lower()
        for key, name in manufacturers.items():
            if key in weapon_lower:
                return name

        return "Unknown"

    def _extract_ic_description(self, text: str, ic_name: str) -> str:
        """Extract IC effect description from surrounding text."""
        # Find the IC name in text and extract the following paragraph
        pattern = re.compile(f"{re.escape(ic_name)}.*?\n\n([^*]+?)(?=\n\n|\*\*|$)", re.DOTALL | re.IGNORECASE)
        match = pattern.search(text)

        if match:
            return match.group(1).strip()

        return ""

    def validate_weapon_capability(self, weapon_name: str, requested_mode: str, weapons_registry: List[WeaponStats]) -> \
    Dict[str, Any]:
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
    """Create entity registry builder instance."""
    return EntityRegistryBuilder()


# Test function for validation
def test_registry_extraction():
    """Test the registry extraction on sample content."""
    builder = create_entity_registry_builder()

    # Test weapon extraction
    weapon_text = """
| Weapon | ACC | Damage | AP | Mode | RC | Ammo | Avail | Cost |
| Ares Predator V | 5 (7) | 8P | –1 | SA | — | 15 (c) | 5R | 725¥ |
| Ares Viper Slivergun | 4 | 9P (f) | +4 | SA/BF | — | 30 (c) | 8F | 380¥ |

**Ares Predator V:** The newest iteration of the most popular handgun in the world.
    """

    test_chunk = {"text": weapon_text, "source": "test"}
    entities = builder.extract_entities_from_chunk(test_chunk)

    print("Extracted Weapons:")
    for weapon in entities["weapons"]:
        print(f"  {weapon.name}: {weapon.mode} mode, {weapon.manufacturer}")

    # Test validation
    validation = builder.validate_weapon_capability("Ares Predator", "burst fire", entities["weapons"])
    print(f"Burst fire validation: {validation}")


if __name__ == "__main__":
    test_registry_extraction()