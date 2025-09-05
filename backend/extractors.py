"""
Rulebook content extractors for populating character database reference tables.
Scans processed markdown files to extract skills, qualities, and gear.
"""

import re
import json
from pathlib import Path
from typing import Dict, List
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExtractedItem:
    """Base class for extracted items."""
    name: str
    description: str = ""
    source_page: str = ""


@dataclass
class ExtractedSkill(ExtractedItem):
    """Extracted skill data."""
    skill_type: str = "active"  # active, knowledge, language
    skill_group: str = ""
    linked_attribute: str = ""
    default_allowed: bool = True


@dataclass
class ExtractedQuality(ExtractedItem):
    """Extracted quality data."""
    quality_type: str = "positive"  # positive, negative
    karma_cost: int = 0
    rating_range: str = ""


@dataclass
class ExtractedGear(ExtractedItem):
    """Extracted gear data."""
    category: str = ""
    subcategory: str = ""
    base_cost: int = 0
    availability: str = ""
    armor_value: int = 0
    rating_range: str = ""
    properties: Dict = None


class RulebookExtractor:
    """Extract game content from processed Shadowrun rulebook markdown files."""

    def __init__(self, processed_dir: str = "data/processed_markdown"):
        self.processed_dir = Path(processed_dir)

        # Gear categories based on your structure
        self.gear_categories = {
            "weapons": {"melee": [], "ranged": []},
            "ammunition": [],
            "clothing_armor": [],
            "cyberware_bioware": [],
            "electronics": {"commlinks": [], "cyberdecks": [], "programs": []},
            "identity": {"fake_sins": [], "licenses": []},
            "tools": {"general": [], "breaking_entering": [], "surveillance": []},
            "lifestyle": [],
            "vehicles": [],
            "drones": [],
            "biotech": [],
            "disguises": [],
            "sensors": [],
            "magical_goods": {"foci": [], "ritual_materials": [], "preparations": []},
            "financial": {"credsticks": [], "contracts": []}
        }

        # Common skill groups and attributes
        self.skill_groups = {
            "acting": ["con", "impersonation", "performance"],
            "athletics": ["gymnastics", "running", "swimming"],
            "biotech": ["cybertechnology", "first_aid", "medicine"],
            "close_combat": ["blades", "clubs", "unarmed_combat"],
            "conjuring": ["banishing", "binding", "summoning"],
            "cracking": ["cybercombat", "electronic_warfare", "hacking"],
            "electronics": ["computer", "hardware", "software"],
            "enchanting": ["alchemy", "artificing", "disenchanting"],
            "engineering": ["aeronautics_mechanics", "automotive_mechanics", "industrial_mechanics"],
            "firearms": ["archery", "automatics", "longarms", "pistols"],
            "influence": ["etiquette", "leadership", "negotiation"],
            "outdoors": ["navigation", "survival", "tracking"],
            "perception": ["assensing", "perception"],
            "sorcery": ["counterspelling", "ritual_spellcasting", "spellcasting"],
            "stealth": ["disguise", "infiltration", "palming"],
            "tasking": ["compiling", "decompiling", "registering"]
        }

        self.attribute_mappings = {
            "body": ["bod", "body"],
            "agility": ["agi", "agility"],
            "reaction": ["rea", "reaction"],
            "strength": ["str", "strength"],
            "willpower": ["wil", "willpower"],
            "logic": ["log", "logic"],
            "intuition": ["int", "intuition"],
            "charisma": ["cha", "charisma"],
            "edge": ["edg", "edge"],
            "magic": ["mag", "magic"],
            "resonance": ["res", "resonance"]
        }

    def find_rulebook_files(self) -> List[Path]:
        """Find all processed markdown files that likely contain rulebook content."""
        if not self.processed_dir.exists():
            logger.warning(f"Processed directory not found: {self.processed_dir}")
            return []

        rulebook_files = []

        # Look for core rulebook files
        for md_file in self.processed_dir.rglob("*.md"):
            content = md_file.read_text(encoding='utf-8', errors='ignore')

            # Skip if it's clearly not a rulebook (too short or wrong content)
            if len(content) < 1000:
                continue

            # Look for rulebook indicators in YAML frontmatter or content
            content_lower = content.lower()

            rulebook_indicators = [
                "document_type: \"rulebook\"",
                "core rulebook", "shadowrun 5", "shadowrun 6",
                "skill", "quality", "gear", "weapon", "armor",
                "chapter", "dice pool", "attribute"
            ]

            if any(indicator in content_lower for indicator in rulebook_indicators):
                rulebook_files.append(md_file)
                logger.info(f"Found rulebook file: {md_file.name}")

        return rulebook_files

    def extract_skills_from_content(self, content: str, source_file: str) -> List[ExtractedSkill]:
        """Extract skills from markdown content."""
        skills = []

        # Look for skill sections and tables
        skill_patterns = [
            # Table format: | Skill Name | Linked Attribute | Group |
            r'\|\s*([A-Z][a-zA-Z\s&\(\)]+?)\s*\|\s*([A-Z][a-z]{2,3})\s*\|\s*([A-Z][a-zA-Z\s]*)\s*\|',
            # List format: - Skill Name (Attribute)
            r'^[\s\-\*]\s*([A-Z][a-zA-Z\s&\(\)]+?)\s*\(([A-Z][a-z]{2,3})\)',
            # Header format: ## Skill Name
            r'^#+\s*([A-Z][a-zA-Z\s&\(\)]+?)(?:\s*\(([A-Z][a-z]{2,3})\))?',
        ]

        lines = content.split('\n')
        current_section = ""

        for i, line in enumerate(lines):
            line = line.strip()

            # Track current section for context
            if line.startswith('#'):
                current_section = line.lower()

            # Determine skill type from section context
            skill_type = "active"
            if any(kw in current_section for kw in ["knowledge", "academic", "street", "professional"]):
                skill_type = "knowledge"
            elif any(kw in current_section for kw in ["language"]):
                skill_type = "language"

            # Try each pattern
            for pattern in skill_patterns:
                matches = re.finditer(pattern, line, re.MULTILINE)

                for match in matches:
                    skill_name = match.group(1).strip()
                    attribute = match.group(2).strip() if len(match.groups()) > 1 and match.group(2) else ""
                    skill_group = match.group(3).strip() if len(match.groups()) > 2 and match.group(3) else ""

                    # Clean up skill name
                    skill_name = self._clean_skill_name(skill_name)

                    # Skip if too short or looks invalid
                    if len(skill_name) < 3 or skill_name.lower() in ["skill", "name", "attribute"]:
                        continue

                    # Map attribute abbreviations to full names
                    mapped_attr = self._map_attribute(attribute)

                    # Determine skill group
                    detected_group = self._detect_skill_group(skill_name, skill_group)

                    skills.append(ExtractedSkill(
                        name=skill_name,
                        skill_type=skill_type,
                        linked_attribute=mapped_attr,
                        skill_group=detected_group,
                        source_page=self._extract_page_reference(line, lines, i),
                        description=self._extract_skill_description(skill_name, lines, i)
                    ))

        # Remove duplicates
        unique_skills = {}
        for skill in skills:
            if skill.name not in unique_skills:
                unique_skills[skill.name] = skill

        logger.info(f"Extracted {len(unique_skills)} skills from {source_file}")
        return list(unique_skills.values())

    def extract_qualities_from_content(self, content: str, source_file: str) -> List[ExtractedQuality]:
        """Extract qualities from markdown content."""
        qualities = []

        # Look for quality sections
        quality_patterns = [
            # Header format: ## Quality Name
            r'^#+\s*([A-Z][a-zA-Z\s\(\)]+?)(?:\s*[\(\[]([0-9\-]+)[\)\]])?',
            # Table format: | Quality Name | Cost | Type |
            r'\|\s*([A-Z][a-zA-Z\s\(\)]+?)\s*\|\s*([0-9\-]+)?\s*\|\s*(Positive|Negative|pos|neg)?\s*\|',
            # List format with cost: - Quality Name [15 karma]
            r'^[\s\-\*]\s*([A-Z][a-zA-Z\s\(\)]+?)\s*[\[\(]([0-9\-]+)[\]\)]',
        ]

        lines = content.split('\n')
        current_section = ""

        for i, line in enumerate(lines):
            line = line.strip()

            # Track section for quality type context
            if line.startswith('#'):
                current_section = line.lower()

            # Determine quality type from section
            quality_type = "positive"
            if any(kw in current_section for kw in ["negative", "flaw", "disadvantage"]):
                quality_type = "negative"

            for pattern in quality_patterns:
                matches = re.finditer(pattern, line, re.MULTILINE)

                for match in matches:
                    quality_name = match.group(1).strip()
                    cost_str = match.group(2).strip() if len(match.groups()) > 1 and match.group(2) else "0"
                    type_str = match.group(3).strip() if len(match.groups()) > 2 and match.group(3) else ""

                    # Clean quality name
                    quality_name = self._clean_quality_name(quality_name)

                    # Skip invalid names
                    if len(quality_name) < 3 or quality_name.lower() in ["quality", "name", "cost", "type"]:
                        continue

                    # Parse karma cost
                    karma_cost = self._parse_karma_cost(cost_str)

                    # Override quality type if specified
                    if type_str.lower().startswith('neg'):
                        quality_type = "negative"
                    elif type_str.lower().startswith('pos'):
                        quality_type = "positive"

                    qualities.append(ExtractedQuality(
                        name=quality_name,
                        quality_type=quality_type,
                        karma_cost=karma_cost,
                        rating_range=self._extract_quality_rating_range(quality_name, lines, i),
                        source_page=self._extract_page_reference(line, lines, i),
                        description=self._extract_quality_description(quality_name, lines, i)
                    ))

        # Remove duplicates
        unique_qualities = {}
        for quality in qualities:
            if quality.name not in unique_qualities:
                unique_qualities[quality.name] = quality

        logger.info(f"Extracted {len(unique_qualities)} qualities from {source_file}")
        return list(unique_qualities.values())

    def extract_gear_from_content(self, content: str, source_file: str) -> List[ExtractedGear]:
        """Extract gear items from markdown content."""
        gear_items = []

        # Gear table patterns
        gear_patterns = [
            # Table format: | Item | Cost | Avail | Notes |
            r'\|\s*([A-Z][a-zA-Z\s\(\)\-,&]+?)\s*\|\s*([0-9,¥]+)?\s*\|\s*([0-9]+[RF]?)?\s*\|\s*([^\|]*?)\s*\|',
            # List format: - Item Name (Cost, Availability)
            r'^[\s\-\*]\s*([A-Z][a-zA-Z\s\(\)\-,&]+?)\s*[\(\[]([0-9,¥]+)?\s*,?\s*([0-9]+[RF]?)?\s*[\)\]]',
        ]

        lines = content.split('\n')
        current_category = ""
        current_subcategory = ""

        for i, line in enumerate(lines):
            line = line.strip()

            # Track sections for categorization
            if line.startswith('#'):
                section_header = line.lower()
                current_category = self._detect_gear_category(section_header)
                current_subcategory = self._detect_gear_subcategory(section_header)

            for pattern in gear_patterns:
                matches = re.finditer(pattern, line, re.MULTILINE)

                for match in matches:
                    item_name = match.group(1).strip()
                    cost_str = match.group(2).strip() if len(match.groups()) > 1 and match.group(2) else "0"
                    avail_str = match.group(3).strip() if len(match.groups()) > 2 and match.group(3) else ""
                    notes = match.group(4).strip() if len(match.groups()) > 3 and match.group(4) else ""

                    # Clean item name
                    item_name = self._clean_gear_name(item_name)

                    # Skip invalid names
                    if len(item_name) < 3 or item_name.lower() in ["item", "name", "cost", "gear", "equipment"]:
                        continue

                    # Parse cost
                    cost = self._parse_cost(cost_str)

                    # Extract armor value if present
                    armor_value = self._extract_armor_value(item_name, notes)

                    # Extract rating range
                    rating_range = self._extract_rating_range(item_name, notes)

                    gear_items.append(ExtractedGear(
                        name=item_name,
                        category=current_category,
                        subcategory=current_subcategory,
                        base_cost=cost,
                        availability=avail_str,
                        armor_value=armor_value,
                        rating_range=rating_range,
                        source_page=self._extract_page_reference(line, lines, i),
                        description=notes,
                        properties=self._extract_gear_properties(notes)
                    ))

        # Remove duplicates
        unique_gear = {}
        for gear in gear_items:
            if gear.name not in unique_gear:
                unique_gear[gear.name] = gear

        logger.info(f"Extracted {len(unique_gear)} gear items from {source_file}")
        return list(unique_gear.values())

    def _clean_skill_name(self, name: str) -> str:
        """Clean and standardize skill names."""
        # Remove common prefixes/suffixes
        name = re.sub(r'\s*\(.*?\)\s*', '', name)  # Remove parenthetical notes
        name = re.sub(r'\s*\[.*?\]\s*', '', name)  # Remove bracketed notes
        name = name.replace('*', '').strip()

        # Standardize common variations
        name_map = {
            "con": "Con",
            "etiquette": "Etiquette",
            "firearms": "Firearms",
            "hacking": "Hacking",
            "perception": "Perception",
            "pistols": "Pistols",
            "unarmed combat": "Unarmed Combat",
            "cybercombat": "Cybercombat"
        }

        return name_map.get(name.lower(), name.title())

    def _map_attribute(self, attr_abbrev: str) -> str:
        """Map attribute abbreviations to full names."""
        if not attr_abbrev:
            return ""

        for full_name, abbrevs in self.attribute_mappings.items():
            if attr_abbrev.lower() in abbrevs:
                return full_name

        return attr_abbrev.lower()

    def _detect_skill_group(self, skill_name: str, declared_group: str = "") -> str:
        """Detect skill group based on skill name and context."""
        if declared_group and declared_group.lower() != "none":
            return declared_group.lower()

        skill_lower = skill_name.lower()

        for group_name, skills in self.skill_groups.items():
            if any(skill in skill_lower for skill in skills):
                return group_name

        return ""

    def _clean_quality_name(self, name: str) -> str:
        """Clean and standardize quality names."""
        name = re.sub(r'\s*\(.*?\)\s*', '', name)
        name = re.sub(r'\s*\[.*?\]\s*', '', name)
        return name.strip()

    def _parse_karma_cost(self, cost_str: str) -> int:
        """Parse karma cost from string."""
        if not cost_str:
            return 0

        # Extract numbers from cost string
        numbers = re.findall(r'[0-9]+', cost_str)
        return int(numbers[0]) if numbers else 0

    def _clean_gear_name(self, name: str) -> str:
        """Clean and standardize gear names."""
        name = re.sub(r'\s*\(Rating [0-9\-]+\)\s*', '', name)
        name = name.strip()
        return name

    def _parse_cost(self, cost_str: str) -> int:
        """Parse cost from string (remove ¥ symbol and commas)."""
        if not cost_str:
            return 0

        # Remove currency symbols and commas
        clean_cost = re.sub(r'[¥,]', '', cost_str)
        numbers = re.findall(r'[0-9]+', clean_cost)
        return int(numbers[0]) if numbers else 0

    def _detect_gear_category(self, section_header: str) -> str:
        """Detect gear category from section header."""
        category_keywords = {
            "weapons": ["weapon", "melee", "firearm", "gun", "blade", "club"],
            "clothing_armor": ["armor", "clothing", "protection"],
            "electronics": ["electronics", "commlink", "cyberdeck", "program"],
            "cyberware_bioware": ["cyberware", "bioware", "augment", "implant"],
            "vehicles": ["vehicle", "car", "bike", "truck"],
            "drones": ["drone", "pilot", "sensor"],
            "tools": ["tool", "kit", "equipment"]
        }

        for category, keywords in category_keywords.items():
            if any(keyword in section_header for keyword in keywords):
                return category

        return "general"

    def _detect_gear_subcategory(self, section_header: str) -> str:
        """Detect gear subcategory from section header."""
        if "melee" in section_header:
            return "melee"
        elif any(kw in section_header for kw in ["ranged", "firearm", "gun", "pistol", "rifle"]):
            return "ranged"
        elif "commlink" in section_header:
            return "commlinks"
        elif "cyberdeck" in section_header:
            return "cyberdecks"
        elif "program" in section_header:
            return "programs"

        return ""

    def _extract_armor_value(self, item_name: str, description: str) -> int:
        """Extract armor value from item name or description."""
        text = f"{item_name} {description}".lower()

        # Look for armor patterns
        armor_patterns = [
            r'armor\s*([0-9]+)',
            r'\+([0-9]+)\s*armor',
            r'protection\s*([0-9]+)'
        ]

        for pattern in armor_patterns:
            match = re.search(pattern, text)
            if match:
                return int(match.group(1))

        return 0

    def _extract_rating_range(self, item_name: str, description: str) -> str:
        """Extract rating range from item name or description."""
        text = f"{item_name} {description}".lower()

        # Look for rating patterns
        rating_patterns = [
            r'rating\s*([0-9]+[\-–][0-9]+)',
            r'rating\s*([0-9]+)',
            r'\(rating\s*([0-9]+[\-–][0-9]+)\)',
            r'levels?\s*([0-9]+[\-–][0-9]+)'
        ]

        for pattern in rating_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)

        return ""

    def _extract_gear_properties(self, description: str) -> Dict:
        """Extract additional properties from gear description."""
        properties = {}

        if "wireless" in description.lower():
            properties["wireless"] = True

        if "smartlink" in description.lower():
            properties["smartlink"] = True

        # Add more property extractions as needed

        return properties

    def _extract_page_reference(self, line: str, lines: List[str], line_index: int) -> str:
        """Extract page reference from line or nearby context."""
        # Look for page patterns in current line and nearby lines
        page_patterns = [r'p\.\s*([0-9]+)', r'page\s*([0-9]+)', r'SR[56]?\s*([0-9]+)']

        search_lines = [line]
        if line_index > 0:
            search_lines.append(lines[line_index - 1])
        if line_index < len(lines) - 1:
            search_lines.append(lines[line_index + 1])

        for search_line in search_lines:
            for pattern in page_patterns:
                match = re.search(pattern, search_line)
                if match:
                    return f"p. {match.group(1)}"

        return ""

    def _extract_skill_description(self, skill_name: str, lines: List[str], line_index: int) -> str:
        """Extract skill description from nearby lines."""
        # Look for descriptive text in the next few lines
        description_lines = []

        for i in range(line_index + 1, min(line_index + 4, len(lines))):
            line = lines[i].strip()

            # Stop at next header or table row
            if line.startswith('#') or line.startswith('|') or line.startswith('-'):
                break

            if line and not line.startswith('*') and len(line) > 10:
                description_lines.append(line)

        return " ".join(description_lines)[:200]  # Limit length

    def _extract_quality_description(self, quality_name: str, lines: List[str], line_index: int) -> str:
        """Extract quality description from nearby lines."""
        return self._extract_skill_description(quality_name, lines, line_index)

    def _extract_quality_rating_range(self, quality_name: str, lines: List[str], line_index: int) -> str:
        """Extract quality rating range."""
        # Look for rating information in nearby lines
        for i in range(line_index, min(line_index + 3, len(lines))):
            line = lines[i].lower()

            rating_match = re.search(r'rating\s*([0-9]+[\-–][0-9]+|[0-9]+)', line)
            if rating_match:
                return rating_match.group(1)

        return ""


def populate_reference_tables():
    """Main function to extract and populate all reference tables."""
    from .characters import get_character_db

    db = get_character_db()
    extractor = RulebookExtractor()

    rulebook_files = extractor.find_rulebook_files()

    if not rulebook_files:
        logger.warning("No rulebook files found for extraction")
        return

    logger.info(f"Processing {len(rulebook_files)} rulebook files...")

    all_skills = []
    all_qualities = []
    all_gear = []

    # Extract from each file
    for file_path in rulebook_files:
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')

            skills = extractor.extract_skills_from_content(content, file_path.name)
            qualities = extractor.extract_qualities_from_content(content, file_path.name)
            gear = extractor.extract_gear_from_content(content, file_path.name)

            all_skills.extend(skills)
            all_qualities.extend(qualities)
            all_gear.extend(gear)

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue

    # Insert into database
    with db.get_connection() as conn:
        cursor = conn.cursor()

        # Clear existing reference data
        cursor.execute("DELETE FROM skills_library")
        cursor.execute("DELETE FROM qualities_library")
        cursor.execute("DELETE FROM gear_library")

        # Insert skills
        for skill in all_skills:
            cursor.execute("""
                INSERT OR IGNORE INTO skills_library 
                (name, skill_type, skill_group, linked_attribute, default_allowed, description, source_page)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                skill.name, skill.skill_type, skill.skill_group,
                skill.linked_attribute, skill.default_allowed,
                skill.description, skill.source_page
            ))

        # Insert qualities
        for quality in all_qualities:
            cursor.execute("""
                INSERT OR IGNORE INTO qualities_library
                (name, quality_type, karma_cost, rating_range, description, source_page)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                quality.name, quality.quality_type, quality.karma_cost,
                quality.rating_range, quality.description, quality.source_page
            ))

        # Insert gear
        for gear in all_gear:
            cursor.execute("""
                INSERT OR IGNORE INTO gear_library
                (name, category, subcategory, base_cost, availability, armor_value, 
                 rating_range, description, source_page, properties)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                gear.name, gear.category, gear.subcategory, gear.base_cost,
                gear.availability, gear.armor_value, gear.rating_range,
                gear.description, gear.source_page,
                json.dumps(gear.properties) if gear.properties else None
            ))

        conn.commit()

    logger.info(f"Successfully populated reference tables:")
    logger.info(f"  - {len(all_skills)} skills")
    logger.info(f"  - {len(all_qualities)} qualities")
    logger.info(f"  - {len(all_gear)} gear items")


if __name__ == "__main__":
    populate_reference_tables()