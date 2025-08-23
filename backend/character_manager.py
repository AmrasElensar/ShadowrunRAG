"""
Character CRUD operations and business logic for Shadowrun RAG character system.
Handles all character data operations, query context generation, and export functionality.
"""

import sqlite3
from typing import Dict, List, Optional, Any, Tuple
import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class CharacterManager:
    """High-level character management operations."""

    def __init__(self, db):
        self.db = db

    # ===== CHARACTER MANAGEMENT =====

    def create_character(self, name: str, metatype: str = "Human", archetype: str = "") -> int:
        """Create a new character with default values."""
        return self.db.create_character(name, metatype, archetype)

    def get_character_list(self) -> List[Dict[str, Any]]:
        """Get list of all characters for dropdown."""
        return self.db.list_characters()

    def delete_character(self, character_id: int) -> bool:
        """Delete a character and all associated data."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM characters WHERE id = ?", (character_id,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error deleting character: {e}")
            return False

    def get_character_full_data(self, character_id: int) -> Optional[Dict[str, Any]]:
        """Get complete character data for editing."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            # Get basic character info
            cursor.execute("SELECT * FROM characters WHERE id = ?", (character_id,))
            char_row = cursor.fetchone()
            if not char_row:
                return None

            # Build character data structure
            char_data = {
                "id": char_row[0],
                "name": char_row[1],
                "metatype": char_row[2],
                "archetype": char_row[3],
                "created_at": char_row[4],
                "updated_at": char_row[5],
                "notes": char_row[6]
            }

            # Get stats
            cursor.execute("SELECT * FROM character_stats WHERE character_id = ?", (character_id,))
            stats_row = cursor.fetchone()
            if stats_row:
                char_data["stats"] = {
                    "body": stats_row[1], "agility": stats_row[2], "reaction": stats_row[3],
                    "strength": stats_row[4], "charisma": stats_row[5], "logic": stats_row[6],
                    "intuition": stats_row[7], "willpower": stats_row[8], "edge": stats_row[9],
                    "essence": stats_row[10], "physical_limit": stats_row[11],
                    "mental_limit": stats_row[12], "social_limit": stats_row[13],
                    "initiative": stats_row[14], "hot_sim_vr": stats_row[15]
                }

            # Get resources
            cursor.execute("SELECT * FROM character_resources WHERE character_id = ?", (character_id,))
            resources_row = cursor.fetchone()
            if resources_row:
                char_data["resources"] = {
                    "nuyen": resources_row[1], "street_cred": resources_row[2],
                    "notoriety": resources_row[3], "public_aware": resources_row[4],
                    "total_karma": resources_row[5], "available_karma": resources_row[6],
                    "edge_pool": resources_row[7]
                }

            # Get qualities
            cursor.execute("""
                SELECT name, rating, karma_cost, description, quality_type
                FROM character_qualities WHERE character_id = ? ORDER BY name
            """, (character_id,))
            char_data["qualities"] = [
                {"name": row[0], "rating": row[1], "karma_cost": row[2],
                 "description": row[3], "quality_type": row[4]}
                for row in cursor.fetchall()
            ]

            # Get skills
            cursor.execute("""
                SELECT name, rating, specialization, skill_type, skill_group, attribute
                FROM character_skills WHERE character_id = ? ORDER BY skill_type, name
            """, (character_id,))
            skills_by_type = {"active": [], "knowledge": [], "language": []}
            for row in cursor.fetchall():
                skill_data = {
                    "name": row[0], "rating": row[1], "specialization": row[2],
                    "skill_type": row[3], "skill_group": row[4], "attribute": row[5]
                }
                skills_by_type[row[3]].append(skill_data)
            char_data["skills"] = skills_by_type

            # Get gear
            cursor.execute("""
                SELECT name, category, subcategory, quantity, rating, armor_value, 
                       cost, availability, description, custom_properties
                FROM character_gear WHERE character_id = ? ORDER BY category, name
            """, (character_id,))
            char_data["gear"] = [
                {
                    "name": row[0], "category": row[1], "subcategory": row[2],
                    "quantity": row[3], "rating": row[4], "armor_value": row[5],
                    "cost": row[6], "availability": row[7], "description": row[8],
                    "custom_properties": json.loads(row[9]) if row[9] else {}
                }
                for row in cursor.fetchall()
            ]

            # Get weapons
            cursor.execute("""
                SELECT name, weapon_type, mode_ammo, accuracy, damage_code,
                       armor_penetration, recoil_compensation, cost, availability, description
                FROM character_weapons WHERE character_id = ? ORDER BY weapon_type, name
            """, (character_id,))
            char_data["weapons"] = [
                {
                    "name": row[0], "weapon_type": row[1], "mode_ammo": row[2],
                    "accuracy": row[3], "damage_code": row[4], "armor_penetration": row[5],
                    "recoil_compensation": row[6], "cost": row[7], "availability": row[8],
                    "description": row[9]
                }
                for row in cursor.fetchall()
            ]

            # Get vehicles
            cursor.execute("""
                SELECT name, vehicle_type, handling, speed, acceleration, body,
                       armor, pilot, sensor, seats, cost, availability, description
                FROM character_vehicles WHERE character_id = ? ORDER BY vehicle_type, name
            """, (character_id,))
            char_data["vehicles"] = [
                {
                    "name": row[0], "vehicle_type": row[1], "handling": row[2],
                    "speed": row[3], "acceleration": row[4], "body": row[5],
                    "armor": row[6], "pilot": row[7], "sensor": row[8], "seats": row[9],
                    "cost": row[10], "availability": row[11], "description": row[12]
                }
                for row in cursor.fetchall()
            ]

            # Get cyberdeck
            cursor.execute("""
                SELECT name, device_rating, attack, sleaze, firewall, data_processing,
                       matrix_damage, cost, availability, description
                FROM character_cyberdeck WHERE character_id = ?
            """, (character_id,))
            cyberdeck_row = cursor.fetchone()
            char_data["cyberdeck"] = None
            if cyberdeck_row:
                char_data["cyberdeck"] = {
                    "name": cyberdeck_row[0], "device_rating": cyberdeck_row[1],
                    "attack": cyberdeck_row[2], "sleaze": cyberdeck_row[3],
                    "firewall": cyberdeck_row[4], "data_processing": cyberdeck_row[5],
                    "matrix_damage": cyberdeck_row[6], "cost": cyberdeck_row[7],
                    "availability": cyberdeck_row[8], "description": cyberdeck_row[9]
                }

            # Get programs
            cursor.execute("""
                SELECT name, rating, program_type, description
                FROM character_programs WHERE character_id = ? ORDER BY name
            """, (character_id,))
            char_data["programs"] = [
                {"name": row[0], "rating": row[1], "program_type": row[2], "description": row[3]}
                for row in cursor.fetchall()
            ]

            return char_data

    # ===== STATS & RESOURCES =====

    def update_character_stats(self, character_id: int, stats: Dict[str, Any]) -> bool:
        """Update character statistics."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    UPDATE character_stats SET
                        body = ?, agility = ?, reaction = ?, strength = ?,
                        charisma = ?, logic = ?, intuition = ?, willpower = ?,
                        edge = ?, essence = ?, physical_limit = ?, mental_limit = ?,
                        social_limit = ?, initiative = ?, hot_sim_vr = ?
                    WHERE character_id = ?
                """, (
                    stats.get("body", 1), stats.get("agility", 1),
                    stats.get("reaction", 1), stats.get("strength", 1),
                    stats.get("charisma", 1), stats.get("logic", 1),
                    stats.get("intuition", 1), stats.get("willpower", 1),
                    stats.get("edge", 1), stats.get("essence", 6.0),
                    stats.get("physical_limit", 1), stats.get("mental_limit", 1),
                    stats.get("social_limit", 1), stats.get("initiative", 1),
                    stats.get("hot_sim_vr", 0), character_id
                ))

                self._update_character_timestamp(cursor, character_id)
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error updating character stats: {e}")
            return False

    def update_character_resources(self, character_id: int, resources: Dict[str, Any]) -> bool:
        """Update character resources."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    UPDATE character_resources SET
                        nuyen = ?, street_cred = ?, notoriety = ?, public_aware = ?,
                        total_karma = ?, available_karma = ?, edge_pool = ?
                    WHERE character_id = ?
                """, (
                    resources.get("nuyen", 0), resources.get("street_cred", 0),
                    resources.get("notoriety", 0), resources.get("public_aware", 0),
                    resources.get("total_karma", 0), resources.get("available_karma", 0),
                    resources.get("edge_pool", 1), character_id
                ))

                self._update_character_timestamp(cursor, character_id)
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error updating character resources: {e}")
            return False

    # ===== SKILLS MANAGEMENT =====

    def add_character_skill(self, character_id: int, skill_data: Dict[str, Any]) -> bool:
        """Add a skill to a character."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT INTO character_skills 
                    (character_id, name, rating, specialization, skill_type, skill_group, attribute)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    character_id, skill_data["name"], skill_data.get("rating", 1),
                    skill_data.get("specialization", ""), skill_data.get("skill_type", "active"),
                    skill_data.get("skill_group", ""), skill_data.get("attribute", "")
                ))

                self._update_character_timestamp(cursor, character_id)
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error adding character skill: {e}")
            return False

    def update_character_skill(self, character_id: int, skill_name: str, skill_data: Dict[str, Any]) -> bool:
        """Update an existing character skill."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    UPDATE character_skills SET
                        rating = ?, specialization = ?, skill_type = ?, 
                        skill_group = ?, attribute = ?
                    WHERE character_id = ? AND name = ?
                """, (
                    skill_data.get("rating", 1), skill_data.get("specialization", ""),
                    skill_data.get("skill_type", "active"), skill_data.get("skill_group", ""),
                    skill_data.get("attribute", ""), character_id, skill_name
                ))

                self._update_character_timestamp(cursor, character_id)
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error updating character skill: {e}")
            return False

    def remove_character_skill(self, character_id: int, skill_name: str) -> bool:
        """Remove a skill from a character."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM character_skills WHERE character_id = ? AND name = ?",
                               (character_id, skill_name))
                self._update_character_timestamp(cursor, character_id)
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error removing character skill: {e}")
            return False

    # ===== QUALITIES MANAGEMENT =====

    def add_character_quality(self, character_id: int, quality_data: Dict[str, Any]) -> bool:
        """Add a quality to a character."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT INTO character_qualities 
                    (character_id, name, rating, karma_cost, description, quality_type)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    character_id, quality_data["name"], quality_data.get("rating", 0),
                    quality_data.get("karma_cost", 0), quality_data.get("description", ""),
                    quality_data.get("quality_type", "positive")
                ))

                self._update_character_timestamp(cursor, character_id)
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error adding character quality: {e}")
            return False

    def remove_character_quality(self, character_id: int, quality_name: str) -> bool:
        """Remove a quality from a character."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM character_qualities WHERE character_id = ? AND name = ?",
                               (character_id, quality_name))
                self._update_character_timestamp(cursor, character_id)
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error removing character quality: {e}")
            return False

    # ===== GEAR MANAGEMENT =====

    def add_character_gear(self, character_id: int, gear_data: Dict[str, Any]) -> bool:
        """Add gear to a character."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT INTO character_gear 
                    (character_id, name, category, subcategory, quantity, rating, 
                     armor_value, cost, availability, description, custom_properties)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    character_id, gear_data["name"], gear_data.get("category", ""),
                    gear_data.get("subcategory", ""), gear_data.get("quantity", 1),
                    gear_data.get("rating", 0), gear_data.get("armor_value", 0),
                    gear_data.get("cost", 0), gear_data.get("availability", ""),
                    gear_data.get("description", ""),
                    json.dumps(gear_data.get("custom_properties", {}))
                ))

                self._update_character_timestamp(cursor, character_id)
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error adding character gear: {e}")
            return False

    def remove_character_gear(self, character_id: int, gear_id: int) -> bool:
        """Remove gear from a character."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM character_gear WHERE character_id = ? AND id = ?",
                               (character_id, gear_id))
                self._update_character_timestamp(cursor, character_id)
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error removing character gear: {e}")
            return False

    # ===== WEAPONS MANAGEMENT =====

    def add_character_weapon(self, character_id: int, weapon_data: Dict[str, Any]) -> bool:
        """Add weapon to a character."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT INTO character_weapons 
                    (character_id, name, weapon_type, mode_ammo, accuracy, damage_code,
                     armor_penetration, recoil_compensation, cost, availability, description)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    character_id, weapon_data["name"], weapon_data.get("weapon_type", "ranged"),
                    weapon_data.get("mode_ammo", ""), weapon_data.get("accuracy", 0),
                    weapon_data.get("damage_code", ""), weapon_data.get("armor_penetration", 0),
                    weapon_data.get("recoil_compensation", 0), weapon_data.get("cost", 0),
                    weapon_data.get("availability", ""), weapon_data.get("description", "")
                ))

                self._update_character_timestamp(cursor, character_id)
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error adding character weapon: {e}")
            return False

    def remove_character_weapon(self, character_id: int, weapon_id: int) -> bool:
        """Remove weapon from a character."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM character_weapons WHERE character_id = ? AND id = ?",
                               (character_id, weapon_id))
                self._update_character_timestamp(cursor, character_id)
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error removing character weapon: {e}")
            return False

    # ===== CYBERDECK MANAGEMENT =====

    def update_character_cyberdeck(self, character_id: int, cyberdeck_data: Dict[str, Any]) -> bool:
        """Update character cyberdeck."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT OR REPLACE INTO character_cyberdeck 
                    (character_id, name, device_rating, attack, sleaze, firewall,
                     data_processing, matrix_damage, cost, availability, description)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    character_id, cyberdeck_data.get("name", ""),
                    cyberdeck_data.get("device_rating", 1),
                    cyberdeck_data.get("attack", 0), cyberdeck_data.get("sleaze", 0),
                    cyberdeck_data.get("firewall", 0), cyberdeck_data.get("data_processing", 0),
                    cyberdeck_data.get("matrix_damage", 0), cyberdeck_data.get("cost", 0),
                    cyberdeck_data.get("availability", ""), cyberdeck_data.get("description", "")
                ))

                self._update_character_timestamp(cursor, character_id)
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error updating character cyberdeck: {e}")
            return False

    def add_character_program(self, character_id: int, program_data: Dict[str, Any]) -> bool:
        """Add program to character's cyberdeck."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT INTO character_programs 
                    (character_id, name, rating, program_type, description)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    character_id, program_data["name"], program_data.get("rating", 1),
                    program_data.get("program_type", "common"), program_data.get("description", "")
                ))

                self._update_character_timestamp(cursor, character_id)
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error adding character program: {e}")
            return False

    def remove_character_program(self, character_id: int, program_id: int) -> bool:
        """Remove program from character."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM character_programs WHERE character_id = ? AND id = ?",
                               (character_id, program_id))
                self._update_character_timestamp(cursor, character_id)
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error removing character program: {e}")
            return False

    # ===== REFERENCE DATA =====

    def get_skills_library(self, skill_type: str = None) -> List[Dict[str, Any]]:
        """Get available skills from library for dropdowns."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            if skill_type:
                cursor.execute("""
                    SELECT name, skill_type, skill_group, linked_attribute, description
                    FROM skills_library WHERE skill_type = ? ORDER BY name
                """, (skill_type,))
            else:
                cursor.execute("""
                    SELECT name, skill_type, skill_group, linked_attribute, description
                    FROM skills_library ORDER BY skill_type, name
                """)

            return [
                {
                    "name": row[0], "skill_type": row[1], "skill_group": row[2],
                    "linked_attribute": row[3], "description": row[4]
                }
                for row in cursor.fetchall()
            ]

    def get_qualities_library(self, quality_type: str = None) -> List[Dict[str, Any]]:
        """Get available qualities from library for dropdowns."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            if quality_type:
                cursor.execute("""
                    SELECT name, quality_type, karma_cost, rating_range, description
                    FROM qualities_library WHERE quality_type = ? ORDER BY name
                """, (quality_type,))
            else:
                cursor.execute("""
                    SELECT name, quality_type, karma_cost, rating_range, description
                    FROM qualities_library ORDER BY name
                """)

            return [
                {
                    "name": row[0], "quality_type": row[1], "karma_cost": row[2],
                    "rating_range": row[3], "description": row[4]
                }
                for row in cursor.fetchall()
            ]

    def get_gear_library(self, category: str = None) -> List[Dict[str, Any]]:
        """Get available gear from library for dropdowns."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            if category:
                cursor.execute("""
                    SELECT name, category, subcategory, base_cost, availability,
                           armor_value, rating_range, description
                    FROM gear_library WHERE category = ? ORDER BY name
                """, (category,))
            else:
                cursor.execute("""
                    SELECT name, category, subcategory, base_cost, availability,
                           armor_value, rating_range, description
                    FROM gear_library ORDER BY category, name
                """)

            return [
                {
                    "name": row[0], "category": row[1], "subcategory": row[2],
                    "base_cost": row[3], "availability": row[4], "armor_value": row[5],
                    "rating_range": row[6], "description": row[7]
                }
                for row in cursor.fetchall()
            ]

    def get_gear_categories(self) -> List[str]:
        """Get list of available gear categories."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT category FROM gear_library ORDER BY category")
            return [row[0] for row in cursor.fetchall()]

    # ===== QUERY CONTEXT GENERATION =====

    def generate_character_context_for_query(self, character_id: int) -> str:
        """Generate character context string for RAG queries."""
        char_summary = self.db.get_character_summary(character_id)
        if not char_summary:
            return ""

        context_parts = []

        # Basic character info
        context_parts.append(f"Active Character: {char_summary['name']} ({char_summary['metatype']})")
        if char_summary.get('archetype'):
            context_parts.append(f"Archetype: {char_summary['archetype']}")

        # Key attributes for quick reference
        attrs = char_summary['attributes']
        context_parts.append(
            f"Attributes: Body {attrs['body']}, Agility {attrs['agility']}, "
            f"Logic {attrs['logic']}, Willpower {attrs['willpower']}, "
            f"Charisma {attrs['charisma']}, Edge {attrs['edge']}"
        )

        # Top skills with dice pools
        if char_summary.get('skills'):
            top_skills = sorted(char_summary['skills'], key=lambda x: x.get('dice_pool', 0), reverse=True)[:5]
            skills_text = ", ".join([
                f"{skill['name']} {skill['rating']} ({skill.get('dice_pool', skill['rating'])} dice)"
                for skill in top_skills
            ])
            context_parts.append(f"Top Skills: {skills_text}")

        # Total armor
        if char_summary.get('total_armor', 0) > 0:
            context_parts.append(f"Total Armor: {char_summary['total_armor']}")

        return " | ".join(context_parts)

    def resolve_dice_pool_query(self, character_id: int, query_text: str) -> Optional[str]:
        """Resolve dice pool calculations from natural language queries."""
        char_data = self.get_character_full_data(character_id)
        if not char_data:
            return None

        query_lower = query_text.lower()

        # Common test patterns
        test_patterns = {
            "hacking": ("logic", "hacking"),
            "shooting": ("agility", "pistols"),  # or automatics, longarms
            "perception": ("intuition", "perception"),
            "stealth": ("agility", "infiltration"),
            "social": ("charisma", "con"),  # or etiquette, negotiation
            "athletics": ("agility", "gymnastics"),  # or running
            "first aid": ("logic", "first aid"),
            "driving": ("reaction", "pilot ground craft"),
        }

        # Find matching test
        for test_name, (attr, skill) in test_patterns.items():
            if test_name in query_lower:
                attr_value = char_data.get('stats', {}).get(attr, 0)

                # Find skill rating
                skill_rating = 0
                for skill_data in char_data.get('skills', {}).get('active', []):
                    if skill.lower() in skill_data['name'].lower():
                        skill_rating = skill_data['rating']
                        break

                dice_pool = attr_value + skill_rating

                return f"{char_data['name']}'s {test_name} test: {attr.title()} {attr_value} + {skill.title()} {skill_rating} = {dice_pool} dice"

        return None

    # ===== EXPORT FUNCTIONALITY =====

    def export_character_json(self, character_id: int) -> str:
        """Export character data as JSON for sharing with GM."""
        char_data = self.get_character_full_data(character_id)
        if not char_data:
            return ""

        # Add export metadata
        export_data = {
            "export_info": {
                "exported_at": datetime.now().isoformat(),
                "exported_from": "Shadowrun RAG Assistant",
                "format_version": "1.0"
            },
            "character": char_data
        }

        return json.dumps(export_data, indent=2, ensure_ascii=False)

    def export_character_csv(self, character_id: int) -> str:
        """Export character data as CSV for sharing with GM."""
        char_data = self.get_character_full_data(character_id)
        if not char_data:
            return ""

        import io
        import csv

        output = io.StringIO()
        writer = csv.writer(output)

        # Character basics
        writer.writerow(["Character", char_data['name']])
        writer.writerow(["Metatype", char_data['metatype']])
        writer.writerow(["Archetype", char_data.get('archetype', '')])
        writer.writerow([])  # Empty row

        # Attributes
        writer.writerow(["ATTRIBUTES"])
        if 'stats' in char_data:
            stats = char_data['stats']
            writer.writerow(["Body", stats.get('body', 0)])
            writer.writerow(["Agility", stats.get('agility', 0)])
            writer.writerow(["Reaction", stats.get('reaction', 0)])
            writer.writerow(["Strength", stats.get('strength', 0)])
            writer.writerow(["Willpower", stats.get('willpower', 0)])
            writer.writerow(["Logic", stats.get('logic', 0)])
            writer.writerow(["Intuition", stats.get('intuition', 0)])
            writer.writerow(["Charisma", stats.get('charisma', 0)])
            writer.writerow(["Edge", stats.get('edge', 0)])
            writer.writerow(["Essence", stats.get('essence', 6.0)])
        writer.writerow([])

        # Skills
        writer.writerow(["ACTIVE SKILLS"])
        writer.writerow(["Skill", "Rating", "Specialization", "Dice Pool"])
        for skill in char_data.get('skills', {}).get('active', []):
            attr_val = char_data.get('stats', {}).get(skill.get('attribute', '').lower(), 0)
            dice_pool = attr_val + skill.get('rating', 0)
            writer.writerow([
                skill['name'], skill['rating'],
                skill.get('specialization', ''), dice_pool
            ])
        writer.writerow([])

        # Qualities
        writer.writerow(["QUALITIES"])
        writer.writerow(["Quality", "Rating", "Type", "Cost"])
        for quality in char_data.get('qualities', []):
            writer.writerow([
                quality['name'], quality.get('rating', ''),
                quality.get('quality_type', ''), quality.get('karma_cost', '')
            ])
        writer.writerow([])

        # Gear
        writer.writerow(["GEAR"])
        writer.writerow(["Item", "Quantity", "Category", "Armor", "Cost"])
        for gear in char_data.get('gear', []):
            writer.writerow([
                gear['name'], gear.get('quantity', 1),
                gear.get('category', ''), gear.get('armor_value', 0),
                gear.get('cost', 0)
            ])

        return output.getvalue()

    # ===== UTILITY METHODS =====

    def _update_character_timestamp(self, cursor, character_id: int):
        """Update character's last modified timestamp."""
        cursor.execute(
            "UPDATE characters SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (character_id,)
        )

    def calculate_total_armor(self, character_id: int) -> int:
        """Calculate total armor value from all equipped gear."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COALESCE(SUM(armor_value * quantity), 0)
                FROM character_gear
                WHERE character_id = ? AND armor_value > 0
            """, (character_id,))
            return cursor.fetchone()[0]

    def get_character_skill_dice_pool(self, character_id: int, skill_name: str) -> Tuple[int, str]:
        """Get dice pool for a specific skill."""
        char_data = self.get_character_full_data(character_id)
        if not char_data:
            return 0, "Character not found"

        # Find the skill
        skill_data = None
        for skill in char_data.get('skills', {}).get('active', []):
            if skill['name'].lower() == skill_name.lower():
                skill_data = skill
                break

        if not skill_data:
            return 0, f"Skill '{skill_name}' not found"

        # Get attribute value
        attr_name = skill_data.get('attribute', '').lower()
        attr_value = char_data.get('stats', {}).get(attr_name, 0)
        skill_rating = skill_data.get('rating', 0)

        dice_pool = attr_value + skill_rating

        specialization = skill_data.get('specialization', '')
        spec_text = f" (+2 for {specialization})" if specialization else ""

        return dice_pool, f"{attr_name.title()} {attr_value} + {skill_name} {skill_rating}{spec_text}"


# ===== GLOBAL MANAGER INSTANCE =====

from .characters import get_character_db


def get_character_manager() -> CharacterManager:
    """Get the global character manager instance."""
    return CharacterManager(get_character_db())