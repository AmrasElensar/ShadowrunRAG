"""
Entity Registry Storage System
Stores and manages extracted entities alongside existing chunk storage.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import asdict
import sqlite3
from datetime import datetime

from tools.entity_registry_builder import WeaponStats, SpellStats, ICStats, EntityRegistryBuilder

logger = logging.getLogger(__name__)


class EntityRegistryStorage:
    """Stores and manages entity registries with SQLite backend."""

    def __init__(self, db_path: str = "data/entity_registry.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

        logger.info(f"Entity Registry Storage initialized: {db_path}")

    def _init_database(self):
        """Initialize SQLite database with entity tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Weapons table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS weapons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    accuracy TEXT,
                    damage TEXT,
                    ap TEXT,
                    mode TEXT,
                    rc TEXT,
                    ammo TEXT,
                    avail TEXT,
                    cost TEXT,
                    category TEXT,
                    manufacturer TEXT,
                    description TEXT,
                    source TEXT,
                    chunk_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(name, source)
                )
            ''')

            # Spells table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS spells (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    spell_type TEXT,
                    range_val TEXT,
                    damage TEXT,
                    duration TEXT,
                    drain TEXT,
                    keywords TEXT,
                    category TEXT,
                    description TEXT,
                    source TEXT,
                    chunk_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(name, source)
                )
            ''')

            # IC Programs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ic_programs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    attack_pattern TEXT,
                    resistance_test TEXT,
                    effect_description TEXT,
                    category TEXT,
                    source TEXT,
                    chunk_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(name, source)
                )
            ''')

            # Entity-to-chunk mapping for cross-references
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS entity_chunks (
                    entity_type TEXT,
                    entity_name TEXT,
                    chunk_id TEXT,
                    relevance_score REAL DEFAULT 1.0,
                    PRIMARY KEY (entity_type, entity_name, chunk_id)
                )
            ''')

            conn.commit()
            logger.info("Entity registry database initialized")

    def store_entities(self, entities: Dict[str, List], chunk_id: str, source: str):
        """Store extracted entities from a chunk."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Store weapons
            for weapon in entities.get("weapons", []):
                try:
                    cursor.execute('''
                        INSERT OR REPLACE INTO weapons 
                        (name, accuracy, damage, ap, mode, rc, ammo, avail, cost, 
                         category, manufacturer, description, source, chunk_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        weapon.name, weapon.accuracy, weapon.damage, weapon.ap,
                        weapon.mode, weapon.rc, weapon.ammo, weapon.avail, weapon.cost,
                        weapon.category, weapon.manufacturer, weapon.description,
                        source, chunk_id
                    ))

                    # Add to entity-chunk mapping
                    cursor.execute('''
                        INSERT OR REPLACE INTO entity_chunks 
                        (entity_type, entity_name, chunk_id)
                        VALUES (?, ?, ?)
                    ''', ("weapon", weapon.name, chunk_id))

                except sqlite3.IntegrityError as e:
                    logger.warning(f"Duplicate weapon entry: {weapon.name} in {source}")

            # Store spells
            for spell in entities.get("spells", []):
                try:
                    cursor.execute('''
                        INSERT OR REPLACE INTO spells 
                        (name, spell_type, range_val, damage, duration, drain, 
                         keywords, category, description, source, chunk_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        spell.name, spell.spell_type, spell.range, spell.damage,
                        spell.duration, spell.drain, ','.join(spell.keywords),
                        spell.category, spell.description, source, chunk_id
                    ))

                    cursor.execute('''
                        INSERT OR REPLACE INTO entity_chunks 
                        (entity_type, entity_name, chunk_id)
                        VALUES (?, ?, ?)
                    ''', ("spell", spell.name, chunk_id))

                except sqlite3.IntegrityError as e:
                    logger.warning(f"Duplicate spell entry: {spell.name} in {source}")

            # Store IC programs
            for ic in entities.get("ic_programs", []):
                try:
                    cursor.execute('''
                        INSERT OR REPLACE INTO ic_programs 
                        (name, attack_pattern, resistance_test, effect_description, 
                         category, source, chunk_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        ic.name, ic.attack_pattern, ic.resistance_test,
                        ic.effect_description, ic.category, source, chunk_id
                    ))

                    cursor.execute('''
                        INSERT OR REPLACE INTO entity_chunks 
                        (entity_type, entity_name, chunk_id)
                        VALUES (?, ?, ?)
                    ''', ("ic", ic.name, chunk_id))

                except sqlite3.IntegrityError as e:
                    logger.warning(f"Duplicate IC entry: {ic.name} in {source}")

            conn.commit()

    def get_weapon(self, name: str) -> Optional[WeaponStats]:
        """Retrieve weapon by name (fuzzy matching)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Try exact match first
            cursor.execute('''
                SELECT * FROM weapons WHERE name = ? LIMIT 1
            ''', (name,))

            result = cursor.fetchone()
            if not result:
                # Try partial match
                cursor.execute('''
                    SELECT * FROM weapons WHERE name LIKE ? LIMIT 1
                ''', (f"%{name}%",))
                result = cursor.fetchone()

            if result:
                return WeaponStats(
                    name=result[1], accuracy=result[2], damage=result[3],
                    ap=result[4], mode=result[5], rc=result[6], ammo=result[7],
                    avail=result[8], cost=result[9], category=result[10],
                    manufacturer=result[11], description=result[12]
                )

            return None

    def get_spell(self, name: str) -> Optional[SpellStats]:
        """Retrieve spell by name (fuzzy matching)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT * FROM spells WHERE name LIKE ? LIMIT 1
            ''', (f"%{name}%",))

            result = cursor.fetchone()
            if result:
                return SpellStats(
                    name=result[1], spell_type=result[2], range=result[3],
                    damage=result[4], duration=result[5], drain=result[6],
                    keywords=result[7].split(',') if result[7] else [],
                    category=result[8], description=result[9]
                )

            return None

    def get_ic_program(self, name: str) -> Optional[ICStats]:
        """Retrieve IC program by name (fuzzy matching)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT * FROM ic_programs WHERE name LIKE ? LIMIT 1
            ''', (f"%{name}%",))

            result = cursor.fetchone()
            if result:
                return ICStats(
                    name=result[1], attack_pattern=result[2],
                    resistance_test=result[3], effect_description=result[4],
                    category=result[5]
                )

            return None

    def get_related_chunks(self, entity_type: str, entity_name: str) -> List[str]:
        """Get chunk IDs related to a specific entity."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT chunk_id FROM entity_chunks 
                WHERE entity_type = ? AND entity_name LIKE ?
                ORDER BY relevance_score DESC
            ''', (entity_type, f"%{entity_name}%"))

            return [row[0] for row in cursor.fetchall()]

    def search_weapons_by_mode(self, mode: str) -> List[WeaponStats]:
        """Find all weapons that support a specific firing mode."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT * FROM weapons WHERE mode LIKE ?
            ''', (f"%{mode.upper()}%",))

            weapons = []
            for result in cursor.fetchall():
                weapons.append(WeaponStats(
                    name=result[1], accuracy=result[2], damage=result[3],
                    ap=result[4], mode=result[5], rc=result[6], ammo=result[7],
                    avail=result[8], cost=result[9], category=result[10],
                    manufacturer=result[11], description=result[12]
                ))

            return weapons

    def get_registry_stats(self) -> Dict[str, int]:
        """Get statistics about stored entities."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            stats = {}

            cursor.execute("SELECT COUNT(*) FROM weapons")
            stats["weapons"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM spells")
            stats["spells"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM ic_programs")
            stats["ic_programs"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT chunk_id) FROM entity_chunks")
            stats["chunks_with_entities"] = cursor.fetchone()[0]

            return stats

    def validate_weapon_mode(self, weapon_name: str, mode: str) -> Dict[str, Any]:
        """Validate if a weapon supports the requested mode."""
        weapon = self.get_weapon(weapon_name)
        if not weapon:
            return {"valid": False, "error": f"Weapon '{weapon_name}' not found"}

        # Parse available modes
        available_modes = [m.strip() for m in weapon.mode.split('/')]

        # Mode aliases for common terms
        mode_aliases = {
            "burst": ["BF", "BURST"],
            "burst fire": ["BF", "BURST"],
            "semi-auto": ["SA"],
            "semi-automatic": ["SA"],
            "full auto": ["FA"],
            "full automatic": ["FA"],
            "single shot": ["SS"]
        }

        check_modes = mode_aliases.get(mode.lower(), [mode.upper()])

        for check_mode in check_modes:
            if any(check_mode in avail_mode for avail_mode in available_modes):
                return {
                    "valid": True,
                    "weapon": weapon,
                    "available_modes": available_modes
                }

        return {
            "valid": False,
            "error": f"{weapon.name} cannot fire in {mode} mode",
            "weapon": weapon,
            "available_modes": available_modes
        }


# Factory function for integration
def create_entity_registry_storage(db_path: str = "data/entity_registry.db") -> EntityRegistryStorage:
    """Create entity registry storage instance."""
    return EntityRegistryStorage(db_path)