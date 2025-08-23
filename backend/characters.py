"""
Character management database schema for Shadowrun RAG system.
SQLite-based storage for character sheets, gear, and campaign data.
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class CharacterDatabase:
    """SQLite database for character management."""

    def __init__(self, db_path: str = "data/characters.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_database()

    def get_connection(self):
        """Get database connection."""
        return sqlite3.connect(self.db_path)

    def init_database(self):
        """Initialize database with all required tables."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Main characters table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS characters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    metatype TEXT DEFAULT 'Human',
                    archetype TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    notes TEXT
                )
            ''')

            # Character statistics (attributes and limits)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS character_stats (
                    character_id INTEGER PRIMARY KEY,
                    body INTEGER DEFAULT 1,
                    agility INTEGER DEFAULT 1,
                    reaction INTEGER DEFAULT 1,
                    strength INTEGER DEFAULT 1,
                    charisma INTEGER DEFAULT 1,
                    logic INTEGER DEFAULT 1,
                    intuition INTEGER DEFAULT 1,
                    willpower INTEGER DEFAULT 1,
                    edge INTEGER DEFAULT 1,
                    essence REAL DEFAULT 6.0,
                    physical_limit INTEGER DEFAULT 1,
                    mental_limit INTEGER DEFAULT 1,
                    social_limit INTEGER DEFAULT 1,
                    initiative INTEGER DEFAULT 1,
                    hot_sim_vr INTEGER DEFAULT 0,
                    FOREIGN KEY (character_id) REFERENCES characters (id) ON DELETE CASCADE
                )
            ''')

            # Character resources
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS character_resources (
                    character_id INTEGER PRIMARY KEY,
                    nuyen INTEGER DEFAULT 0,
                    street_cred INTEGER DEFAULT 0,
                    notoriety INTEGER DEFAULT 0,
                    public_aware INTEGER DEFAULT 0,
                    total_karma INTEGER DEFAULT 0,
                    available_karma INTEGER DEFAULT 0,
                    edge_pool INTEGER DEFAULT 1,
                    FOREIGN KEY (character_id) REFERENCES characters (id) ON DELETE CASCADE
                )
            ''')

            # Character qualities
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS character_qualities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    character_id INTEGER,
                    name TEXT NOT NULL,
                    rating INTEGER DEFAULT 0,
                    karma_cost INTEGER DEFAULT 0,
                    description TEXT,
                    quality_type TEXT DEFAULT 'positive',
                    FOREIGN KEY (character_id) REFERENCES characters (id) ON DELETE CASCADE
                )
            ''')

            # Character skills (active, knowledge, language)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS character_skills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    character_id INTEGER,
                    name TEXT NOT NULL,
                    rating INTEGER DEFAULT 0,
                    specialization TEXT,
                    skill_type TEXT NOT NULL, -- 'active', 'knowledge', 'language'
                    skill_group TEXT,
                    attribute TEXT,
                    FOREIGN KEY (character_id) REFERENCES characters (id) ON DELETE CASCADE
                )
            ''')

            # Character gear
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS character_gear (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    character_id INTEGER,
                    name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    subcategory TEXT,
                    quantity INTEGER DEFAULT 1,
                    rating INTEGER,
                    armor_value INTEGER DEFAULT 0,
                    cost INTEGER,
                    availability TEXT,
                    description TEXT,
                    custom_properties TEXT, -- JSON for flexible properties
                    FOREIGN KEY (character_id) REFERENCES characters (id) ON DELETE CASCADE
                )
            ''')

            # Character weapons
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS character_weapons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    character_id INTEGER,
                    name TEXT NOT NULL,
                    weapon_type TEXT, -- 'melee', 'ranged'
                    mode_ammo TEXT,
                    accuracy INTEGER,
                    damage_code TEXT,
                    armor_penetration INTEGER DEFAULT 0,
                    recoil_compensation INTEGER DEFAULT 0,
                    cost INTEGER,
                    availability TEXT,
                    description TEXT,
                    FOREIGN KEY (character_id) REFERENCES characters (id) ON DELETE CASCADE
                )
            ''')

            # Character vehicles and drones
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS character_vehicles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    character_id INTEGER,
                    name TEXT NOT NULL,
                    vehicle_type TEXT, -- 'vehicle', 'drone'
                    handling INTEGER,
                    speed INTEGER,
                    acceleration INTEGER,
                    body INTEGER,
                    armor INTEGER,
                    pilot INTEGER,
                    sensor INTEGER,
                    seats INTEGER,
                    cost INTEGER,
                    availability TEXT,
                    description TEXT,
                    FOREIGN KEY (character_id) REFERENCES characters (id) ON DELETE CASCADE
                )
            ''')

            # Character cyberdeck
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS character_cyberdeck (
                    character_id INTEGER PRIMARY KEY,
                    name TEXT,
                    device_rating INTEGER DEFAULT 1,
                    attack INTEGER DEFAULT 0,
                    sleaze INTEGER DEFAULT 0,
                    firewall INTEGER DEFAULT 0,
                    data_processing INTEGER DEFAULT 0,
                    matrix_damage INTEGER DEFAULT 0,
                    cost INTEGER,
                    availability TEXT,
                    description TEXT,
                    FOREIGN KEY (character_id) REFERENCES characters (id) ON DELETE CASCADE
                )
            ''')

            # Cyberdeck programs
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS character_programs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    character_id INTEGER,
                    name TEXT NOT NULL,
                    rating INTEGER DEFAULT 1,
                    program_type TEXT, -- 'common', 'hacking', 'cybercombat', etc.
                    description TEXT,
                    FOREIGN KEY (character_id) REFERENCES characters (id) ON DELETE CASCADE
                )
            ''')

            # Reference tables for dropdowns (populated from rulebooks)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS gear_library (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    category TEXT NOT NULL,
                    subcategory TEXT,
                    base_cost INTEGER,
                    availability TEXT,
                    armor_value INTEGER DEFAULT 0,
                    rating_range TEXT, -- e.g., "1-6" for variable ratings
                    description TEXT,
                    source_page TEXT,
                    properties TEXT -- JSON for additional properties
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS skills_library (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    skill_type TEXT NOT NULL,
                    skill_group TEXT,
                    linked_attribute TEXT,
                    default_allowed BOOLEAN DEFAULT 1,
                    description TEXT,
                    source_page TEXT
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS qualities_library (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    quality_type TEXT, -- 'positive', 'negative'
                    karma_cost INTEGER,
                    rating_range TEXT,
                    description TEXT,
                    source_page TEXT
                )
            ''')

            # Application settings
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS app_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            ''')

            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_character_skills_char_id ON character_skills(character_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_character_gear_char_id ON character_gear(character_id)')
            cursor.execute(
                'CREATE INDEX IF NOT EXISTS idx_character_weapons_char_id ON character_weapons(character_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_gear_library_category ON gear_library(category)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_skills_library_type ON skills_library(skill_type)')

            conn.commit()
            logger.info("Character database initialized successfully")

    def get_active_character_id(self) -> Optional[int]:
        """Get the currently active character ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM app_settings WHERE key = 'active_character_id'")
            result = cursor.fetchone()
            return int(result[0]) if result else None

    def set_active_character_id(self, character_id: int):
        """Set the active character ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO app_settings (key, value) VALUES ('active_character_id', ?)",
                (str(character_id),)
            )
            conn.commit()

    def create_character(self, name: str, metatype: str = "Human", archetype: str = "") -> int:
        """Create a new character with default values."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Create main character record
            cursor.execute(
                "INSERT INTO characters (name, metatype, archetype) VALUES (?, ?, ?)",
                (name, metatype, archetype)
            )
            character_id = cursor.lastrowid

            # Create default stats record
            cursor.execute(
                "INSERT INTO character_stats (character_id) VALUES (?)",
                (character_id,)
            )

            # Create default resources record
            cursor.execute(
                "INSERT INTO character_resources (character_id) VALUES (?)",
                (character_id,)
            )

            conn.commit()
            logger.info(f"Created character: {name} (ID: {character_id})")
            return character_id

    def list_characters(self) -> List[Dict[str, Any]]:
        """List all characters."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, metatype, archetype, created_at FROM characters ORDER BY name")
            rows = cursor.fetchall()
            return [
                {
                    "id": row[0],
                    "name": row[1],
                    "metatype": row[2],
                    "archetype": row[3],
                    "created_at": row[4]
                }
                for row in rows
            ]

    def get_character_summary(self, character_id: int) -> Optional[Dict[str, Any]]:
        """Get character basic info and computed stats for query context."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Get basic character info
            cursor.execute("""
                SELECT c.name, c.metatype, c.archetype,
                       s.body, s.agility, s.reaction, s.strength, 
                       s.charisma, s.logic, s.intuition, s.willpower,
                       s.edge, s.essence, s.initiative
                FROM characters c
                JOIN character_stats s ON c.id = s.character_id
                WHERE c.id = ?
            """, (character_id,))

            row = cursor.fetchone()
            if not row:
                return None

            character_summary = {
                "id": character_id,
                "name": row[0],
                "metatype": row[1],
                "archetype": row[2],
                "attributes": {
                    "body": row[3], "agility": row[4], "reaction": row[5], "strength": row[6],
                    "charisma": row[7], "logic": row[8], "intuition": row[9], "willpower": row[10],
                    "edge": row[11], "essence": row[12], "initiative": row[13]
                }
            }

            # Get active skills for quick reference
            cursor.execute("""
                SELECT name, rating, specialization, attribute
                FROM character_skills 
                WHERE character_id = ? AND skill_type = 'active' AND rating > 0
                ORDER BY rating DESC, name
            """, (character_id,))

            skills = []
            for skill_row in cursor.fetchall():
                skill = {"name": skill_row[0], "rating": skill_row[1]}
                if skill_row[2]:  # specialization
                    skill["specialization"] = skill_row[2]
                if skill_row[3]:  # linked attribute
                    skill["attribute"] = skill_row[3]
                    skill["dice_pool"] = character_summary["attributes"].get(skill_row[3].lower(), 0) + skill_row[1]
                skills.append(skill)

            character_summary["skills"] = skills

            # Get total armor value
            cursor.execute("""
                SELECT COALESCE(SUM(armor_value * quantity), 0)
                FROM character_gear
                WHERE character_id = ? AND armor_value > 0
            """, (character_id,))

            armor_total = cursor.fetchone()[0]
            character_summary["total_armor"] = armor_total

            return character_summary


# Global database instance
db = CharacterDatabase()


def get_character_db() -> CharacterDatabase:
    """Get the global character database instance."""
    return db