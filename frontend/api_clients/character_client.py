"""
Character API Client for Shadowrun RAG System
Handles all character-related API operations with proper error handling and caching.
"""

import requests
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class CharacterAPIClient:
    """Clean, organized API client for character management operations."""

    def __init__(self, api_url: str):
        self.api_url = api_url
        self._cache = {
            'characters': None,
            'skills_library': {},
            'qualities_library': {},
            'gear_library': {},
            'gear_categories': None
        }

    # ===== CHARACTER MANAGEMENT =====

    def list_characters(self) -> List[Dict]:
        """Get list of all characters with caching."""
        try:
            response = requests.get(f"{self.api_url}/characters", timeout=5)
            response.raise_for_status()
            characters = response.json().get("characters", [])
            self._cache['characters'] = characters
            return characters
        except Exception as e:
            logger.error(f"Failed to list characters: {e}")
            return self._cache['characters'] or []

    def create_character(self, name: str, metatype: str = "Human", archetype: str = "") -> Dict:
        """Create new character and invalidate cache."""
        try:
            response = requests.post(
                f"{self.api_url}/characters",
                json={"name": name, "metatype": metatype, "archetype": archetype},
                timeout=10
            )
            response.raise_for_status()
            # Invalidate cache
            self._cache['characters'] = None
            return response.json()
        except Exception as e:
            logger.error(f"Failed to create character: {e}")
            return {"error": str(e)}

    def get_character(self, character_id: int) -> Dict:
        """Get full character data."""
        try:
            response = requests.get(f"{self.api_url}/characters/{character_id}", timeout=10)
            response.raise_for_status()
            return response.json().get("character", {})
        except Exception as e:
            logger.error(f"Failed to get character: {e}")
            return {"error": str(e)}

    def delete_character(self, character_id: int) -> Dict:
        """Delete character and invalidate cache."""
        try:
            response = requests.delete(f"{self.api_url}/characters/{character_id}", timeout=5)
            response.raise_for_status()
            # Invalidate cache
            self._cache['characters'] = None
            return response.json()
        except Exception as e:
            logger.error(f"Failed to delete character: {e}")
            return {"error": str(e)}

    def get_active_character(self) -> Optional[Dict]:
        """Get currently active character."""
        try:
            response = requests.get(f"{self.api_url}/characters/active", timeout=5)
            response.raise_for_status()
            return response.json().get("active_character")
        except Exception as e:
            logger.error(f"Failed to get active character: {e}")
            return None

    def set_active_character(self, character_id: int) -> Dict:
        """Set active character for queries."""
        try:
            response = requests.post(f"{self.api_url}/characters/{character_id}/activate", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to set active character: {e}")
            return {"error": str(e)}

    # ===== CHARACTER DATA UPDATES =====

    def update_character_stats(self, character_id: int, stats: Dict) -> Dict:
        """Update character statistics."""
        try:
            response = requests.put(
                f"{self.api_url}/characters/{character_id}/stats",
                json=stats,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to update character stats: {e}")
            return {"error": str(e)}

    def update_character_resources(self, character_id: int, resources: Dict) -> Dict:
        """Update character resources."""
        try:
            response = requests.put(
                f"{self.api_url}/characters/{character_id}/resources",
                json=resources,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to update character resources: {e}")
            return {"error": str(e)}

    # ===== SKILLS MANAGEMENT =====

    def add_character_skill(self, character_id: int, skill_data: Dict) -> Dict:
        """Add skill to character."""
        try:
            response = requests.post(
                f"{self.api_url}/characters/{character_id}/skills",
                json=skill_data,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to add skill: {e}")
            return {"error": str(e)}

    def remove_character_skill(self, character_id: int, skill_name: str) -> Dict:
        """Remove skill from character."""
        try:
            response = requests.delete(
                f"{self.api_url}/characters/{character_id}/skills/{skill_name}",
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to remove skill: {e}")
            return {"error": str(e)}

    # ===== QUALITIES MANAGEMENT =====

    def add_character_quality(self, character_id: int, quality_data: Dict) -> Dict:
        """Add quality to character."""
        try:
            response = requests.post(
                f"{self.api_url}/characters/{character_id}/qualities",
                json=quality_data,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to add quality: {e}")
            return {"error": str(e)}

    def remove_character_quality(self, character_id: int, quality_name: str) -> Dict:
        """Remove quality from character."""
        try:
            response = requests.delete(
                f"{self.api_url}/characters/{character_id}/qualities/{quality_name}",
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to remove quality: {e}")
            return {"error": str(e)}

    # ===== GEAR MANAGEMENT =====

    def add_character_gear(self, character_id: int, gear_data: Dict) -> Dict:
        """Add gear to character."""
        try:
            response = requests.post(
                f"{self.api_url}/characters/{character_id}/gear",
                json=gear_data,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to add gear: {e}")
            return {"error": str(e)}

    def remove_character_gear(self, character_id: int, gear_id: int) -> Dict:
        """Remove gear from character."""
        try:
            response = requests.delete(
                f"{self.api_url}/characters/{character_id}/gear/{gear_id}",
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to remove gear: {e}")
            return {"error": str(e)}

    # ===== WEAPONS & VEHICLES =====

    def add_character_weapon(self, character_id: int, weapon_data: Dict) -> Dict:
        """Add weapon to character."""
        try:
            response = requests.post(
                f"{self.api_url}/characters/{character_id}/weapons",
                json=weapon_data,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to add weapon: {e}")
            return {"error": str(e)}

    def add_character_vehicle(self, character_id: int, vehicle_data: Dict) -> Dict:
        """Add vehicle to character."""
        try:
            response = requests.post(
                f"{self.api_url}/characters/{character_id}/vehicles",
                json=vehicle_data,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to add vehicle: {e}")
            return {"error": str(e)}

    def update_character_cyberdeck(self, character_id: int, cyberdeck_data: Dict) -> Dict:
        """Update character cyberdeck."""
        try:
            response = requests.put(
                f"{self.api_url}/characters/{character_id}/cyberdeck",
                json=cyberdeck_data,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to update cyberdeck: {e}")
            return {"error": str(e)}

    # ===== REFERENCE DATA (WITH PROPER CACHING) =====

    def get_skills_library(self, skill_type: str = None, force_refresh: bool = False) -> List[Dict]:
        """Get skills library with caching - THIS FIXES THE DROPDOWN ISSUE."""
        cache_key = skill_type or 'all'

        if not force_refresh and cache_key in self._cache['skills_library']:
            return self._cache['skills_library'][cache_key]

        try:
            url = f"{self.api_url}/reference/skills"
            if skill_type:
                url += f"?skill_type={skill_type}"

            response = requests.get(url, timeout=10)
            response.raise_for_status()
            skills = response.json().get("skills", [])

            # Cache the result
            self._cache['skills_library'][cache_key] = skills
            logger.info(f"Loaded {len(skills)} skills for type '{skill_type}'")
            return skills

        except Exception as e:
            logger.error(f"Failed to get skills library: {e}")
            return self._cache['skills_library'].get(cache_key, [])

    def get_qualities_library(self, quality_type: str = None, force_refresh: bool = False) -> List[Dict]:
        """Get qualities library with caching - THIS FIXES THE DROPDOWN ISSUE."""
        cache_key = quality_type or 'all'

        if not force_refresh and cache_key in self._cache['qualities_library']:
            return self._cache['qualities_library'][cache_key]

        try:
            url = f"{self.api_url}/reference/qualities"
            if quality_type:
                url += f"?quality_type={quality_type}"

            response = requests.get(url, timeout=10)
            response.raise_for_status()
            qualities = response.json().get("qualities", [])

            # Cache the result
            self._cache['qualities_library'][cache_key] = qualities
            logger.info(f"Loaded {len(qualities)} qualities for type '{quality_type}'")
            return qualities

        except Exception as e:
            logger.error(f"Failed to get qualities library: {e}")
            return self._cache['qualities_library'].get(cache_key, [])

    def get_gear_library(self, category: str = None, force_refresh: bool = False) -> List[Dict]:
        """Get gear library with caching - THIS FIXES THE DROPDOWN ISSUE."""
        cache_key = category or 'all'

        if not force_refresh and cache_key in self._cache['gear_library']:
            return self._cache['gear_library'][cache_key]

        try:
            url = f"{self.api_url}/reference/gear"
            if category:
                url += f"?category={category}"

            response = requests.get(url, timeout=10)
            response.raise_for_status()
            gear = response.json().get("gear", [])

            # Cache the result
            self._cache['gear_library'][cache_key] = gear
            logger.info(f"Loaded {len(gear)} gear items for category '{category}'")
            return gear

        except Exception as e:
            logger.error(f"Failed to get gear library: {e}")
            return self._cache['gear_library'].get(cache_key, [])

    def get_gear_categories(self, force_refresh: bool = False) -> List[str]:
        """Get gear categories with caching."""
        if not force_refresh and self._cache['gear_categories'] is not None:
            return self._cache['gear_categories']

        try:
            response = requests.get(f"{self.api_url}/reference/gear/categories", timeout=5)
            response.raise_for_status()
            categories = response.json().get("categories", [])

            # Cache the result
            self._cache['gear_categories'] = categories
            return categories

        except Exception as e:
            logger.error(f"Failed to get gear categories: {e}")
            return self._cache['gear_categories'] or []

    def populate_reference_data(self) -> Dict:
        """Populate reference data from rulebooks."""
        try:
            response = requests.post(f"{self.api_url}/reference/populate", timeout=60)
            response.raise_for_status()
            # Clear cache after population
            self._clear_reference_cache()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to populate reference data: {e}")
            return {"error": str(e)}

    # ===== UTILITY FUNCTIONS =====

    def get_dice_pool(self, character_id: int, skill_name: str) -> Dict:
        """Get dice pool calculation for skill."""
        try:
            response = requests.get(f"{self.api_url}/characters/{character_id}/dice_pool/{skill_name}", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get dice pool: {e}")
            return {"error": str(e)}

    def get_character_query_context(self, character_id: int) -> Dict:
        """Get character context for RAG queries."""
        try:
            response = requests.get(f"{self.api_url}/characters/{character_id}/context", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get character context: {e}")
            return {"error": str(e)}

    def export_character_json(self, character_id: int) -> bytes:
        """Export character as JSON."""
        try:
            response = requests.get(f"{self.api_url}/characters/{character_id}/export/json", timeout=10)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"Failed to export character JSON: {e}")
            return b""

    # ===== CACHE MANAGEMENT =====

    def _clear_reference_cache(self):
        """Clear reference data cache."""
        self._cache['skills_library'] = {}
        self._cache['qualities_library'] = {}
        self._cache['gear_library'] = {}
        self._cache['gear_categories'] = None

    def clear_character_cache(self):
        """Clear character list cache."""
        self._cache['characters'] = None

    def clear_all_cache(self):
        """Clear all cached data."""
        self._cache = {
            'characters': None,
            'skills_library': {},
            'qualities_library': {},
            'gear_library': {},
            'gear_categories': None
        }

    # ===== DROPDOWN HELPERS (FIXED IMPLEMENTATION) =====

    def get_skills_dropdown_choices(self, skill_type: str) -> List[tuple]:
        """Get formatted choices for skills dropdown - PROPERLY IMPLEMENTED."""
        skills = self.get_skills_library(skill_type)

        if not skills:
            return [("No skills found - try 'Populate Reference Data'", None)]

        choices = []
        for skill in skills:
            # Create descriptive choice with attribute
            attr_text = f" ({skill.get('linked_attribute', 'Unknown')})" if skill.get('linked_attribute') else ""
            choice_text = f"{skill['name']}{attr_text}"
            choices.append((choice_text, skill['name']))

        return choices

    def get_qualities_dropdown_choices(self, quality_type: str) -> List[tuple]:
        """Get formatted choices for qualities dropdown - PROPERLY IMPLEMENTED."""
        qualities = self.get_qualities_library(quality_type)

        if not qualities:
            return [("No qualities found - try 'Populate Reference Data'", None)]

        choices = []
        for quality in qualities:
            # Create descriptive choice with karma cost
            karma_cost = quality.get('karma_cost', 0)
            karma_text = f" ({karma_cost} karma)" if karma_cost != 0 else ""
            choice_text = f"{quality['name']}{karma_text}"
            choices.append((choice_text, quality['name']))

        return choices

    def get_gear_dropdown_choices(self, category: str) -> List[tuple]:
        """Get formatted choices for gear dropdown - PROPERLY IMPLEMENTED."""
        if not category:
            return []

        gear_items = self.get_gear_library(category)

        if not gear_items:
            return [("No gear found in this category", None)]

        choices = []
        for gear in gear_items:
            # Create descriptive choice with cost
            cost = gear.get('base_cost', 0)
            cost_text = f" ({cost}¥)" if cost > 0 else ""
            choice_text = f"{gear['name']}{cost_text}"
            choices.append((choice_text, gear['name']))

        return choices

    def get_character_dropdown_choices(self) -> List[tuple]:
        """Get formatted choices for character dropdown."""
        characters = self.list_characters()

        if not characters:
            return [("No characters", None)]

        choices = []
        active_char = self.get_active_character()
        active_id = active_char.get("id") if active_char else None

        for char in characters:
            label = f"👤 {char['name']} ({char['metatype']})"
            if char['id'] == active_id:
                label += " ⭐"
            choices.append((label, char['id']))

        return choices