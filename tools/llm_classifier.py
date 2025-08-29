"""
LLM-based chunk classifier for Shadowrun content using local small models.
Replaces pattern-based classification with semantic understanding.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import ollama

logger = logging.getLogger(__name__)


class LLMShadowrunClassifier:
    """LLM-powered classifier that understands Shadowrun content semantically."""

    def __init__(self, model_name: str = "phi4-mini", timeout: int = 30):
        self.model_name = model_name
        self.timeout = timeout
        self.fallback_classifier = PatternBasedFallback()

        # Test if model is available
        self._ensure_model_available()

        # Classification prompt template
        self.classification_prompt = """Analyze this Shadowrun tabletop RPG content and classify it accurately.

Content to classify:
{text}

Instructions:
1. Select ALL applicable categories (can be multiple)
2. Determine the content type
3. Assess if it contains actual game rules

Categories (select all that apply):
- Combat: Attack rolls, damage, initiative, armor, weapon mechanics, condition monitors
- Magic: Spells, spirits, astral projection, magical traditions, drain, force ratings  
- Matrix: Hacking, cybercombat, programs, IC, hosts, data processing, cyberdecks
- Skills: Skill tests, dice pools, thresholds, specializations, defaulting rules
- Gear: Equipment stats, weapons, armor, electronics, availability, costs
- Character_Creation: Priority system, attribute assignment, skill points, metatypes
- Social: Reputation, contacts, etiquette, negotiation, lifestyle
- Riggers: Drones, vehicles, jumped-in control, vehicle combat, pilot programs
- Game_Mechanics: Core rules, test procedures, karma, edge, limits, game flow
- Setting: Background, corporations, locations, history, world information

Content Types:
- explicit_rule: Contains specific game mechanics, dice formulas, procedures
- example: Character stories, scenarios, sample situations  
- table: Equipment lists, reference tables, stat blocks
- narrative: Background text, flavor text, setting descriptions
- index: Page references, table of contents, cross-references

CRITICAL: Respond ONLY with valid JSON in this exact format:
{{
  "categories": ["category1", "category2"],
  "content_type": "explicit_rule", 
  "contains_rules": true,
  "confidence": 0.85,
  "reasoning": "Brief explanation of classification"
}}

CRITICAL: Your response must be ONLY the JSON object. No explanations, no reasoning steps, no other text."""

    def _ensure_model_available(self):
        """Check if the specified model is available, suggest alternatives if not."""
        try:
            # Test if model responds
            test_response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": "Test"}],
                options={"num_predict": 5}
            )
            logger.info(f"LLM classifier using model: {self.model_name}")

        except Exception as e:
            logger.warning(f"Model {self.model_name} not available: {e}")

            # Try common alternatives
            alternatives = ["qwen3:1.7b", "llama3.2:1b", "gemma3:1b"]

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

            logger.error("No suitable LLM model found. Using pattern-based fallback only.")
            self.model_name = None

    def classify_content(self, text: str, source: str) -> Dict[str, Any]:
        """Main classification method using LLM + fallback."""

        # Check for index content first (quick filter)
        if self._is_index_content(text):
            return self._create_index_metadata(source)

        # Try LLM classification first
        if self.model_name:
            llm_result = self._classify_with_llm(text)
            if llm_result and llm_result.get("confidence", 0) > 0.4:
                # LLM classification succeeded
                return self._create_metadata_from_llm_result(llm_result, text, source)

        # Fallback to pattern-based classification
        logger.warning(f"Using fallback classification for chunk from {source}")
        return self.fallback_classifier.classify_content(text, source)

    def _classify_with_llm(self, text: str, max_retries: int = 2) -> Optional[Dict]:
        """Classify content using LLM with retry logic."""

        # Truncate text for efficiency (keep first and last parts for context)
        text_sample = self._prepare_text_for_llm(text)

        for attempt in range(max_retries + 1):
            try:
                # Make LLM request
                response = ollama.chat(
                    model=self.model_name,
                    messages=[{
                        "role": "user",
                        "content": self.classification_prompt.format(text=text_sample)
                    }],
                    options={
                        "temperature": 0.2,  # Low temperature for consistent classification
                        "num_predict": 1000,  # Limit response length
                        "top_p": 0.9
                    }
                )

                # Parse JSON response
                response_text = response['message']['content'].strip()

                # Clean up response (remove any markdown formatting)
                response_text = self._clean_json_response(response_text)

                result = json.loads(response_text)

                # Validate result format
                if self._validate_llm_result(result):
                    return result
                else:
                    logger.warning(f"Invalid LLM response format (attempt {attempt + 1}): {result}")

            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error (attempt {attempt + 1}): {e}")
                logger.warning(f"Raw response: {response_text}")

            except Exception as e:
                logger.warning(f"LLM classification error (attempt {attempt + 1}): {e}")

            # Add small delay between retries
            if attempt < max_retries:
                time.sleep(0.5)

        logger.error(f"LLM classification failed after {max_retries + 1} attempts")
        return None

    def _prepare_text_for_llm(self, text: str, max_chars: int = 1500) -> str:
        """Prepare text sample for LLM analysis."""

        if len(text) <= max_chars:
            return text

        # Take first part + last part for context
        first_part = text[:max_chars // 2]
        last_part = text[-max_chars // 2:]

        return f"{first_part}\n\n[... content truncated ...]\n\n{last_part}"

    def _clean_json_response(self, response: str) -> str:
        """Clean LLM response to extract valid JSON from reasoning models."""

        # Handle multiple patterns of thinking tags
        patterns_to_remove = [
            r'<think>.*?</think>',  # Normal thinking blocks
            r'<thinking>.*?</thinking>',  # Alternative thinking blocks
            r'<think><think>.*?</think>',  # Nested opening tags
            r'<think>.*?</think></think>',  # Nested closing tags
        ]

        for pattern in patterns_to_remove:
            response = re.sub(pattern, '', response, flags=re.DOTALL)

        # Handle unclosed thinking tags - remove everything from <think> to first {
        if '<think>' in response and response.count('<think>') != response.count('</think>'):
            # Find the last <think> without matching </think>
            parts = response.split('<think>')
            if len(parts) > 1:
                # Keep everything before first <think>, then look for JSON after
                before_think = parts[0]
                after_think = '<think>'.join(parts[1:])

                # Look for JSON in the after_think portion
                json_start = after_think.find('{')
                if json_start != -1:
                    response = before_think + after_think[json_start:]
                else:
                    response = before_think

        # Remove markdown code blocks
        response = re.sub(r'```json\s*', '', response, flags=re.DOTALL)
        response = re.sub(r'```\s*', '', response, flags=re.DOTALL)

        # Find JSON object boundaries
        start = response.find('{')
        end = response.rfind('}') + 1

        if start != -1 and end > start:
            return response[start:end]

        return response.strip()

    def _validate_llm_result(self, result: Dict) -> bool:
        """Validate LLM classification result format."""

        # Required fields
        required_keys = ["categories", "content_type", "contains_rules"]

        if not all(key in result for key in required_keys):
            return False

        if not isinstance(result["categories"], list):
            return False

        # Confidence is optional - add default if missing
        if "confidence" not in result:
            result["confidence"] = 0.7  # Default confidence

        if not isinstance(result["confidence"], (int, float)):
            return False

        if result["confidence"] < 0 or result["confidence"] > 1:
            return False

        return True

    def _create_metadata_from_llm_result(self, llm_result: Dict, text: str, source: str) -> Dict[str, Any]:
        """Create full metadata object from LLM classification result."""

        categories = llm_result.get("categories", ["General"])
        primary_category = categories[0] if categories else "General"

        # Build comprehensive metadata
        metadata = {
            "source": source,
            "document_type": self._detect_document_type(text, source),
            "edition": self._detect_edition(text, source),
            "sections": categories,
            "primary_section": primary_category,
            "main_section": primary_category,  # For compatibility
            "confidence_scores": {primary_category: llm_result.get("confidence", 0.7)},
            "content_type": llm_result.get("content_type", "general"),
            "contains_dice_pools": self._contains_dice_pools(text),
            "contains_tables": self._contains_tables(text),
            "is_rule_definition": llm_result.get("contains_rules", False),
            "mechanical_keywords": self._extract_mechanical_keywords(text),
            "page_references": self._extract_page_references(text),
            "llm_reasoning": llm_result.get("reasoning", ""),
            "classification_method": "llm",
            "llm_confidence": llm_result.get("confidence", 0.7)
        }

        return metadata

    def _create_index_metadata(self, source: str) -> Dict[str, Any]:
        """Create metadata for index content (to be excluded)."""

        return {
            "source": source,
            "document_type": "index_excluded",
            "edition": "unknown",
            "sections": ["Index_Excluded"],
            "primary_section": "Index_Excluded",
            "main_section": "Index_Excluded",
            "confidence_scores": {},
            "content_type": "index",
            "contains_dice_pools": False,
            "contains_tables": False,
            "is_rule_definition": False,
            "mechanical_keywords": [],
            "page_references": [],
            "classification_method": "rule_based"
        }

    def _is_index_content(self, text: str) -> bool:
        """Detect master index content that should be excluded."""

        # Count page reference patterns like "**SR5** 169"
        page_ref_pattern = r'\*\*[A-Z0-9]+\*\*\s+\d+[-,\s\d]*'
        page_ref_count = len(re.findall(page_ref_pattern, text))

        # Index content is dense with page references
        if page_ref_count > 20:
            return True

        # Check for explicit index indicators
        index_indicators = [
            "master index", "shadowrun, fifth edition master index",
            "table of contents"
        ]

        text_lower = text.lower()
        if any(indicator in text_lower for indicator in index_indicators):
            return True

        # Check ratio of page refs to content length
        if len(text) > 0 and page_ref_count / len(text) * 1000 > 5:
            return True

        return False

    def _detect_document_type(self, text: str, source: str) -> str:
        """Detect document type from content and filename."""

        text_lower = text.lower()
        filename_lower = Path(source).name.lower()

        if "character" in filename_lower or "sheet" in filename_lower:
            return "character_sheet"

        if any(term in text_lower for term in ["core rules", "rulebook", "dice pool", "test"]):
            return "rulebook"

        if "adventure" in filename_lower or "scenario" in text_lower:
            return "adventure"

        return "unknown"

    def _detect_edition(self, text: str, source: str) -> str:
        """Detect Shadowrun edition."""

        combined_text = f"{Path(source).name} {text[:1000]}".lower()

        edition_patterns = {
            "SR6": ["shadowrun 6", "6th edition", "sr6", "sixth edition"],
            "SR5": ["shadowrun 5", "5th edition", "sr5", "fifth edition"],
            "SR4": ["shadowrun 4", "4th edition", "sr4", "fourth edition"]
        }

        for edition, patterns in edition_patterns.items():
            if any(pattern in combined_text for pattern in patterns):
                return edition

        return "unknown"

    def _contains_dice_pools(self, text: str) -> bool:
        """Check if content contains dice pool references."""
        dice_patterns = [r"dice pool", r"\d+d6", r"roll.*\+", r"test.*:", r"threshold.*\d+"]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in dice_patterns)

    def _contains_tables(self, text: str) -> bool:
        """Check if content contains tables."""
        return text.count("|") > 10 and ("---" in text or re.search(r"\|\s*\w+\s*\|\s*\d+\s*\|", text))

    def _extract_mechanical_keywords(self, text: str) -> List[str]:
        """Extract game mechanical terms."""

        keywords = []
        text_lower = text.lower()

        # Attributes
        attributes = ["body", "agility", "reaction", "strength", "charisma",
                      "intuition", "logic", "willpower", "edge", "magic", "resonance"]
        keywords.extend([attr for attr in attributes if attr in text_lower])

        # Game mechanics
        mechanics = ["initiative", "dice pool", "threshold", "hits", "glitch",
                     "armor", "damage", "stun", "physical", "condition monitor"]
        keywords.extend([mech for mech in mechanics if mech in text_lower])

        return list(set(keywords))

    def _extract_page_references(self, text: str) -> List[str]:
        """Extract page references from text."""

        pattern = r'\[([^\]]+)\]\(#page-(\d+)-(\d+)\)'
        matches = re.findall(pattern, text)

        return [f"{name} (page {page})" for name, page, _ in matches]


class PatternBasedFallback:
    """Fallback pattern-based classifier when LLM is unavailable."""

    def __init__(self):
        self.section_patterns = {
            "Combat": ["damage", "armor", "weapon", "initiative", "attack"],
            "Magic": ["spell", "magic", "astral", "spirit", "mage"],
            "Matrix": ["matrix", "decker", "program", "hacking", "cyberdeck"],
            "Skills": ["test", "dice pool", "threshold", "skill"],
            "Gear": ["gear", "equipment", "cost", "availability"]
        }

    def classify_content(self, text: str, source: str) -> Dict[str, Any]:
        """Simple pattern-based classification."""

        text_lower = text.lower()

        # Score each section
        section_scores = {}
        for section, patterns in self.section_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text_lower)
            if score > 0:
                section_scores[section] = score

        # Determine primary section
        if section_scores:
            primary_section = max(section_scores, key=section_scores.get)
            sections = [primary_section]
        else:
            primary_section = "General"
            sections = ["General"]

        return {
            "source": source,
            "document_type": "rulebook",
            "edition": "unknown",
            "sections": sections,
            "primary_section": primary_section,
            "main_section": primary_section,
            "confidence_scores": section_scores,
            "content_type": "general",
            "contains_dice_pools": "dice pool" in text_lower,
            "contains_tables": text.count("|") > 5,
            "is_rule_definition": "test" in text_lower or "dice pool" in text_lower,
            "mechanical_keywords": [],
            "page_references": [],
            "classification_method": "pattern_fallback"
        }


# Integration function for indexer.py
def create_llm_classifier(model_name: str = "phi4-mini") -> LLMShadowrunClassifier:
    """Factory function to create LLM classifier."""
    return LLMShadowrunClassifier(model_name=model_name)