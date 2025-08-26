#!/usr/bin/env python3
"""
Improved indexer with multi-label classification and semantic chunking.

This replaces the problematic functions in backend/indexer.py
"""

import re
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Set, Tuple
import logging

logger = logging.getLogger(__name__)


class ImprovedShadowrunClassifier:
    """Enhanced multi-label classifier for Shadowrun content with all required methods."""

    def __init__(self):
        # Updated patterns based on your debug output
        self.section_patterns = {
            "Combat": {
                "primary": [
                    # Exact keywords from your content
                    "combat", "weapon", "armor", "damage", "attack", "defense",
                    "initiative", "stun", "ranged", "melee",
                    # Table/link patterns from your chunks
                    "heavy weapons", "longarms", "pistols", "firearms",
                    "call a shot", "multiple attacks", "combat turns",
                    "condition monitor", "wound", "knockdown"
                ],
                "secondary": [
                    "free actions", "simple actions", "complex actions",
                    "physical damage", "stun damage", "armor penetration",
                    "recoil", "scatter", "suppressing fire", "full auto"
                ],
                "exclude_if": ["spell", "program", "matrix action"]
            },

            "Magic": {
                "primary": [
                    # From your debug: magic(39), mage(30), spell(22), force(17), spirit(15)
                    "magic", "mage", "spell", "force", "spirit", "astral",
                    "tradition", "adept", "summoning", "enchanting",
                    "drain", "spellcasting", "magical"
                ],
                "secondary": [
                    "astral space", "astral perception", "astral combat",
                    "spirit services", "binding", "banishing", "focus",
                    "ritual", "metamagic", "initiation", "lodge"
                ],
                "exclude_if": ["cyberdeck", "program", "ic", "matrix"]
            },

            "Matrix": {
                "primary": [
                    # Use word boundaries for IC - matches " ic " or "ic," or "ic." etc.
                    r"\bic\b",  # Word boundary pattern
                    "matrix", "decker", "program", "hacking", "cyberdeck",
                    "host", "firewall", "data processing", "sleaze",
                    "technomancer", "complex form", "cybercombat"
                ],
                "secondary": [
                    "electronic warfare", "hack on the fly",
                    "brute force", "crack file", "cyberjack", "persona",
                    "access id", "admin", "user", "grid", "black ic", "white ic"
                ],
                "exclude_if": [
                    "physical damage", "astral", "spell", "armor", "weapon",
                    "corporation", "public", "magic"
                ]
            },

            "Skills": {
                "primary": [
                    # From your content structure
                    "test", "dice pool", "threshold", "hits", "glitches",
                    "success tests", "opposed tests", "extended tests",
                    "teamwork tests", "defaulting", "specialization"
                ],
                "secondary": [
                    "skill group", "active skills", "knowledge skills",
                    "language skills", "limit", "edge", "buying hits",
                    "trying again"
                ],
                "exclude_if": ["spell", "program", "gear rating"]
            },

            "Character_Creation": {
                "primary": [
                    "priority", "attributes", "metatype", "character creation",
                    "build points", "karma", "starting", "background",
                    "your character"
                ],
                "secondary": [
                    "contacts", "lifestyle", "starting gear", "starting nuyen",
                    "knowledge skills", "active skills", "street cred",
                    "notoriety", "public awareness"
                ],
                "exclude_if": ["combat", "spellcasting", "hacking"]
            },

            "Gear": {
                "primary": [
                    # Look for actual gear patterns in your content
                    "availability", "cost", "rating", "capacity",
                    "gear", "equipment", "electronics", "nuyen", "¥",
                    "ammunition", "accessories"
                ],
                "secondary": [
                    "cyberware", "bioware", "vehicle", "drone",
                    "weapon modification", "armor modification",
                    "lifestyle", "fake id", "credstick"
                ],
                "exclude_if": ["dice pool", "test", "initiative"]
            },

            "Riggers": {
                "primary": [
                    "rigger", "drone", "vehicle", "pilot", "jumped in",
                    "control rig", "vehicle test", "handling", "speed",
                    "acceleration"
                ],
                "secondary": [
                    "autosofts", "sensor", "ram", "pursuit",
                    "vehicle modification", "body", "armor",
                    "pilot program"
                ],
                "exclude_if": ["spell", "astral", "matrix ic"]
            },

            "Social": {
                "primary": [
                    "social", "etiquette", "negotiation", "leadership",
                    "contacts", "street cred", "notoriety", "reputation",
                    "charisma"
                ],
                "secondary": [
                    "social test", "social modifiers", "lifestyle",
                    "networking", "favors", "connection", "loyalty"
                ],
                "exclude_if": ["combat", "magic", "matrix", "hacking"]
            },

            # New section based on your content
            "Game_Mechanics": {
                "primary": [
                    "shadowrun concepts", "how to make things happen",
                    "hits & thresholds", "buying hits", "glitches",
                    "tests and limits", "time passing", "actions"
                ],
                "secondary": [
                    "gamemaster", "the game & you", "success tests",
                    "opposed tests", "extended tests", "teamwork tests"
                ],
                "exclude_if": ["specific spell", "specific gear"]
            }
        }

        # Updated content type patterns for your format
        self.content_type_patterns = {
            "explicit_rule": [
                r"dice pool.*:", r"test.*:", r"threshold.*:",
                r"make a.*test", r"roll.*\+", r"opposed by",
                r"resistance.*:", r"the dice pool for"
            ],
            "example": [
                r"example:", r"for example", r"e\.g\.", r"suppose",
                r"let's say", r"imagine", r"scenario:", r"for instance"
            ],
            "table_content": [
                # Matches your table format
                r"\|\s*[A-Z][a-zA-Z\s]+\s*\|\s*\d+\s*\|",  # | Action Name | 163 |
                r"\|\s*[-=]+\s*\|",  # Table separators
                r"^\s*\|.*\|.*\|",  # Any table row
            ],
            "page_reference": [
                # Matches your link format
                r"\[.*\]\(#page-\d+-\d+\)",  # [Heavy Weapons](#page-133-0)
                r"see page", r"refer to", r"as described in"
            ],
            "header_section": [
                r"^#+\s+",  # Markdown headers
                r"#### \*\*[A-Z\s]+\*\*"  # #### **SHADOWRUN CONCEPTS 44**
            ]
        }

        # Document type indicators
        self.document_type_patterns = {
            "rulebook": [
                "core rules", "game master", "dice pool", "shadowrun", "catalyst game labs",
                "table of contents", "chapter", "game mechanics", "test", "threshold",
                "shadowrun concepts", "how to make things happen"
            ],
            "character_sheet": [
                "player character", "character name", "karma", "total karma",
                "character points", "street cred", "notoriety", "contacts:"
            ],
            "adventure": [
                "adventure", "scenario", "handout", "scene", "gamemaster section",
                "plot hook", "background", "getting the team together"
            ]
        }

    def classify_content(self, text: str, source: str) -> Dict:
        """Enhanced multi-label classification that works with your content format."""

        # Check for index content first - exclude it entirely
        if self._is_index_content(text):
            # Return metadata that will exclude this from search
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
                "page_references": []
            }

        # Analyze more content for better accuracy
        full_text = text.lower()
        filename = Path(source).name.lower()

        metadata = {
            "source": source,
            "document_type": self._detect_document_type(full_text, filename),
            "edition": self._detect_edition(full_text, filename),
            "sections": [],
            "primary_section": "General",
            "confidence_scores": {},
            "content_type": self._detect_content_type(text),  # Pass original case for patterns
            "contains_dice_pools": "dice pool" in full_text,
            "contains_tables": self._contains_tables(text),
            "is_rule_definition": self._is_rule_definition(full_text),
            "mechanical_keywords": self._extract_mechanical_keywords(full_text),
            "page_references": self._extract_page_references(text)
        }

        # Multi-label section classification
        section_scores = self._calculate_section_scores(full_text)

        # Set confidence threshold lower to catch more sections
        valid_sections = {
            section: score for section, score in section_scores.items()
            if score > 0.1  # Lower threshold
        }

        if valid_sections:
            # Sort by confidence
            sorted_sections = sorted(valid_sections.items(), key=lambda x: x[1], reverse=True)

            # Take top sections (allow more multi-labeling)
            top_sections = [section for section, score in sorted_sections if score > 0.2]

            metadata["sections"] = top_sections if top_sections else [sorted_sections[0][0]]
            metadata["primary_section"] = sorted_sections[0][0]
            metadata["main_section"] = sorted_sections[0][0]
            metadata["confidence_scores"] = {k: round(v, 3) for k, v in valid_sections.items()}
        else:
            # Enhanced fallback - look for any content indicators
            fallback_section = self._fallback_classification(full_text)
            metadata["sections"] = [fallback_section]
            metadata["primary_section"] = fallback_section
            metadata["confidence_scores"] = {fallback_section: 0.5}

        return metadata

    def _is_index_content(self, text: str) -> bool:
        """Detect master index content that should be excluded from search."""

        # Count page reference patterns like "**SR5** 169" or "**RG** 84-87"
        page_ref_pattern = r'\*\*[A-Z0-9]+\*\*\s+\d+[-,\s\d]*'
        page_ref_count = len(re.findall(page_ref_pattern, text))

        # Index content is dense with page references
        if page_ref_count > 20:  # High threshold to catch index pages
            return True

        # Check for explicit index indicators
        index_indicators = [
            "master index", "shadowrun, fifth edition master index",
            "table of contents", "index"
        ]

        text_lower = text.lower()
        if any(indicator in text_lower for indicator in index_indicators):
            return True

        # Check ratio of page refs to total content
        if len(text) > 0 and page_ref_count / len(text) * 1000 > 5:  # Very dense page refs
            return True

        return False

    def _detect_document_type(self, text: str, filename: str) -> str:
        """Detect document type with improved accuracy."""

        combined_text = f"{filename} {text}"

        scores = {}
        for doc_type, patterns in self.document_type_patterns.items():
            score = sum(1 for pattern in patterns if pattern in combined_text)
            scores[doc_type] = score

        if not scores or max(scores.values()) == 0:
            return "unknown"

        return max(scores, key=scores.get)

    def _detect_edition(self, text: str, filename: str) -> str:
        """Detect Shadowrun edition."""

        combined_text = f"{filename} {text}"

        edition_patterns = {
            "SR6": ["shadowrun 6", "6th edition", "sr6", "sixth edition"],
            "SR5": ["shadowrun 5", "5th edition", "sr5", "fifth edition"],
            "SR4": ["shadowrun 4", "4th edition", "sr4", "fourth edition"],
            "SR3": ["shadowrun 3", "3rd edition", "sr3", "third edition"]
        }

        for edition, patterns in edition_patterns.items():
            if any(pattern in combined_text for pattern in patterns):
                return edition

        return "unknown"

    def _detect_content_type(self, text: str) -> str:
        """Detect the type of content (rule, example, table, etc.)."""

        type_scores = {}

        for content_type, patterns in self.content_type_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    score += 1
            type_scores[content_type] = score

        if not type_scores or max(type_scores.values()) == 0:
            return "general"

        return max(type_scores, key=type_scores.get)

    def _calculate_section_scores(self, text: str) -> Dict[str, float]:
        """Updated scoring with regex pattern support."""

        text_lower = text
        scores = {}

        for section, patterns in self.section_patterns.items():
            # Count primary keywords (weight: 1.0)
            primary_score = 0
            for pattern in patterns["primary"]:
                if pattern.startswith(r'\b') and pattern.endswith(r'\b'):
                    # Handle regex patterns
                    import re
                    matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
                    primary_score += matches * 1.0
                else:
                    # Handle regular string patterns
                    if pattern in text_lower:
                        count = text_lower.count(pattern)
                        primary_score += count * 1.0

            # Count secondary keywords (weight: 0.5)
            secondary_score = 0
            for pattern in patterns["secondary"]:
                if pattern.startswith(r'\b') and pattern.endswith(r'\b'):
                    import re
                    matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
                    secondary_score += matches * 0.5
                else:
                    if pattern in text_lower:
                        count = text_lower.count(pattern)
                        secondary_score += count * 0.5

            # Apply exclusion penalties (weight: -0.8)
            exclusion_penalty = 0
            for pattern in patterns["exclude_if"]:
                if pattern in text_lower:
                    count = text_lower.count(pattern)
                    exclusion_penalty += count * 0.8

            # Calculate final score
            total_score = primary_score + secondary_score - exclusion_penalty

            # Normalize and ensure positive
            normalized_score = max(0.0, total_score / len(text_lower) * 1000)
            scores[section] = normalized_score

        return scores

    def _fallback_classification(self, text: str) -> str:
        """Better fallback when no patterns match."""

        # Look for any strong indicators
        if any(word in text for word in ["combat", "weapon", "damage", "armor"]):
            return "Combat"
        elif any(word in text for word in ["magic", "spell", "astral", "mage"]):
            return "Magic"
        elif any(word in text for word in ["matrix", "ic", "decker", "program"]):
            return "Matrix"
        elif any(word in text for word in ["test", "dice pool", "threshold"]):
            return "Skills"
        elif any(word in text for word in ["gear", "equipment", "cost", "availability"]):
            return "Gear"
        else:
            return "General"  # Instead of "Unknown"

    def _contains_tables(self, text: str) -> bool:
        """Check for your table format."""
        return text.count("|") > 10 and ("---" in text or "Free Actions" in text)

    def _is_rule_definition(self, text: str) -> bool:
        """Check for rule definitions in your format."""
        rule_indicators = [
            "dice pool", "test:", "threshold", "opposed by",
            "make a", "roll", "resistance"
        ]
        return any(indicator in text for indicator in rule_indicators)

    def _extract_mechanical_keywords(self, text: str) -> List[str]:
        """Extract game mechanics from your content."""

        keywords = []

        # Attributes
        attributes = ["body", "agility", "reaction", "strength", "charisma",
                      "intuition", "logic", "willpower", "edge", "magic", "resonance"]
        keywords.extend([attr for attr in attributes if attr in text])

        # Game mechanics
        mechanics = ["initiative", "dice pool", "threshold", "hits", "glitch",
                     "armor", "damage", "stun", "physical"]
        keywords.extend([mech for mech in mechanics if mech in text])

        return list(set(keywords))  # Remove duplicates

    def _extract_page_references(self, text: str) -> List[str]:
        """Extract page references from your link format."""

        # Match your format: [Heavy Weapons](#page-133-0)
        pattern = r'\[([^\]]+)\]\(#page-(\d+)-(\d+)\)'
        matches = re.findall(pattern, text)

        return [f"{name} (page {page})" for name, page, _ in matches]

    def _find_cross_references(self, text: str) -> List[str]:
        """Extract cross-references to other rules/pages."""
        references = []

        ref_patterns = [
            r"see page (\d+)", r"see chapter (\d+)",
            r"refer to ([A-Z][a-z ]+)", r"as described in ([A-Z][a-z ]+)"
        ]

        for pattern in ref_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            references.extend(matches)

        return references

    def _contains_dice_pools(self, text: str) -> bool:
        """Check if content contains dice pool definitions."""
        dice_pool_patterns = [
            r"dice pool.*[+\-]", r"roll.*\+.*attribute", r"body\s*\+\s*armor",
            r"test.*:.*\+", r"opposed.*:.*vs", r"resistance.*:"
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in dice_pool_patterns)


class ImprovedSemanticChunker:
    """Semantic-aware chunking that respects content boundaries."""

    def __init__(self, chunk_size: int = 800, overlap: int = 100):
        self.chunk_size = chunk_size  # Reduced from 1024 for better retrieval
        self.overlap = overlap
        self.classifier = ImprovedShadowrunClassifier()

    def chunk_document(self, text: str, source: str, count_tokens_fn) -> List[Dict]:
        """Create semantically aware chunks with improved metadata."""

        # Clean and normalize text
        text = self._clean_text(text)

        # Split by main headers first (# headers)
        main_sections = self._split_by_headers(text)

        if not main_sections:
            # Fallback to paragraph-based splitting
            logger.warning(f"No headers found in {source}, using paragraph splitting")
            return self._chunk_by_paragraphs(text, source, count_tokens_fn)

        all_chunks = []

        for section_idx, section in enumerate(main_sections):
            section_chunks = self._chunk_section_semantically(
                section, source, count_tokens_fn, section_idx
            )
            all_chunks.extend(section_chunks)

        # Add global sequential links
        self._add_global_links(all_chunks)

        logger.info(f"Created {len(all_chunks)} semantic chunks from {len(main_sections)} sections")
        return all_chunks

    def _clean_text(self, text: str) -> str:
        """Clean text while preserving structure."""

        # Remove excessive whitespace but preserve paragraph breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)

        # Clean up table formatting
        text = re.sub(r'\|\s*\n\s*\|', '|\n|', text)

        return text.strip()

    def _split_by_headers(self, text: str) -> List[Dict]:
        """Split by headers while preserving semantic units."""

        lines = text.split('\n')
        sections = []
        current_section = {"title": "Introduction", "content": [], "level": 1}

        for line in lines:
            # Check for headers (# or ##, but not ###+ to avoid over-splitting)
            header_match = re.match(r'^(#{1,2})\s+(.+)$', line.strip())

            if header_match:
                # Save previous section
                if current_section["content"]:
                    current_section["content"] = '\n'.join(current_section["content"]).strip()
                    if len(current_section["content"]) > 50:  # Skip tiny sections
                        sections.append(current_section)

                # Start new section
                level = len(header_match.group(1))
                title = header_match.group(2).strip()

                current_section = {
                    "title": title,
                    "content": [line],  # Include header in content
                    "level": level
                }
            else:
                current_section["content"].append(line)

        # Add final section
        if current_section["content"]:
            current_section["content"] = '\n'.join(current_section["content"]).strip()
            if len(current_section["content"]) > 50:
                sections.append(current_section)

        return sections

    def _chunk_section_semantically(self, section: Dict, source: str,
                                    count_tokens_fn, section_idx: int) -> List[Dict]:
        """Chunk a section while respecting semantic boundaries."""

        title = section["title"]
        content = section["content"]

        # Get base metadata for this section
        base_metadata = self.classifier.classify_content(content, source)
        base_metadata.update({
            "section_title": title,
            "section_index": section_idx,
            "section_id": self._generate_section_id(title)
        })

        token_count = count_tokens_fn(content)

        # If section fits in one chunk, return it as-is
        if token_count <= self.chunk_size:
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                "chunk_index": 0,
                "total_chunks_in_section": 1,
                "section_complete": True,
                "token_count": token_count
            })

            return [{
                "text": content,
                "source": source,
                "metadata": chunk_metadata
            }]

        # Split large section intelligently
        return self._split_large_section(content, base_metadata, source, count_tokens_fn)

    def _split_large_section(self, content: str, base_metadata: Dict,
                             source: str, count_tokens_fn) -> List[Dict]:
        """Split large sections at semantic boundaries."""

        chunks = []

        # Try to split at natural boundaries (paragraphs, tables, lists)
        segments = self._find_semantic_segments(content)

        current_chunk = []
        current_tokens = 0

        for segment in segments:
            segment_tokens = count_tokens_fn(segment)

            # If adding this segment would exceed chunk size, finalize current chunk
            if current_tokens + segment_tokens > self.chunk_size and current_chunk:
                chunk_text = '\n'.join(current_chunk)

                # Add overlap from next segment if possible
                if self.overlap > 0:
                    overlap_text = segment[:self.overlap] if len(segment) > self.overlap else segment
                    chunk_text += f"\n{overlap_text}"

                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    "chunk_index": len(chunks),
                    "token_count": count_tokens_fn(chunk_text),
                    "section_complete": False
                })

                chunks.append({
                    "text": chunk_text,
                    "source": source,
                    "metadata": chunk_metadata
                })

                # Start new chunk with overlap
                if self.overlap > 0 and current_chunk:
                    overlap_segments = current_chunk[-1:]  # Keep last segment as overlap
                    current_chunk = overlap_segments + [segment]
                    current_tokens = count_tokens_fn('\n'.join(current_chunk))
                else:
                    current_chunk = [segment]
                    current_tokens = segment_tokens
            else:
                current_chunk.append(segment)
                current_tokens += segment_tokens

        # Add final chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                "chunk_index": len(chunks),
                "token_count": count_tokens_fn(chunk_text),
                "section_complete": True
            })

            chunks.append({
                "text": chunk_text,
                "source": source,
                "metadata": chunk_metadata
            })

        # Update total chunks count for all chunks in this section
        for chunk in chunks:
            chunk["metadata"]["total_chunks_in_section"] = len(chunks)

        return chunks

    def _find_semantic_segments(self, content: str) -> List[str]:
        """Split content into semantic segments (paragraphs, tables, lists)."""

        segments = []
        current_segment = []

        lines = content.split('\n')
        in_table = False

        for line in lines:
            # Detect table boundaries
            if '|' in line and line.count('|') > 1:
                if not in_table:
                    # Starting a table - finalize current segment
                    if current_segment:
                        segments.append('\n'.join(current_segment).strip())
                        current_segment = []
                    in_table = True
                current_segment.append(line)

            elif in_table and line.strip() == '':
                # End of table
                if current_segment:
                    segments.append('\n'.join(current_segment).strip())
                    current_segment = []
                in_table = False

            elif line.strip() == '' and not in_table:
                # Paragraph break
                if current_segment:
                    segments.append('\n'.join(current_segment).strip())
                    current_segment = []

            else:
                current_segment.append(line)

        # Add final segment
        if current_segment:
            segments.append('\n'.join(current_segment).strip())

        return [seg for seg in segments if len(seg.strip()) > 10]

    def _chunk_by_paragraphs(self, text: str, source: str, count_tokens_fn) -> List[Dict]:
        """Fallback paragraph-based chunking."""

        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        base_metadata = self.classifier.classify_content(text, source)

        chunks = []
        current_chunk = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = count_tokens_fn(para)

            if current_tokens + para_tokens > self.chunk_size and current_chunk:
                # Finalize current chunk
                chunk_text = '\n\n'.join(current_chunk)
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    "chunk_index": len(chunks),
                    "token_count": current_tokens,
                    "section_complete": False
                })

                chunks.append({
                    "text": chunk_text,
                    "source": source,
                    "metadata": chunk_metadata
                })

                current_chunk = [para]
                current_tokens = para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens

        # Add final chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                "chunk_index": len(chunks),
                "token_count": current_tokens,
                "section_complete": True
            })

            chunks.append({
                "text": chunk_text,
                "source": source,
                "metadata": chunk_metadata
            })

        return chunks

    def _generate_section_id(self, title: str) -> str:
        """Generate clean section ID."""
        clean_title = re.sub(r'[^a-zA-Z0-9\s]', '', title)
        clean_title = re.sub(r'\s+', '_', clean_title.strip())
        return clean_title.lower()[:50]

    def _add_global_links(self, chunks: List[Dict]) -> None:
        """Add navigation links between chunks."""

        for i, chunk in enumerate(chunks):
            metadata = chunk['metadata']

            # Global navigation
            if i > 0:
                metadata['prev_chunk_global'] = f"chunk_{i - 1:03d}"
            if i < len(chunks) - 1:
                metadata['next_chunk_global'] = f"chunk_{i + 1:03d}"

            # Section-specific navigation
            section_id = metadata.get('section_id')
            if section_id:
                section_chunks = [
                    (j, c) for j, c in enumerate(chunks)
                    if c['metadata'].get('section_id') == section_id
                ]

                # Find position within section
                section_position = next(
                    (pos for pos, (chunk_idx, _) in enumerate(section_chunks) if chunk_idx == i),
                    None
                )

                if section_position is not None:
                    if section_position > 0:
                        prev_idx = section_chunks[section_position - 1][0]
                        metadata['prev_chunk_section'] = f"chunk_{prev_idx:03d}"

                    if section_position < len(section_chunks) - 1:
                        next_idx = section_chunks[section_position + 1][0]
                        metadata['next_chunk_section'] = f"chunk_{next_idx:03d}"

            # Unique chunk ID
            metadata['chunk_id'] = f"chunk_{i:03d}"


# Integration functions to replace existing indexer methods
def replace_extract_shadowrun_metadata(content: str, source: str) -> Dict:
    """Replacement for _extract_shadowrun_metadata in indexer.py"""
    classifier = ImprovedShadowrunClassifier()
    return classifier.classify_content(content, source)


def replace_chunk_text_semantic(text: str, source: str, count_tokens_fn) -> List[Dict]:
    """Replacement for _chunk_text_semantic in indexer.py"""
    chunker = ImprovedSemanticChunker(chunk_size=800, overlap=100)  # Smaller chunks
    return chunker.chunk_document(text, source, count_tokens_fn)