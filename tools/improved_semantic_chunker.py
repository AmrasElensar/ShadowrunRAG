"""
Improved semantic chunking system that fixes boundary issues.
Addresses mid-sentence and mid-table splits we saw in the original query.
"""

import re
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Callable

logger = logging.getLogger(__name__)


class ImprovedSemanticChunker:
    """Enhanced semantic chunker with better boundary detection."""

    def __init__(self, chunk_size: int = 800, overlap: int = 150):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = 200  # Minimum viable chunk size
        self.max_chunk_size = 1200  # Maximum before forced split

        # Import existing components
        from tools.regex_text_cleaner import create_regex_cleaner
        self.regex_cleaner = create_regex_cleaner()

        logger.info(f"Improved semantic chunker initialized: {chunk_size} tokens, {overlap} overlap")

    def chunk_document(self, text: str, source: str, count_tokens_fn: Callable,
                       classifier) -> List[Dict]:
        """Create semantically aware chunks with improved boundaries."""

        # Use existing regex cleaner
        cleaned_text = self.regex_cleaner.clean_text(text)

        if not cleaned_text.strip():
            logger.warning(f"No content after cleaning: {source}")
            return []

        # First pass: Split by major sections (headers)
        sections = self._split_by_semantic_boundaries(cleaned_text)

        if not sections:
            sections = [{"title": "Content", "content": cleaned_text, "level": 1, "start_pos": 0}]

        all_chunks = []

        for section_idx, section in enumerate(sections):
            section_chunks = self._chunk_section_with_smart_boundaries(
                section, source, count_tokens_fn, classifier, section_idx
            )
            all_chunks.extend(section_chunks)

        # Add sequential links and improve metadata
        self._add_enhanced_links(all_chunks)

        logger.info(f"Created {len(all_chunks)} chunks from {len(sections)} sections in {source}")
        return all_chunks

    def _split_by_semantic_boundaries(self, text: str) -> List[Dict]:
        """Split text by semantic boundaries (headers, tables, lists)."""

        sections = []
        current_pos = 0

        # Enhanced header pattern that captures more structures
        boundary_patterns = [
            # Markdown headers
            r'^(#{1,6})\s+(.+?)$',
            # Table headers (weapon tables, etc.)
            r'^\|\s*(?:WEAPON|FIREARM|GEAR|ITEM|NAME)\s*\|.*\|$',
            # Major section dividers
            r'^[A-Z][A-Z\s&]{10,}$',
            # Numbered sections
            r'^\d+\.\s+[A-Z].*$'
        ]

        # Find all boundaries
        boundaries = []
        for pattern in boundary_patterns:
            for match in re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE):
                boundaries.append({
                    'pos': match.start(),
                    'end': match.end(),
                    'title': match.group().strip(),
                    'level': self._determine_section_level(match.group())
                })

        # Sort boundaries by position
        boundaries.sort(key=lambda x: x['pos'])

        # Create sections
        for i, boundary in enumerate(boundaries):
            start_pos = boundary['pos']
            end_pos = boundaries[i + 1]['pos'] if i + 1 < len(boundaries) else len(text)

            content = text[start_pos:end_pos].strip()
            if content and len(content) > 50:  # Skip very short sections
                sections.append({
                    'title': boundary['title'],
                    'content': content,
                    'level': boundary['level'],
                    'start_pos': start_pos,
                    'end_pos': end_pos
                })

        return sections

    def _determine_section_level(self, header_text: str) -> int:
        """Determine the hierarchical level of a section."""

        # Markdown headers
        if header_text.startswith('#'):
            return header_text.count('#')

        # Table headers are usually level 3
        if '|' in header_text and any(word in header_text.upper() for word in ['WEAPON', 'GEAR', 'ITEM']):
            return 3

        # All caps sections are usually level 2
        if header_text.isupper() and len(header_text) > 10:
            return 2

        # Numbered sections
        if re.match(r'^\d+\.', header_text):
            return 2

        return 3  # Default level

    def _chunk_section_with_smart_boundaries(self, section: Dict, source: str,
                                             count_tokens_fn: Callable, classifier,
                                             section_idx: int) -> List[Dict]:
        """Chunk section content with intelligent boundary detection."""

        title = section["title"]
        content = section["content"]
        chunks = []

        if not content.strip():
            return chunks

        # Check if this is a table section - handle differently
        if self._is_table_section(content):
            return self._chunk_table_content(section, source, count_tokens_fn, classifier, section_idx)

        # Regular content chunking with smart boundaries
        current_pos = 0
        chunk_idx = 0

        while current_pos < len(content):

            # Determine ideal chunk end position
            ideal_end = current_pos + self.chunk_size

            if ideal_end >= len(content):
                # Last chunk - take everything
                chunk_end = len(content)
                chunk_text = content[current_pos:chunk_end]
            else:
                # Find the best boundary near ideal end position
                chunk_end, chunk_text = self._find_optimal_boundary(
                    content, current_pos, ideal_end
                )

            # Skip very short chunks unless it's the last one
            if len(chunk_text.strip()) < self.min_chunk_size and chunk_end < len(content):
                current_pos = min(current_pos + self.min_chunk_size, len(content))
                continue

            # Create chunk with enhanced metadata
            chunk_id = f"{source}_section_{section_idx}_chunk_{chunk_idx}"

            # Classify chunk
            classification = classifier.classify_content(chunk_text, source)

            chunk_data = {
                "id": chunk_id,
                "text": chunk_text.strip(),
                "source": source,
                "metadata": {
                    **classification,  # Include all classification metadata
                    "section_title": title,
                    "chunk_index": chunk_idx,
                    "char_start": current_pos,
                    "char_end": chunk_end,
                    "token_count": count_tokens_fn(chunk_text) if count_tokens_fn else len(chunk_text.split()),
                    "section_index": section_idx,
                    "is_table_content": False,
                    "boundary_quality": self._assess_boundary_quality(chunk_text),
                    "chunk_type": self._determine_chunk_type(chunk_text)
                }
            }

            chunks.append(chunk_data)

            # Calculate next position with smart overlap
            if chunk_end >= len(content):
                break

            # Smart overlap calculation
            overlap_start = max(0, chunk_end - self.overlap)
            next_pos = self._find_good_overlap_start(content, overlap_start, current_pos)

            current_pos = next_pos
            chunk_idx += 1

        logger.debug(f"Section '{title}' created {len(chunks)} chunks with smart boundaries")
        return chunks

    def _is_table_section(self, content: str) -> bool:
        """Detect if content is primarily a table."""

        lines = content.split('\n')
        table_indicators = 0

        for line in lines[:10]:  # Check first 10 lines
            if '|' in line and line.count('|') >= 3:
                table_indicators += 1

        # If more than 30% of lines look like table rows
        return table_indicators / min(len(lines), 10) > 0.3

    def _chunk_table_content(self, section: Dict, source: str, count_tokens_fn: Callable,
                             classifier, section_idx: int) -> List[Dict]:
        """Special handling for table content to avoid splitting mid-row."""

        content = section["content"]
        title = section["title"]

        # Split by table rows, keeping header with each chunk
        lines = content.split('\n')
        table_header = None
        chunks = []

        # Find table header
        for i, line in enumerate(lines):
            if '|' in line and any(word in line.upper() for word in ['WEAPON', 'FIREARM', 'ACC', 'DAMAGE']):
                table_header = line
                break

        # Group table rows into chunks
        current_chunk_lines = []
        if table_header:
            current_chunk_lines.append(table_header)

        current_tokens = 0
        chunk_idx = 0

        for line in lines:
            line_tokens = count_tokens_fn(line) if count_tokens_fn else len(line.split())

            # Check if adding this line would exceed chunk size
            if current_tokens + line_tokens > self.chunk_size and len(current_chunk_lines) > 1:
                # Create chunk from current lines
                chunk_text = '\n'.join(current_chunk_lines)

                chunk_id = f"{source}_table_{section_idx}_chunk_{chunk_idx}"
                classification = classifier.classify_content(chunk_text, source)

                chunk_data = {
                    "id": chunk_id,
                    "text": chunk_text.strip(),
                    "source": source,
                    "metadata": {
                        **classification,
                        "section_title": title,
                        "chunk_index": chunk_idx,
                        "token_count": current_tokens,
                        "section_index": section_idx,
                        "is_table_content": True,
                        "table_header": table_header,
                        "chunk_type": "table"
                    }
                }

                chunks.append(chunk_data)

                # Start new chunk with header
                current_chunk_lines = [table_header] if table_header else []
                current_tokens = count_tokens_fn(table_header) if table_header and count_tokens_fn else 0
                chunk_idx += 1

            current_chunk_lines.append(line)
            current_tokens += line_tokens

        # Add final chunk if there's content
        if len(current_chunk_lines) > 1:
            chunk_text = '\n'.join(current_chunk_lines)
            chunk_id = f"{source}_table_{section_idx}_chunk_{chunk_idx}"
            classification = classifier.classify_content(chunk_text, source)

            chunk_data = {
                "id": chunk_id,
                "text": chunk_text.strip(),
                "source": source,
                "metadata": {
                    **classification,
                    "section_title": title,
                    "chunk_index": chunk_idx,
                    "token_count": current_tokens,
                    "section_index": section_idx,
                    "is_table_content": True,
                    "table_header": table_header,
                    "chunk_type": "table"
                }
            }

            chunks.append(chunk_data)

        return chunks

    def _find_optimal_boundary(self, content: str, start_pos: int, ideal_end: int) -> tuple[int, str]:
        """Find the best boundary position near the ideal end."""

        search_window = 150  # Characters to search around ideal position

        # Priority order for boundary types
        boundary_finders = [
            self._find_paragraph_boundary,
            self._find_sentence_boundary,
            self._find_clause_boundary,
            self._find_word_boundary
        ]

        for finder in boundary_finders:
            boundary_pos = finder(content, ideal_end, search_window)
            if boundary_pos != ideal_end:  # Found a better boundary
                chunk_text = content[start_pos:boundary_pos]
                return boundary_pos, chunk_text

        # Fallback to ideal position
        chunk_text = content[start_pos:ideal_end]
        return ideal_end, chunk_text

    def _find_paragraph_boundary(self, content: str, ideal_pos: int, window: int) -> int:
        """Find paragraph break near ideal position."""

        start_search = max(0, ideal_pos - window)
        end_search = min(len(content), ideal_pos + window)

        # Look for double newlines (paragraph breaks)
        search_text = content[start_search:end_search]

        # Find paragraph breaks
        para_pattern = r'\n\s*\n'
        matches = list(re.finditer(para_pattern, search_text))

        if matches:
            # Find the match closest to our ideal position
            best_match = min(matches, key=lambda m: abs((start_search + m.end()) - ideal_pos))
            return start_search + best_match.end()

        return ideal_pos

    def _find_sentence_boundary(self, content: str, ideal_pos: int, window: int) -> int:
        """Find sentence ending near ideal position."""

        start_search = max(0, ideal_pos - window)
        end_search = min(len(content), ideal_pos + window)

        search_text = content[start_search:end_search]

        # Look for sentence endings
        sentence_pattern = r'[.!?]\s+[A-Z]'
        matches = list(re.finditer(sentence_pattern, search_text))

        if matches:
            # Find closest match to ideal position
            best_match = min(matches, key=lambda m: abs((start_search + m.start() + 1) - ideal_pos))
            return start_search + best_match.start() + 1

        return ideal_pos

    def _find_clause_boundary(self, content: str, ideal_pos: int, window: int) -> int:
        """Find clause break (comma, semicolon) near ideal position."""

        start_search = max(0, ideal_pos - window)
        end_search = min(len(content), ideal_pos + window)

        search_text = content[start_search:end_search]

        # Look for clause boundaries
        clause_pattern = r'[,;]\s+'
        matches = list(re.finditer(clause_pattern, search_text))

        if matches:
            best_match = min(matches, key=lambda m: abs((start_search + m.end()) - ideal_pos))
            return start_search + best_match.end()

        return ideal_pos

    def _find_word_boundary(self, content: str, ideal_pos: int, window: int) -> int:
        """Find word boundary near ideal position."""

        start_search = max(0, ideal_pos - window // 2)
        end_search = min(len(content), ideal_pos + window // 2)

        search_text = content[start_search:end_search]

        # Look for word boundaries (spaces)
        word_pattern = r'\s+'
        matches = list(re.finditer(word_pattern, search_text))

        if matches:
            best_match = min(matches, key=lambda m: abs((start_search + m.end()) - ideal_pos))
            return start_search + best_match.end()

        return ideal_pos

    def _find_good_overlap_start(self, content: str, overlap_start: int, current_pos: int) -> int:
        """Find a good starting position for overlap that doesn't split words."""

        # Ensure we always move forward
        min_advance = 300  # Minimum characters to advance
        min_next_pos = current_pos + min_advance

        if overlap_start < min_next_pos:
            overlap_start = min_next_pos

        # Find word boundary near overlap start
        search_window = 50
        for i in range(overlap_start, min(len(content), overlap_start + search_window)):
            if content[i].isspace():
                return i + 1  # Start after the space

        # Fallback to overlap_start if no space found
        return min(overlap_start, len(content))

    def _assess_boundary_quality(self, chunk_text: str) -> str:
        """Assess the quality of chunk boundaries."""

        # Check if chunk starts/ends mid-sentence
        starts_mid_sentence = chunk_text[0].islower() if chunk_text else True
        ends_mid_sentence = not chunk_text.rstrip()[-1] in '.!?' if chunk_text else True

        if not starts_mid_sentence and not ends_mid_sentence:
            return "excellent"
        elif not starts_mid_sentence or not ends_mid_sentence:
            return "good"
        else:
            return "poor"

    def _determine_chunk_type(self, text: str) -> str:
        """Determine the type of content in the chunk."""

        text_lower = text.lower()

        # Table detection
        if '|' in text and text.count('|') > 5:
            return "table"

        # Rule detection
        if any(term in text_lower for term in ["dice pool", "test:", "modifier", "resistance"]):
            return "rule"

        # Example detection
        if any(term in text_lower for term in ["example", "for instance", "suppose"]):
            return "example"

        # List detection
        if text.count('\n-') > 2 or text.count('\n•') > 2:
            return "list"

        return "narrative"

    def _add_enhanced_links(self, chunks: List[Dict]) -> None:
        """Add enhanced sequential linking metadata."""

        for i, chunk in enumerate(chunks):
            metadata = chunk["metadata"]

            # Basic sequential links
            if i > 0:
                metadata["prev_chunk_global"] = chunks[i - 1]["id"]
            if i < len(chunks) - 1:
                metadata["next_chunk_global"] = chunks[i + 1]["id"]

            # Section-level links
            same_section_chunks = [
                c["id"] for c in chunks
                if c.get("metadata", {}).get("section_index") == metadata.get("section_index")
                   and c["id"] != chunk["id"]
            ]
            metadata["section_chunk_ids"] = same_section_chunks[:5]

            # Table continuation links (for table chunks)
            if metadata.get("is_table_content"):
                table_chunks = [
                    c["id"] for c in chunks
                    if c.get("metadata", {}).get("table_header") == metadata.get("table_header")
                       and c["id"] != chunk["id"]
                ]
                metadata["table_continuation_ids"] = table_chunks[:3]

    def get_chunk_statistics(self, chunks: List[Dict]) -> Dict:
        """Generate enhanced statistics about chunking quality."""

        if not chunks:
            return {"error": "No chunks provided"}

        # Basic stats
        word_counts = [chunk.get("metadata", {}).get("token_count", 0) for chunk in chunks]
        boundary_qualities = [chunk.get("metadata", {}).get("boundary_quality", "unknown") for chunk in chunks]
        chunk_types = [chunk.get("metadata", {}).get("chunk_type", "unknown") for chunk in chunks]

        # Quality assessment
        excellent_boundaries = boundary_qualities.count("excellent")
        good_boundaries = boundary_qualities.count("good")
        poor_boundaries = boundary_qualities.count("poor")

        # Type distribution
        from collections import Counter
        type_distribution = dict(Counter(chunk_types))

        return {
            "total_chunks": len(chunks),
            "word_count_stats": {
                "min": min(word_counts) if word_counts else 0,
                "max": max(word_counts) if word_counts else 0,
                "avg": sum(word_counts) / len(word_counts) if word_counts else 0
            },
            "boundary_quality": {
                "excellent": excellent_boundaries,
                "good": good_boundaries,
                "poor": poor_boundaries,
                "quality_score": (excellent_boundaries * 3 + good_boundaries * 2 + poor_boundaries) / (
                            len(chunks) * 3) * 100
            },
            "chunk_types": type_distribution,
            "table_chunks": sum(1 for chunk in chunks if chunk.get("metadata", {}).get("is_table_content")),
            "recommendation": self._generate_chunking_recommendation(chunks)
        }

    def _generate_chunking_recommendation(self, chunks: List[Dict]) -> str:
        """Generate recommendations based on chunk analysis."""

        poor_boundary_pct = sum(1 for chunk in chunks
                                if chunk.get("metadata", {}).get("boundary_quality") == "poor") / len(chunks) * 100

        if poor_boundary_pct > 20:
            return "HIGH PRIORITY: >20% chunks have poor boundaries - review chunking parameters"
        elif poor_boundary_pct > 10:
            return "MEDIUM PRIORITY: Some boundary issues detected - monitor chunking quality"
        else:
            return "GOOD: Chunk boundaries are well-formed"


# Factory function for integration
def create_improved_semantic_chunker(chunk_size: int = 800, overlap: int = 150) -> ImprovedSemanticChunker:
    """Create improved semantic chunker instance."""
    return ImprovedSemanticChunker(chunk_size=chunk_size, overlap=overlap)