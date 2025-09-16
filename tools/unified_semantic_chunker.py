"""
Unified semantic chunker that fixes mid-sentence and mid-table splits.
Consolidates all chunking logic into one clean implementation.
"""

import re
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Callable

logger = logging.getLogger(__name__)


class UnifiedSemanticChunker:
    """Enhanced semantic chunker with intelligent boundary detection."""

    def __init__(self, chunk_size: int = 800, overlap: int = 150):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = 200  # Minimum viable chunk size
        self.max_chunk_size = 1200  # Maximum before forced split

        # Import existing components
        from tools.regex_text_cleaner import create_regex_cleaner
        from tools.unified_classifier import create_unified_classifier

        self.regex_cleaner = create_regex_cleaner()
        self.classifier = create_unified_classifier()

        logger.info(f"Unified semantic chunker initialized: {chunk_size} tokens, {overlap} overlap")

    def chunk_document(self, text: str, source: str, count_tokens_fn: Callable) -> List[Dict]:
        """Create semantically aware chunks with improved boundaries."""

        # Use existing regex cleaner
        cleaned_text = self.regex_cleaner.clean_text(text)

        if not cleaned_text.strip():
            logger.warning(f"No content after cleaning: {source}")
            return []

        # First pass: Split by major semantic boundaries
        sections = self._split_by_semantic_boundaries(cleaned_text)

        if not sections:
            sections = [{"title": "Content", "content": cleaned_text, "level": 1, "start_pos": 0}]

        all_chunks = []

        for section_idx, section in enumerate(sections):
            section_chunks = self._chunk_section_with_smart_boundaries(
                section, source, count_tokens_fn, section_idx
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
            r'^\d+\.\s+[A-Z].+$'
        ]

        # Combine all patterns
        combined_pattern = '|'.join(f'({pattern})' for pattern in boundary_patterns)

        lines = text.split('\n')
        current_section = []
        current_title = "Content"
        current_level = 1

        for line_idx, line in enumerate(lines):
            line_stripped = line.strip()

            # Check if this line is a boundary
            boundary_match = re.match(combined_pattern, line_stripped, re.MULTILINE)

            if boundary_match and current_section:
                # Save previous section
                section_content = '\n'.join(current_section).strip()
                if section_content:
                    sections.append({
                        "title": current_title,
                        "content": section_content,
                        "level": current_level,
                        "start_pos": current_pos,
                        "line_start": line_idx - len(current_section),
                        "line_end": line_idx - 1
                    })

                # Start new section
                current_section = [line]
                current_title = line_stripped
                current_level = self._determine_header_level(line_stripped)
                current_pos = len('\n'.join(lines[:line_idx]))

            else:
                current_section.append(line)

        # Add final section
        if current_section:
            section_content = '\n'.join(current_section).strip()
            if section_content:
                sections.append({
                    "title": current_title,
                    "content": section_content,
                    "level": current_level,
                    "start_pos": current_pos,
                    "line_start": len(lines) - len(current_section),
                    "line_end": len(lines) - 1
                })

        return sections

    def _determine_header_level(self, line: str) -> int:
        """Determine the level of a header."""

        # Markdown headers
        if line.startswith('#'):
            return len(line) - len(line.lstrip('#'))

        # Table headers
        if '|' in line and any(term in line.upper() for term in ['WEAPON', 'FIREARM', 'GEAR']):
            return 2

        # All caps sections
        if line.isupper() and len(line) > 10:
            return 1

        # Numbered sections
        if re.match(r'^\d+\.', line):
            return 2

        return 3

    def _chunk_section_with_smart_boundaries(self, section: Dict, source: str,
                                             count_tokens_fn: Callable, section_idx: int) -> List[Dict]:
        """Chunk a section while respecting semantic boundaries."""

        content = section["content"]
        section_title = section["title"]

        # Check if section is small enough to keep as one chunk
        token_count = count_tokens_fn(content)
        if token_count <= self.chunk_size:
            chunk_id = self._generate_chunk_id(content, source, 0)
            classification = self.classifier.classify_content(content, source)

            return [{
                "id": chunk_id,
                "text": content,
                "source": source,
                "metadata": {
                    **classification,
                    "section_title": section_title,
                    "section_index": section_idx,
                    "chunk_index": 0,
                    "token_count": token_count,
                    "boundary_quality": "excellent",
                    "chunk_type": self._determine_chunk_type(content),
                    "is_complete_section": True,
                    "is_table_content": self._is_table_content(content)
                }
            }]

        # Section is too large - need to split with smart boundaries
        return self._split_large_section(section, source, count_tokens_fn, section_idx)

    def _split_large_section(self, section: Dict, source: str,
                             count_tokens_fn: Callable, section_idx: int) -> List[Dict]:
        """Split large section using intelligent boundary detection."""

        content = section["content"]
        section_title = section["title"]

        # Try different splitting strategies in order of preference

        # Strategy 1: Split by tables (highest priority for Shadowrun content)
        if self._contains_tables(content):
            chunks = self._split_by_tables(content, source, count_tokens_fn, section_idx, section_title)
            if chunks:
                return chunks

        # Strategy 2: Split by paragraphs
        chunks = self._split_by_paragraphs(content, source, count_tokens_fn, section_idx, section_title)
        if chunks:
            return chunks

        # Strategy 3: Split by sentences (fallback)
        return self._split_by_sentences(content, source, count_tokens_fn, section_idx, section_title)

    def _contains_tables(self, text: str) -> bool:
        """Check if text contains table structures."""

        # Check for markdown tables
        if text.count('|') >= 6 and text.count('\n|') >= 2:
            return True

        # Check for weapon stat patterns
        if re.search(r'(?:ACC|Damage|AP|Mode|RC|Ammo|Avail|Cost)', text, re.IGNORECASE):
            return True

        return False

    def _split_by_tables(self, content: str, source: str, count_tokens_fn: Callable,
                         section_idx: int, section_title: str) -> List[Dict]:
        """Split content by table boundaries."""

        chunks = []
        current_chunk = []
        current_tokens = 0
        in_table = False
        table_header = None

        lines = content.split('\n')

        for line in lines:
            line_tokens = count_tokens_fn(line)

            # Detect table start
            if '|' in line and not in_table:
                # Save current chunk if it exists
                if current_chunk and current_tokens > self.min_chunk_size:
                    chunk_text = '\n'.join(current_chunk).strip()
                    if chunk_text:
                        chunks.append(self._create_chunk(
                            chunk_text, source, len(chunks), section_idx,
                            section_title, current_tokens, "paragraph"
                        ))
                    current_chunk = []
                    current_tokens = 0

                in_table = True
                table_header = line.strip()
                current_chunk.append(line)
                current_tokens += line_tokens

            # Detect table end
            elif in_table and '|' not in line.strip() and line.strip():
                in_table = False
                table_header = None

                # Save table chunk
                chunk_text = '\n'.join(current_chunk).strip()
                if chunk_text:
                    chunks.append(self._create_chunk(
                        chunk_text, source, len(chunks), section_idx,
                        section_title, current_tokens, "table", table_header
                    ))
                current_chunk = [line]
                current_tokens = line_tokens

            else:
                current_chunk.append(line)
                current_tokens += line_tokens

                # Check if chunk is getting too large
                if current_tokens > self.chunk_size and not in_table:
                    # Find good break point
                    break_point = self._find_paragraph_break(current_chunk)
                    if break_point > 0:
                        chunk_lines = current_chunk[:break_point]
                        chunk_text = '\n'.join(chunk_lines).strip()
                        if chunk_text:
                            chunk_tokens = count_tokens_fn(chunk_text)
                            chunks.append(self._create_chunk(
                                chunk_text, source, len(chunks), section_idx,
                                section_title, chunk_tokens, "paragraph"
                            ))

                        # Start new chunk with overlap
                        overlap_start = max(0, break_point - 2)
                        current_chunk = current_chunk[overlap_start:]
                        current_tokens = count_tokens_fn('\n'.join(current_chunk))

        # Save final chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk).strip()
            if chunk_text:
                chunks.append(self._create_chunk(
                    chunk_text, source, len(chunks), section_idx,
                    section_title, current_tokens,
                    "table" if in_table else "paragraph", table_header
                ))

        return chunks

    def _split_by_paragraphs(self, content: str, source: str, count_tokens_fn: Callable,
                             section_idx: int, section_title: str) -> List[Dict]:
        """Split content by paragraph boundaries."""

        paragraphs = re.split(r'\n\s*\n', content)
        chunks = []
        current_chunk = []
        current_tokens = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = count_tokens_fn(para)

            # If adding this paragraph would exceed chunk size
            if current_tokens + para_tokens > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = '\n\n'.join(current_chunk).strip()
                if chunk_text:
                    chunks.append(self._create_chunk(
                        chunk_text, source, len(chunks), section_idx,
                        section_title, current_tokens, "paragraph"
                    ))

                # Start new chunk with overlap
                if len(current_chunk) > 1:
                    current_chunk = [current_chunk[-1], para]  # Keep last paragraph for context
                    current_tokens = count_tokens_fn(current_chunk[-2]) + para_tokens
                else:
                    current_chunk = [para]
                    current_tokens = para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens

        # Save final chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk).strip()
            if chunk_text:
                chunks.append(self._create_chunk(
                    chunk_text, source, len(chunks), section_idx,
                    section_title, current_tokens, "paragraph"
                ))

        return chunks

    def _split_by_sentences(self, content: str, source: str, count_tokens_fn: Callable,
                            section_idx: int, section_title: str) -> List[Dict]:
        """Split content by sentence boundaries (fallback method)."""

        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', content)
        chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_tokens = count_tokens_fn(sentence)

            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk).strip()
                if chunk_text:
                    chunks.append(self._create_chunk(
                        chunk_text, source, len(chunks), section_idx,
                        section_title, current_tokens, "sentence"
                    ))

                # Start new chunk with overlap
                if len(current_chunk) > 2:
                    current_chunk = current_chunk[-2:] + [sentence]  # Keep last 2 sentences
                    current_tokens = count_tokens_fn(' '.join(current_chunk))
                else:
                    current_chunk = [sentence]
                    current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens

        # Save final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk).strip()
            if chunk_text:
                chunks.append(self._create_chunk(
                    chunk_text, source, len(chunks), section_idx,
                    section_title, current_tokens, "sentence"
                ))

        return chunks

    def _create_chunk(self, text: str, source: str, chunk_idx: int, section_idx: int,
                      section_title: str, token_count: int, chunk_type: str,
                      table_header: str = None) -> Dict:
        """Create a chunk with comprehensive metadata."""

        chunk_id = self._generate_chunk_id(text, source, chunk_idx)
        classification = self.classifier.classify_content(text, source)

        return {
            "id": chunk_id,
            "text": text,
            "source": source,
            "metadata": {
                **classification,
                "section_title": section_title,
                "section_index": section_idx,
                "chunk_index": chunk_idx,
                "token_count": token_count,
                "boundary_quality": self._assess_boundary_quality(text),
                "chunk_type": chunk_type,
                "is_complete_section": False,
                "is_table_content": self._is_table_content(text),
                "table_header": table_header,
                "starts_mid_sentence": self._starts_mid_sentence(text),
                "ends_mid_sentence": self._ends_mid_sentence(text)
            }
        }

    def _generate_chunk_id(self, text: str, source: str, chunk_idx: int) -> str:
        """Generate unique chunk ID."""
        content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        source_name = Path(source).stem
        return f"{source_name}_{chunk_idx}_{content_hash}"

    def _find_paragraph_break(self, lines: List[str]) -> int:
        """Find the best paragraph break point."""

        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip() == "" or (lines[i - 1].strip() == "" and lines[i].strip()):
                return i

        return len(lines) // 2  # Fallback to middle

    def _assess_boundary_quality(self, text: str) -> str:
        """Assess the quality of chunk boundaries."""

        # Check if chunk starts/ends mid-sentence
        starts_mid_sentence = self._starts_mid_sentence(text)
        ends_mid_sentence = self._ends_mid_sentence(text)

        if not starts_mid_sentence and not ends_mid_sentence:
            return "excellent"
        elif not starts_mid_sentence or not ends_mid_sentence:
            return "good"
        else:
            return "poor"

    def _starts_mid_sentence(self, text: str) -> bool:
        """Check if text starts mid-sentence."""
        first_char = text.strip()[0] if text.strip() else ""
        return first_char.islower()

    def _ends_mid_sentence(self, text: str) -> bool:
        """Check if text ends mid-sentence."""
        last_char = text.rstrip()[-1] if text.rstrip() else ""
        return last_char not in '.!?'

    def _determine_chunk_type(self, text: str) -> str:
        """Determine the type of content in the chunk."""

        text_lower = text.lower()

        # Table detection
        if self._is_table_content(text):
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

    def _is_table_content(self, text: str) -> bool:
        """Check if content is table-related."""

        # Check for markdown tables
        if text.count('|') > 5 and text.count('\n|') > 1:
            return True

        # Check for weapon statistics patterns
        weapon_stats = ['acc', 'damage', 'ap', 'mode', 'rc', 'ammo', 'avail', 'cost']
        if sum(1 for stat in weapon_stats if stat in text.lower()) >= 3:
            return True

        return False

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
        token_counts = [chunk.get("metadata", {}).get("token_count", 0) for chunk in chunks]
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
            "token_count_stats": {
                "min": min(token_counts) if token_counts else 0,
                "max": max(token_counts) if token_counts else 0,
                "avg": sum(token_counts) / len(token_counts) if token_counts else 0
            },
            "boundary_quality": {
                "excellent": excellent_boundaries,
                "good": good_boundaries,
                "poor": poor_boundaries,
                "quality_score": (excellent_boundaries * 3 + good_boundaries * 2 + poor_boundaries) / (
                        len(chunks) * 3) * 100 if chunks else 0
            },
            "chunk_types": type_distribution,
            "table_chunks": sum(1 for chunk in chunks if chunk.get("metadata", {}).get("is_table_content")),
            "poor_boundary_chunks": sum(1 for chunk in chunks
                                        if chunk.get("metadata", {}).get("starts_mid_sentence") or
                                        chunk.get("metadata", {}).get("ends_mid_sentence")),
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


# Factory functions for integration
def create_unified_semantic_chunker(chunk_size: int = 800, overlap: int = 150) -> UnifiedSemanticChunker:
    """Create unified semantic chunker instance."""
    return UnifiedSemanticChunker(chunk_size=chunk_size, overlap=overlap)


def create_improved_semantic_chunker(chunk_size: int = 800, overlap: int = 150) -> UnifiedSemanticChunker:
    """Backward compatibility function."""
    return create_unified_semantic_chunker(chunk_size, overlap)


def create_semantic_chunker(chunk_size: int = 800, overlap: int = 150) -> UnifiedSemanticChunker:
    """Backward compatibility function."""
    return create_unified_semantic_chunker(chunk_size, overlap)