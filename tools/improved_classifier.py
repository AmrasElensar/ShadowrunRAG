#!/usr/bin/env python3
"""
Cleaned semantic chunker that uses LLM-based classification.
Replaces tools/improved_classifier.py with only the chunking logic.
"""

import re
import logging
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)


class ImprovedSemanticChunker:
    """Semantic-aware chunking that uses LLM classification."""

    def __init__(self, chunk_size: int = 800, overlap: int = 100):
        self.chunk_size = chunk_size  # Reduced from 1024 for better retrieval
        self.overlap = overlap
        # Use LLM classifier instead of pattern-based
        from tools.llm_classifier import create_llm_classifier
        self.classifier = create_llm_classifier(model_name="phi4-mini")

    def chunk_document(self, text: str, source: str, count_tokens_fn) -> List[Dict]:
        """Create semantically aware chunks with LLM-based metadata."""

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

        # Get base metadata for this section using LLM classifier
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


# Integration function for indexer.py
def replace_chunk_text_semantic(text: str, source: str, count_tokens_fn) -> List[Dict]:
    """Replacement for _chunk_text_semantic in indexer.py"""
    chunker = ImprovedSemanticChunker(chunk_size=800, overlap=100)
    return chunker.chunk_document(text, source, count_tokens_fn)