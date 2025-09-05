"""
Clean semantic chunker that uses existing regex cleaner and unified classification.
Replaces tools/improved_classifier.py completely.
"""

import re
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Callable

logger = logging.getLogger(__name__)


class CleanSemanticChunker:
    """Semantic chunker using existing regex cleaner and unified LLM classification."""

    def __init__(self, chunk_size: int = 800, overlap: int = 150):
        self.chunk_size = chunk_size
        self.overlap = overlap

        # Import existing components
        from tools.llm_classifier import create_llm_classifier
        from tools.regex_text_cleaner import create_regex_cleaner

        self.classifier = create_llm_classifier(model_name="qwen2.5:14b-instruct-q6_K")
        self.regex_cleaner = create_regex_cleaner()

        logger.info(f"Clean semantic chunker initialized: {chunk_size} tokens, {overlap} overlap")

    def chunk_document(self, text: str, source: str, count_tokens_fn: Callable) -> List[Dict]:
        """Create semantically aware chunks using existing regex cleaner."""

        # Use existing regex cleaner
        cleaned_text = self.regex_cleaner.clean_text(text)

        if not cleaned_text.strip():
            logger.warning(f"No content after cleaning: {source}")
            return []

        # Split by headers to preserve document structure
        sections = self._split_by_headers(cleaned_text)

        if not sections:
            # No headers - treat as single section
            sections = [{"title": "Content", "content": cleaned_text, "level": 1}]

        all_chunks = []

        for section_idx, section in enumerate(sections):
            section_chunks = self._chunk_section_with_overlap(
                section, source, count_tokens_fn, section_idx
            )
            all_chunks.extend(section_chunks)

        # Add sequential links for context preservation
        self._add_sequential_links(all_chunks)

        logger.info(f"Created {len(all_chunks)} chunks from {len(sections)} sections in {source}")
        return all_chunks

    def _split_by_headers(self, text: str) -> List[Dict]:
        """Split text by markdown headers while preserving semantic structure."""

        # Look for markdown headers (# ## ###)
        header_pattern = r'^(#{1,3})\s+(.+?)$'
        lines = text.split('\n')

        sections = []
        current_section = {"title": "", "content": "", "level": 1}

        for line in lines:
            match = re.match(header_pattern, line.strip())

            if match:
                # Save previous section if it has content
                if current_section["content"].strip():
                    sections.append(current_section)

                # Start new section
                level = len(match.group(1))
                title = match.group(2).strip()

                current_section = {
                    "title": title,
                    "content": "",
                    "level": level
                }
            else:
                # Add line to current section
                current_section["content"] += line + "\n"

        # Add final section
        if current_section["content"].strip():
            sections.append(current_section)

        if sections:
            logger.info(f"Split into {len(sections)} sections by headers")

        return sections

    def _chunk_section_with_overlap(self, section: Dict, source: str,
                                    count_tokens_fn: Callable, section_idx: int) -> List[Dict]:
        """Chunk a section with consistent overlap and unified classification."""

        content = section["content"]
        title = section["title"]

        # In _chunk_section_with_overlap, add logging:
        logger.info(f"Section '{title}' will create approximately {len(content) // self.chunk_size} chunks")

        if not content.strip():
            return []

        chunks = []
        current_pos = 0
        chunk_idx = 0

        while current_pos < len(content):
            # Extract chunk with overlap consideration
            chunk_end = min(current_pos + self.chunk_size, len(content))

            # Try to end at sentence boundary
            chunk_text = content[current_pos:chunk_end]

            # Look for good break point (sentence end)
            if chunk_end < len(content):
                # Find last sentence end in chunk
                sentence_ends = [m.end() for m in re.finditer(r'[.!?]\s+', chunk_text)]
                if sentence_ends and len(chunk_text) - sentence_ends[-1] > 50:
                    # Use sentence boundary if it's not too close to the end
                    chunk_end = current_pos + sentence_ends[-1]
                    chunk_text = content[current_pos:chunk_end]

            # Skip very short chunks (unless it's the last one)
            if len(chunk_text.strip()) < 100 and chunk_end < len(content):
                current_pos += 200  # Skip forward
                continue

            # Create chunk with metadata
            chunk_id = f"{source}_section_{section_idx}_chunk_{chunk_idx}"

            # Classify chunk using unified classifier
            classification = self.classifier.classify_content(chunk_text, source)
            logger.info(f"Classification result: {classification}")


            # Build chunk metadata
            chunk_data = {
                "id": chunk_id,
                "text": chunk_text.strip(),
                "source": source,
                # Classification metadata from unified classifier
                "metadata": {
                    "sections": [classification.get("primary_section", "Unknown")],  # In metadata
                    "next_chunk_global": None,  # In metadata
                    "prev_chunk_global": None,  # In metadata
                    "section_title": title,
                    "chunk_index": chunk_idx,
                    "char_start": current_pos,
                    "char_end": chunk_end,
                    "token_count": count_tokens_fn(chunk_text) if count_tokens_fn else len(chunk_text.split()),
                    "section_index": section_idx,
                    "primary_section": classification.get("primary_section", "Unknown"),
                    "content_type": classification.get("content_type", "unknown"),
                    "contains_rules": classification.get("contains_rules", False),
                    "is_rule_definition": classification.get("is_rule_definition", False),
                    "contains_dice_pools": classification.get("contains_dice_pools", False),
                    "contains_examples": classification.get("contains_examples", False),
                    "confidence": classification.get("confidence", 0.5),
                    "mechanical_keywords": classification.get("mechanical_keywords", []),
                    "specific_topics": classification.get("specific_topics", []),
                    "classification_method": classification.get("classification_method", "unknown"),
                    # Document metadata
                    "document_type": classification.get("document_type", "rulebook"),
                    "edition": classification.get("edition", "SR5")
                }
            }

            chunks.append(chunk_data)

            # Move forward with overlap
            if chunk_end >= len(content):
                break

            old_pos = current_pos
            overlap_start = max(0, chunk_end - self.overlap)
            current_pos = overlap_start
            logger.info(f"Position update: {old_pos} -> {current_pos}")

            # Safety check to prevent infinite loops
            if current_pos <= old_pos and chunk_end < len(content):
                logger.error(f"Chunker stuck! Position not advancing: {old_pos} -> {current_pos}")
                current_pos = old_pos + 200  # Force advancement

            chunk_idx += 1

        logger.debug(f"Section '{title}' created {len(chunks)} chunks")
        return chunks

    def _add_sequential_links(self, chunks: List[Dict]) -> None:
        """Add sequential linking metadata for better context retrieval."""

        for i, chunk in enumerate(chunks):
            if i > 0:
                chunk["metadata"]["prev_chunk_global"] = chunks[i - 1]["id"]
            if i < len(chunks) - 1:
                chunk["metadata"]["next_chunk_global"] = chunks[i + 1]["id"]

            # Add section-level links
            same_section_chunks = [
                c["id"] for c in chunks
                if c.get("metadata", {}).get("section_index") == chunk.get("metadata", {}).get("section_index")
                   and c["id"] != chunk["id"]
            ]
            chunk["metadata"]["section_chunk_ids"] = same_section_chunks[:5]

    def get_chunk_statistics(self, chunks: List[Dict]) -> Dict:
        """Generate statistics about chunking quality for monitoring."""

        if not chunks:
            return {"total_chunks": 0}

        token_counts = [c["token_count"] for c in chunks]
        classifications = [c["primary_section"] for c in chunks]
        content_types = [c["content_type"] for c in chunks]

        from collections import Counter

        stats = {
            "total_chunks": len(chunks),
            "avg_token_count": sum(token_counts) / len(token_counts),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "section_distribution": dict(Counter(classifications)),
            "content_type_distribution": dict(Counter(content_types)),
            "rule_chunks": sum(1 for c in chunks if c["contains_rules"]),
            "example_chunks": sum(1 for c in chunks if c["contains_examples"]),
            "dice_pool_chunks": sum(1 for c in chunks if c["contains_dice_pools"]),
            "matrix_chunks": sum(1 for c in chunks if c["primary_section"] == "Matrix"),
            "combat_chunks": sum(1 for c in chunks if c["primary_section"] == "Combat")
        }

        return stats


# Factory function for indexer.py integration
def create_semantic_chunker(chunk_size: int = 800, overlap: int = 150) -> CleanSemanticChunker:
    """Factory function for indexer integration."""
    return CleanSemanticChunker(chunk_size=chunk_size, overlap=overlap)