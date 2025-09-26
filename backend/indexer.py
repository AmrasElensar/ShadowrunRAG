"""
Enhanced IncrementalIndexer with built-in entity extraction and ALL original functionality preserved.
This combines the table-aware chunking, detailed logging, and all features from indexer_old.py
with the new entity extraction capabilities.
"""

import hashlib
import json
from pathlib import Path
from typing import List, Dict, Tuple
import chromadb
from chromadb.config import Settings
import ollama
from tqdm import tqdm
import logging
import time
import re

# Consolidated imports using the new unified modules
from tools.unified_classifier import create_unified_classifier
from tools.enhanced_query_processor import create_enhanced_query_processor

# Entity extraction imports
from tools.entity_registry_builder import create_entity_registry_builder
from tools.registry_storage import create_entity_registry_storage

# Check for optional dependencies
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logging.warning("tiktoken not available - using word-based chunking")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedIncrementalIndexer:
    """Enhanced incremental indexer with built-in entity extraction and ALL original functionality."""

    def __init__(
        self,
        chroma_path: str = "data/chroma_db",
        collection_name: str = "shadowrun_docs",
        embedding_model: str = "nomic-embed-text",
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        use_semantic_splitting: bool = True,
        entity_db_path: str = "data/entity_registry.db"
    ):
        logger.info(f"Initializing Enhanced Table-Aware Indexer with entity extraction")
        logger.info(f"Embedding model: {embedding_model}, chunk_size={chunk_size} tokens")

        self.chroma_path = Path(chroma_path)
        self.chroma_path.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.chroma_path),
            settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_semantic_splitting = use_semantic_splitting

        # Track indexed files
        self.indexed_files_path = self.chroma_path / "indexed_files.json"
        self.indexed_files = self._load_index_metadata()

        # Set up tokenizer
        if TIKTOKEN_AVAILABLE:
            self.encoder = tiktoken.get_encoding("cl100k_base")
            self._count_tokens = lambda text: len(self.encoder.encode(text))
            logger.info("Using tiktoken for accurate token counting")
        else:
            self._count_tokens = lambda text: len(text.split())
            logger.info("Using word-based token approximation")

        # Initialize components (original functionality)
        self.classifier = create_unified_classifier()
        self.query_processor = create_enhanced_query_processor()

        # Initialize entity extraction components (new functionality)
        logger.info("Initializing entity extraction components...")
        self.entity_builder = create_entity_registry_builder()
        self.entity_storage = create_entity_registry_storage(entity_db_path)

        # Entity extraction statistics
        self.entity_stats = {
            "total_entities_extracted": 0,
            "weapons_extracted": 0,
            "spells_extracted": 0,
            "ic_programs_extracted": 0,
            "chunks_with_entities": 0
        }

        current_chunks = self.collection.count()
        registry_stats = self.entity_storage.get_registry_stats()
        logger.info(f"Indexer ready. Collection has {current_chunks} existing chunks")
        logger.info(f"Entity Registry contains: {registry_stats}")

    def _load_index_metadata(self) -> Dict:
        """Load metadata about previously indexed files."""
        if self.indexed_files_path.exists():
            return json.loads(self.indexed_files_path.read_text())
        return {}

    def _save_index_metadata(self):
        """Save metadata about indexed files."""
        self.indexed_files_path.write_text(json.dumps(self.indexed_files, indent=2))

    def _clean_metadata_for_chroma(self, metadata: Dict) -> Dict:
        """Clean metadata to ensure ChromaDB compatibility (no lists/dicts/None)."""
        cleaned = {}

        for key, value in metadata.items():
            if value is None:
                continue
            elif isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            elif isinstance(value, list):
                if value:
                    cleaned[key] = ", ".join(str(item) for item in value)
            elif isinstance(value, dict):
                if value:
                    cleaned[key] = json.dumps(value)
            else:
                str_value = str(value)
                if str_value and str_value != "None":
                    cleaned[key] = str_value

        return cleaned

    def _get_file_hash(self, file_path: Path) -> str:
        """Get hash of file content for change detection."""
        return hashlib.md5(file_path.read_bytes()).hexdigest()

    def _detect_table_boundaries(self, text: str) -> List[Tuple[int, int, str]]:
        """Detect table boundaries in text. Returns list of (start_line, end_line, table_type)."""

        lines = text.split('\n')
        tables = []

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Markdown table detection
            if '|' in line and line.count('|') >= 2:
                # Look for table header pattern
                if i + 1 < len(lines) and re.match(r'^[\s\|:\-]+$', lines[i + 1].strip()):
                    # Found markdown table header
                    start_line = i
                    table_type = "markdown_table"

                    # Find end of table
                    j = i + 2
                    while j < len(lines) and '|' in lines[j].strip() and lines[j].strip():
                        j += 1

                    end_line = j - 1
                    tables.append((start_line, end_line, table_type))
                    logger.info(f"      Found markdown table: lines {start_line}-{end_line}")
                    i = j
                    continue

            # Shadowrun weapon stats table detection (without markdown formatting)
            elif any(term in line.lower() for term in ['acc', 'damage', 'mode', 'ammo', 'avail', 'cost']):
                # Look for weapon stats pattern
                start_line = i
                table_type = "weapon_stats"

                # Find end of stats block
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()
                    if not next_line:
                        j += 1
                        continue
                    # Stop if we hit a header or clearly non-stats content
                    if next_line.startswith('#') or len(next_line.split()) > 20:
                        break
                    # Continue if it looks like stats (numbers, short phrases)
                    if re.search(r'\d+[PS]|\d+¥|[\d\-]+', next_line):
                        j += 1
                    else:
                        break

                end_line = j - 1
                if end_line > start_line:  # Only if we found a multi-line block
                    tables.append((start_line, end_line, table_type))
                    logger.info(f"      Found weapon stats: lines {start_line}-{end_line}")
                    i = j
                    continue

            i += 1

        return tables

    def _chunk_document_with_table_awareness(self, text: str, source: str) -> List[Dict]:
        """Chunk document while preserving table integrity AND extracting entities."""

        # Import and use regex cleaner directly
        from tools.regex_text_cleaner import create_regex_cleaner
        regex_cleaner = create_regex_cleaner()

        logger.info("  Starting document cleaning...")
        cleaned_text = regex_cleaner.clean_text(text)

        if not cleaned_text.strip():
            logger.warning(f"  No content after cleaning")
            return []

        char_count = len(cleaned_text)
        token_count = self._count_tokens(cleaned_text)
        line_count = cleaned_text.count('\n') + 1
        logger.info(f"  Cleaned text: {char_count:,} chars, {token_count:,} tokens, {line_count:,} lines")

        # Detect tables first
        logger.info("  Detecting tables...")
        table_boundaries = self._detect_table_boundaries(cleaned_text)
        logger.info(f"  Found {len(table_boundaries)} tables")

        # Split by headers to preserve document structure
        logger.info("  Analyzing document structure...")
        sections = self._split_by_headers(cleaned_text)

        if not sections:
            logger.info("  No headers found, treating as single section")
            sections = [{"title": "Content", "content": cleaned_text, "level": 1}]
        else:
            logger.info(f"  Found {len(sections)} sections:")
            for i, section in enumerate(sections[:5]):  # Show first 5
                section_tokens = self._count_tokens(section['content'])
                logger.info(f"    {i+1}. {section['title'][:50]}... ({section_tokens} tokens)")

        all_chunks = []
        file_entity_count = 0

        for section_idx, section in enumerate(sections):
            section_tokens = self._count_tokens(section['content'])
            logger.info(f"  Processing section {section_idx + 1}/{len(sections)}: '{section['title']}' ({section_tokens} tokens)")

            section_chunks = self._chunk_section_with_table_awareness(
                section, source, section_idx, table_boundaries
            )

            # Process each chunk for entity extraction
            for chunk in section_chunks:
                logger.info(f"    Extracting entities from chunk {chunk['id']}")
                entities = self.entity_builder.extract_entities_from_chunk(chunk)

                # Store entities if any found
                if any(entities.values()):
                    chunk_id = chunk.get("id", "")
                    logger.info(f"      Storing entities for chunk {chunk_id}")
                    self.entity_storage.store_entities(entities, chunk_id, source)

                    # Update statistics
                    entity_counts = {
                        "weapons": len(entities.get("weapons", [])),
                        "spells": len(entities.get("spells", [])),
                        "ic_programs": len(entities.get("ic_programs", []))
                    }

                    total_entities = sum(entity_counts.values())
                    file_entity_count += total_entities

                    # Add entity metadata to chunk
                    chunk_metadata = chunk.get("metadata", {})
                    chunk_metadata["extracted_entities"] = entity_counts
                    chunk_metadata["has_entities"] = True
                    chunk["metadata"] = chunk_metadata

                    logger.info(f"      Extracted {total_entities} entities: {entity_counts}")
                else:
                    logger.info(f"      No entities found in chunk")

            all_chunks.extend(section_chunks)
            logger.info(f"    Section created {len(section_chunks)} chunks")

        # Add sequential links for context preservation
        logger.info("  Adding sequential links between chunks...")
        self._add_sequential_links(all_chunks)

        # Update global statistics
        if file_entity_count > 0:
            chunks_with_entities = len([c for c in all_chunks if c.get("metadata", {}).get("has_entities")])
            self.entity_stats["chunks_with_entities"] += chunks_with_entities
            self.entity_stats["total_entities_extracted"] += file_entity_count
            logger.info(f"  Entity extraction summary: {file_entity_count} total entities from {chunks_with_entities}/{len(all_chunks)} chunks")

        logger.info(f"  Document chunking complete: {len(all_chunks)} total chunks")
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

        return sections

    def _chunk_section_with_table_awareness(self, section: Dict, source: str, section_idx: int,
                                          table_boundaries: List[Tuple[int, int, str]]) -> List[Dict]:
        """Chunk a section while preserving table integrity."""

        content = section["content"]
        title = section["title"]

        if not content.strip():
            logger.info(f"    Section '{title}' is empty, skipping")
            return []

        # Estimate chunks based on tokens
        total_tokens = self._count_tokens(content)
        estimated_chunks = max(1, total_tokens // (self.chunk_size - self.chunk_overlap))
        logger.info(f"    Section length: {total_tokens:,} tokens → ~{estimated_chunks} chunks estimated")

        # Identify tables within this section
        section_lines = content.split('\n')
        section_tables = []

        for start_line, end_line, table_type in table_boundaries:
            # Check if table overlaps with this section (simplified check)
            section_tables.append((start_line, end_line, table_type))

        logger.info(f"    Section contains {len(section_tables)} tables")

        chunks = []
        chunk_idx = 0

        # Parse content into blocks (tables vs text)
        content_blocks = self._parse_content_blocks(content, section_tables)
        logger.info(f"    Split into {len(content_blocks)} content blocks")

        current_chunk_blocks = []
        current_chunk_tokens = 0

        for block_idx, block in enumerate(content_blocks):
            block_tokens = self._count_tokens(block['content'])
            block_type = block['type']

            logger.info(f"      Block {block_idx + 1}: {block_type} ({block_tokens} tokens)")

            # Handle tables specially
            if block_type == 'table':
                # If table is too big for current chunk, finalize current chunk first
                if current_chunk_blocks and current_chunk_tokens + block_tokens > self.chunk_size:
                    chunk_text = self._combine_blocks(current_chunk_blocks)
                    if chunk_text.strip():
                        logger.info(f"    Creating chunk {chunk_idx + 1} (before table) with {current_chunk_tokens} tokens")
                        chunk_data = self._create_chunk_with_metadata(
                            chunk_text, source, section_idx, chunk_idx, title, current_chunk_tokens, "mixed"
                        )
                        chunks.append(chunk_data)
                        chunk_idx += 1

                    current_chunk_blocks = []
                    current_chunk_tokens = 0

                # If table fits in remaining space, add to current chunk
                if current_chunk_tokens + block_tokens <= self.chunk_size:
                    current_chunk_blocks.append(block)
                    current_chunk_tokens += block_tokens
                    logger.info(f"        Added table to current chunk (now {current_chunk_tokens} tokens)")
                else:
                    # Table is too big - create dedicated table chunk
                    if block_tokens > self.chunk_size:
                        logger.warning(f"        Large table ({block_tokens} tokens) exceeds chunk size, splitting")
                        table_chunks = self._split_large_table(block['content'], source, section_idx, chunk_idx, title)
                        chunks.extend(table_chunks)
                        chunk_idx += len(table_chunks)
                    else:
                        logger.info(f"        Creating dedicated table chunk with {block_tokens} tokens")
                        chunk_data = self._create_chunk_with_metadata(
                            block['content'], source, section_idx, chunk_idx, title, block_tokens, "table"
                        )
                        chunks.append(chunk_data)
                        chunk_idx += 1

            # Handle text blocks
            else:
                # Split text into sentences for normal processing
                sentences = re.split(r'(?<=[.!?])\s+', block['content'])
                if not sentences[-1].strip():
                    sentences = sentences[:-1]

                for sentence in sentences:
                    sentence_tokens = self._count_tokens(sentence)

                    # Check if adding this sentence would exceed chunk size
                    if current_chunk_tokens + sentence_tokens > self.chunk_size and current_chunk_blocks:
                        # Create chunk from current content
                        chunk_text = self._combine_blocks(current_chunk_blocks)
                        if chunk_text.strip():
                            logger.info(f"    Creating chunk {chunk_idx + 1} with {current_chunk_tokens} tokens")
                            chunk_data = self._create_chunk_with_metadata(
                                chunk_text, source, section_idx, chunk_idx, title, current_chunk_tokens, "text"
                            )
                            chunks.append(chunk_data)
                            chunk_idx += 1

                        # Start new chunk with overlap
                        overlap_content = self._get_overlap_content(current_chunk_blocks, self.chunk_overlap)
                        current_chunk_blocks = [{'type': 'text', 'content': overlap_content}] if overlap_content else []
                        current_chunk_tokens = self._count_tokens(overlap_content) if overlap_content else 0

                    # Add sentence to current chunk
                    if current_chunk_blocks and current_chunk_blocks[-1]['type'] == 'text':
                        current_chunk_blocks[-1]['content'] += ' ' + sentence
                    else:
                        current_chunk_blocks.append({'type': 'text', 'content': sentence})
                    current_chunk_tokens += sentence_tokens

        # Create final chunk if there's remaining content
        if current_chunk_blocks:
            chunk_text = self._combine_blocks(current_chunk_blocks)
            if chunk_text.strip():
                logger.info(f"    Creating final chunk {chunk_idx + 1} with {current_chunk_tokens} tokens")
                chunk_data = self._create_chunk_with_metadata(
                    chunk_text, source, section_idx, chunk_idx, title, current_chunk_tokens, "text"
                )
                chunks.append(chunk_data)

        logger.info(f"    Section '{title}' chunking complete: {len(chunks)} chunks created")
        return chunks

    def _parse_content_blocks(self, content: str, tables: List[Tuple[int, int, str]]) -> List[Dict]:
        """Parse content into blocks of tables vs text."""

        lines = content.split('\n')
        blocks = []
        current_text_lines = []

        i = 0
        while i < len(lines):
            line = lines[i]

            # Check if this line starts a table
            in_table = False
            for start_line, end_line, table_type in tables:
                # Simplified table detection within content
                if ('|' in line and line.count('|') >= 2) or \
                   any(term in line.lower() for term in ['acc', 'damage', 'mode', 'ammo']):

                    # Save accumulated text as a block
                    if current_text_lines:
                        text_content = '\n'.join(current_text_lines).strip()
                        if text_content:
                            blocks.append({'type': 'text', 'content': text_content})
                        current_text_lines = []

                    # Extract table content
                    table_lines = [line]
                    j = i + 1
                    while j < len(lines):
                        next_line = lines[j]
                        if ('|' in next_line) or \
                           (table_type == 'weapon_stats' and re.search(r'\d+[PS]|\d+¥|[\d\-]+', next_line)):
                            table_lines.append(next_line)
                            j += 1
                        elif next_line.strip() == '':
                            table_lines.append(next_line)
                            j += 1
                        else:
                            break

                    table_content = '\n'.join(table_lines).strip()
                    if table_content:
                        blocks.append({'type': 'table', 'content': table_content})

                    i = j
                    in_table = True
                    break

            if not in_table:
                current_text_lines.append(line)
                i += 1

        # Add remaining text
        if current_text_lines:
            text_content = '\n'.join(current_text_lines).strip()
            if text_content:
                blocks.append({'type': 'text', 'content': text_content})

        return blocks

    def _combine_blocks(self, blocks: List[Dict]) -> str:
        """Combine content blocks into a single text."""
        return '\n\n'.join(block['content'] for block in blocks if block['content'].strip())

    def _get_overlap_content(self, blocks: List[Dict], overlap_tokens: int) -> str:
        """Get overlap content from the end of current blocks."""
        if not blocks:
            return ""

        # Take content from the last text block for overlap
        for block in reversed(blocks):
            if block['type'] == 'text':
                content = block['content']
                sentences = re.split(r'(?<=[.!?])\s+', content)

                overlap_sentences = []
                current_tokens = 0

                for sentence in reversed(sentences):
                    sentence_tokens = self._count_tokens(sentence)
                    if current_tokens + sentence_tokens <= overlap_tokens:
                        overlap_sentences.insert(0, sentence)
                        current_tokens += sentence_tokens
                    else:
                        break

                return ' '.join(overlap_sentences)

        return ""

    def _split_large_table(self, table_content: str, source: str, section_idx: int,
                          chunk_idx: int, title: str) -> List[Dict]:
        """Split a large table while preserving headers."""

        lines = table_content.split('\n')
        header_lines = []
        data_lines = []

        # Identify header vs data lines
        for line in lines:
            if '|' in line and any(term in line.lower() for term in ['weapon', 'acc', 'damage', 'name']):
                header_lines.append(line)
            elif '|' in line or re.search(r'\d+[PS]|\d+¥', line):
                data_lines.append(line)

        logger.info(f"        Splitting large table: {len(header_lines)} header lines, {len(data_lines)} data lines")

        chunks = []
        current_chunk_idx = chunk_idx

        # Create chunks with headers + subset of data
        rows_per_chunk = 10  # Adjust based on token limits

        for i in range(0, len(data_lines), rows_per_chunk):
            chunk_lines = header_lines + data_lines[i:i + rows_per_chunk]
            chunk_content = '\n'.join(chunk_lines)
            chunk_tokens = self._count_tokens(chunk_content)

            chunk_data = self._create_chunk_with_metadata(
                chunk_content, source, section_idx, current_chunk_idx, title, chunk_tokens, "table_split"
            )
            chunks.append(chunk_data)
            current_chunk_idx += 1

        return chunks

    def _create_chunk_with_metadata(self, chunk_text: str, source: str, section_idx: int,
                                  chunk_idx: int, title: str, token_count: int, chunk_type: str) -> Dict:
        """Create a chunk with full metadata including classification."""

        chunk_id = f"{Path(source).stem}_section_{section_idx}_chunk_{chunk_idx}"

        # Classify chunk using unified classifier with logging
        logger.info(f"      Classifying {chunk_type} chunk content...")
        classification_start = time.time()
        classification = self.classifier.classify_content(chunk_text, source)
        classification_time = time.time() - classification_start

        logger.info(f"      Classification result ({classification_time:.2f}s):")
        logger.info(f"        Section: {classification.get('primary_section', 'Unknown')}")
        logger.info(f"        Type: {classification.get('content_type', 'unknown')}")
        logger.info(f"        Rules: {classification.get('contains_rules', False)}")
        logger.info(f"        Confidence: {classification.get('confidence', 0):.2f}")

        # Build chunk metadata
        return {
            "id": chunk_id,
            "text": chunk_text,
            "source": source,
            "metadata": {
                "section_title": title,
                "chunk_index": chunk_idx,
                "token_count": token_count,
                "char_count": len(chunk_text),
                "section_index": section_idx,
                "chunk_type": chunk_type,
                "is_table_content": chunk_type in ['table', 'table_split'],
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
                "document_type": classification.get("document_type", "rulebook"),
                "edition": classification.get("edition", "SR5")
            }
        }

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

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings from Ollama with progress logging."""
        logger.info(f"  Generating {len(texts)} embeddings with {self.embedding_model}")

        embeddings = []
        start_time = time.time()

        for i, text in enumerate(texts):
            try:
                response = ollama.embeddings(
                    model=self.embedding_model,
                    prompt=text
                )
                embeddings.append(response['embedding'])

                # Log progress every 25% or every 10 chunks, whichever is less frequent
                log_interval = max(10, len(texts) // 4)
                if i % log_interval == 0 and i > 0:
                    elapsed = time.time() - start_time
                    rate = i / elapsed
                    eta = (len(texts) - i) / rate if rate > 0 else 0
                    logger.info(f"    Embedding progress: {i}/{len(texts)} ({rate:.1f}/sec, ETA: {eta:.0f}s)")

            except Exception as e:
                logger.error(f"    Error generating embedding {i+1}: {e}")
                # Create a zero embedding as fallback
                embeddings.append([0.0] * 768)

        total_elapsed = time.time() - start_time
        logger.info(f"  Embeddings complete: {len(texts)} in {total_elapsed:.1f}s")

        return embeddings

    def index_file(self, file_path: Path) -> bool:
        """Index a single file with comprehensive logging and entity extraction."""

        logger.info(f"Processing: {file_path.name}")

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False

        # File info
        file_size = file_path.stat().st_size
        logger.info(f"  File size: {file_size:,} bytes")

        # Check if file has changed
        current_hash = self._get_file_hash(file_path)
        file_key = str(file_path)

        if file_key in self.indexed_files:
            stored_hash = self.indexed_files[file_key]['hash']
            if stored_hash == current_hash:
                logger.info(f"  Unchanged, skipping")
                return True
            else:
                logger.info(f"  File changed, re-indexing")
                self._remove_file_chunks(file_key)
        else:
            logger.info(f"  New file, indexing")

        try:
            # Read file content
            logger.info("  Reading file...")
            content = file_path.read_text(encoding='utf-8')

            char_count = len(content)
            token_count = self._count_tokens(content)
            line_count = content.count('\n') + 1
            logger.info(f"  Content: {char_count:,} chars, {token_count:,} tokens, {line_count:,} lines")

            # Create chunks with table awareness AND entity extraction
            logger.info("  Starting table-aware token-based chunking with entity extraction...")
            chunks = self._chunk_document_with_table_awareness(content, str(file_path))

            if not chunks:
                logger.warning(f"  No chunks created")
                return False

            # Log chunk size statistics
            token_counts = [chunk["metadata"]["token_count"] for chunk in chunks]
            table_chunks = sum(1 for chunk in chunks if chunk["metadata"]["is_table_content"])
            text_chunks = len(chunks) - table_chunks

            avg_tokens = sum(token_counts) / len(token_counts)
            min_tokens = min(token_counts)
            max_tokens = max(token_counts)
            logger.info(f"  Chunk statistics: {len(chunks)} total ({table_chunks} tables, {text_chunks} text)")
            logger.info(f"  Token distribution: avg={avg_tokens:.0f}, min={min_tokens}, max={max_tokens}")

            # Prepare data for ChromaDB
            texts = [chunk["text"] for chunk in chunks]
            ids = [chunk["id"] for chunk in chunks]

            logger.info("  Cleaning metadata for ChromaDB...")
            metadatas = [self._clean_metadata_for_chroma(chunk["metadata"]) for chunk in chunks]

            # Generate embeddings
            embeddings = self._get_embeddings(texts)

            # Add to ChromaDB
            logger.info("  Storing in ChromaDB...")
            storage_start = time.time()

            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )

            storage_time = time.time() - storage_start
            logger.info(f"  Storage complete in {storage_time:.1f}s")

            # Update index metadata
            self.indexed_files[file_key] = {
                'hash': current_hash,
                'chunk_count': len(chunks),
                'table_chunks': table_chunks,
                'text_chunks': text_chunks,
                'indexed_at': str(file_path.absolute()),
                'file_size': file_size,
                'indexed_timestamp': time.time()
            }

            self._save_index_metadata()

            final_count = self.collection.count()
            logger.info(f"  Success: {len(chunks)} chunks added (collection: {final_count} total)")

            return True

        except Exception as e:
            logger.error(f"  Error: {e}")
            import traceback
            logger.debug(f"Full traceback:\n{traceback.format_exc()}")
            return False

    def _remove_file_chunks(self, file_key: str):
        """Remove chunks for a specific file from the collection."""
        try:
            results = self.collection.get(where={"source": file_key})

            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"  Removed {len(results['ids'])} old chunks")

        except Exception as e:
            logger.warning(f"  Error removing old chunks: {e}")

    def index_directory(self, directory_path: str, force_reindex: bool = False) -> Dict:
        """Index all markdown files in a directory with progress tracking and entity extraction."""

        logger.info(f"Starting enhanced directory indexing: {directory_path}")
        logger.info(f"Using TABLE-AWARE token-based chunking with ENTITY EXTRACTION: {self.chunk_size} tokens per chunk, {self.chunk_overlap} token overlap")
        if force_reindex:
            logger.info("Force reindex enabled - will process all files")

        directory = Path(directory_path)
        if not directory.exists():
            logger.error(f"Directory not found: {directory}")
            return {"success": False, "error": f"Directory not found: {directory}"}

        # Find all markdown files
        markdown_files = list(directory.glob("**/*.md"))

        if not markdown_files:
            logger.warning(f"No markdown files found in {directory}")
            return {"success": False, "error": "No markdown files found"}

        total_size = sum(f.stat().st_size for f in markdown_files)
        logger.info(f"Found {len(markdown_files)} files ({total_size/1024/1024:.1f} MB total)")

        # Reset entity statistics for this run
        self.entity_stats = {
            "total_entities_extracted": 0,
            "weapons_extracted": 0,
            "spells_extracted": 0,
            "ic_programs_extracted": 0,
            "chunks_with_entities": 0
        }

        results = {
            "success": True,
            "total_files": len(markdown_files),
            "indexed_files": 0,
            "failed_files": 0,
            "skipped_files": 0,
            "errors": []
        }

        start_time = time.time()

        for file_idx, file_path in enumerate(markdown_files):
            logger.info(f"\n[{file_idx + 1}/{len(markdown_files)}] Starting: {file_path.name}")

            try:
                file_key = str(file_path)
                current_hash = self._get_file_hash(file_path)

                # Check if file needs indexing
                if not force_reindex and file_key in self.indexed_files:
                    if self.indexed_files[file_key]['hash'] == current_hash:
                        logger.info(f"  Skipping (unchanged)")
                        results["skipped_files"] += 1
                        continue

                file_start_time = time.time()
                success = self.index_file(file_path)
                file_elapsed = time.time() - file_start_time

                if success:
                    results["indexed_files"] += 1
                    logger.info(f"  Completed in {file_elapsed:.1f}s")
                else:
                    results["failed_files"] += 1
                    results["errors"].append(f"Failed: {file_path.name}")
                    logger.error(f"  Failed to index")

                # Progress estimate
                elapsed = time.time() - start_time
                avg_time = elapsed / (file_idx + 1)
                remaining = (len(markdown_files) - file_idx - 1) * avg_time
                logger.info(f"  Progress: {file_idx + 1}/{len(markdown_files)}, ETA: {remaining/60:.1f}m")

            except Exception as e:
                results["failed_files"] += 1
                results["errors"].append(f"Error with {file_path.name}: {str(e)}")
                logger.error(f"  Fatal error: {e}")

        total_elapsed = time.time() - start_time
        final_registry_stats = self.entity_storage.get_registry_stats()

        logger.info(f"\nEnhanced indexing complete in {total_elapsed/60:.1f} minutes")
        logger.info(f"Results: {results['indexed_files']} indexed, {results['skipped_files']} skipped, {results['failed_files']} failed")
        logger.info(f"Entity Extraction Summary:")
        logger.info(f"  - Total entities extracted: {self.entity_stats['total_entities_extracted']}")
        logger.info(f"  - Chunks with entities: {self.entity_stats['chunks_with_entities']}")
        logger.info(f"  - Registry contents: {final_registry_stats}")

        if results['errors']:
            logger.warning(f"Errors: {results['errors'][:3]}")

        final_count = self.collection.count()
        logger.info(f"Collection now has {final_count} total chunks")

        # Add entity processing information to results
        results["entity_processing"] = {
            "entities_extracted": self.entity_stats["total_entities_extracted"],
            "chunks_with_entities": self.entity_stats["chunks_with_entities"],
            "registry_stats": final_registry_stats
        }

        return results

    def get_collection_stats(self) -> Dict:
        """Get statistics about the indexed collection."""
        try:
            count = self.collection.count()

            if count == 0:
                return {
                    "total_chunks": 0,
                    "indexed_files": 0,
                    "collection_name": self.collection.name
                }

            sample_size = min(100, count)
            sample = self.collection.peek(limit=sample_size)

            section_counts = {}
            content_type_counts = {}
            chunk_type_counts = {}
            table_chunks = 0

            for metadata in sample.get('metadatas', []):
                if metadata:
                    section = metadata.get('primary_section', 'Unknown')
                    section_counts[section] = section_counts.get(section, 0) + 1

                    content_type = metadata.get('content_type', 'unknown')
                    content_type_counts[content_type] = content_type_counts.get(content_type, 0) + 1

                    chunk_type = metadata.get('chunk_type', 'unknown')
                    chunk_type_counts[chunk_type] = chunk_type_counts.get(chunk_type, 0) + 1

                    if metadata.get('is_table_content') == 'true':  # String comparison for ChromaDB
                        table_chunks += 1

            return {
                "total_chunks": count,
                "indexed_files": len(self.indexed_files),
                "collection_name": self.collection.name,
                "embedding_model": self.embedding_model,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "semantic_splitting": self.use_semantic_splitting,
                "section_distribution": section_counts,
                "content_type_distribution": content_type_counts,
                "chunk_type_distribution": chunk_type_counts,
                "table_chunks": table_chunks,
                "sample_size": sample_size
            }

        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}

    def search_debug(self, query: str, n_results: int = 5) -> Dict:
        """Debug search to test retrieval with entity awareness."""
        try:
            # Use query processor to enhance query
            analysis = self.query_processor.analyze_query(query)
            enhanced_query = self.query_processor.build_enhanced_query(query, analysis)

            # Extract entities from query
            entities = self.entity_builder.extract_entities_from_chunk({
                "text": query,
                "source": "query"
            })

            # Search collection
            results = self.collection.query(
                query_texts=[enhanced_query],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )

            return {
                "original_query": query,
                "enhanced_query": enhanced_query,
                "analysis": {
                    "intent": analysis.intent,
                    "entities": analysis.entities,
                    "query_type": analysis.query_type,
                    "confidence": analysis.confidence
                },
                "extracted_entities": entities,
                "results_count": len(results['documents'][0]) if results['documents'] else 0,
                "results": results
            }

        except Exception as e:
            logger.error(f"Error in debug search: {e}")
            return {"error": str(e)}

    def search(self, query: str, n_results: int = 5) -> Dict:
        """Enhanced search with entity awareness and detailed logging."""

        logger.info(f"Starting enhanced search for: '{query[:50]}{'...' if len(query) > 50 else ''}'")

        try:
            # Generate query embedding
            logger.info(f"Generating query embedding...")
            embedding_start_time = time.time()
            query_response = ollama.embeddings(model=self.embedding_model, prompt=query)
            query_embedding = query_response['embedding']
            embedding_time = time.time() - embedding_start_time
            logger.info(f"Query embedding generated in {embedding_time:.2f}s")

            # Perform base semantic search
            logger.info(f"Performing semantic search (requesting {n_results * 2} results for reranking)...")
            search_start_time = time.time()
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results * 2  # Get more for entity reranking
            )
            search_time = time.time() - search_start_time
            logger.info(f"Semantic search complete in {search_time:.2f}s")

            # Extract entities from query
            logger.info(f"Extracting entities from query...")
            entities = self.entity_builder.extract_entities_from_chunk({
                "text": query,
                "source": "query"
            })

            entity_count = sum(len(entity_list) for entity_list in entities.values())
            if entity_count > 0:
                logger.info(f"Found {entity_count} entities in query: {entities}")
            else:
                logger.info(f"No entities detected in query")

            # Validate entity combinations in query
            logger.info(f"Validating entity combinations...")
            validation_warnings = []
            if entities.get("weapons"):
                # Check for weapon + firing mode combinations
                firing_modes = self._extract_firing_modes(query)
                if firing_modes:
                    logger.info(f"Found firing modes in query: {firing_modes}")

                for weapon in entities["weapons"]:
                    for mode in firing_modes:
                        validation = self.entity_storage.validate_weapon_mode(weapon.name, mode)
                        if not validation.get("valid", True):
                            warning = validation.get("error", "Invalid combination")
                            validation_warnings.append(warning)
                            logger.warning(f"Validation warning: {warning}")

            # Add entity information to results
            entity_info = {
                "entities_detected": [],
                "validation_warnings": validation_warnings,
                "registry_stats": self.entity_storage.get_registry_stats()
            }

            # Add detected entities info
            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    entity_info["entities_detected"].append({
                        "type": entity_type[:-1],  # Remove 's' from 'weapons', 'spells'
                        "name": entity.name,
                        "confidence": 0.8
                    })

            # Enhance results format
            enhanced_results = results.copy()
            enhanced_results["entity_info"] = entity_info

            result_count = len(results.get("documents", [[]])[0])
            logger.info(f"Search complete. Returning {min(result_count, n_results)} results with entity information")

            return enhanced_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {"documents": [[]], "metadatas": [[]], "ids": [[]], "distances": [[]]}

    def _extract_firing_modes(self, query: str) -> List[str]:
        """Extract firing mode references from query."""
        mode_patterns = {
            "burst": r'\bburst\s*(?:fire|mode)?\b',
            "semi-auto": r'\bsemi[-\s]*auto(?:matic)?\b',
            "full auto": r'\bfull[-\s]*auto(?:matic)?\b',
            "single shot": r'\bsingle[-\s]*shot\b'
        }

        modes = []
        query_lower = query.lower()

        for mode_name, pattern in mode_patterns.items():
            if re.search(pattern, query_lower):
                modes.append(mode_name)

        return modes

    def get_entity_stats(self, entity_name: str, entity_type: str = None):
        """Get detailed stats for a specific entity with logging."""

        logger.info(f"Looking up entity stats for: '{entity_name}' (type: {entity_type or 'any'})")

        if entity_type == "weapon" or entity_type is None:
            weapon = self.entity_storage.get_weapon(entity_name)
            if weapon:
                logger.info(f"Found weapon: {weapon.name}")
                return {
                    "type": "weapon",
                    "name": weapon.name,
                    "stats": {
                        "accuracy": weapon.accuracy,
                        "damage": weapon.damage,
                        "ap": weapon.ap,
                        "mode": weapon.mode,
                        "rc": weapon.rc,
                        "ammo": weapon.ammo,
                        "avail": weapon.avail,
                        "cost": weapon.cost
                    },
                    "category": weapon.category,
                    "manufacturer": weapon.manufacturer,
                    "description": weapon.description
                }

        if entity_type == "spell" or entity_type is None:
            spell = self.entity_storage.get_spell(entity_name)
            if spell:
                logger.info(f"Found spell: {spell.name}")
                return {
                    "type": "spell",
                    "name": spell.name,
                    "stats": {
                        "spell_type": spell.spell_type,
                        "range": spell.range,
                        "damage": spell.damage,
                        "duration": spell.duration,
                        "drain": spell.drain
                    },
                    "keywords": spell.keywords,
                    "category": spell.category,
                    "description": spell.description
                }

        logger.info(f"No entity found for: '{entity_name}'")
        return None

    def validate_weapon_mode(self, weapon_name: str, mode: str) -> Dict:
        """Validate weapon firing mode capability with logging."""
        logger.info(f"Validating: {weapon_name} can fire in {mode} mode")
        result = self.entity_storage.validate_weapon_mode(weapon_name, mode)

        if result.get("valid"):
            logger.info(f"Validation passed: {weapon_name} supports {mode}")
        else:
            logger.warning(f"Validation failed: {result.get('error', 'Unknown error')}")

        return result

    def get_registry_stats(self) -> Dict:
        """Get entity registry statistics with logging."""
        logger.info(f"Retrieving entity registry statistics...")
        stats = self.entity_storage.get_registry_stats()
        logger.info(f"Registry contains: {stats}")
        return stats

    def optimize_collection(self):
        """Optimize the ChromaDB collection."""
        try:
            stats = self.get_collection_stats()
            logger.info(f"Collection optimization check - {stats.get('total_chunks', 0)} chunks indexed")

            # Check for any cleanup needed
            orphaned_files = []
            for file_key in self.indexed_files.keys():
                if not Path(file_key).exists():
                    orphaned_files.append(file_key)

            if orphaned_files:
                logger.info(f"Found {len(orphaned_files)} orphaned file references")
                for file_key in orphaned_files:
                    self._remove_file_chunks(file_key)
                    del self.indexed_files[file_key]
                self._save_index_metadata()

        except Exception as e:
            logger.error(f"Error optimizing collection: {e}")

    def reset_collection(self):
        """Reset the entire collection (use with caution)."""
        try:
            self.client.delete_collection(self.collection.name)
            self.collection = self.client.create_collection(
                name=self.collection.name,
                metadata={"hnsw:space": "cosine"}
            )
            self.indexed_files = {}
            self._save_index_metadata()
            logger.info("Collection reset successfully")

        except Exception as e:
            logger.error(f"Error resetting collection: {e}")

    def reextract_entities_from_existing(self, entity_types: List[str] = None, chunk_limit: int = None, force_update: bool = False) -> Dict:
        """Re-extract entities from existing chunks with improved patterns without reindexing."""

        logger.info("Starting entity re-extraction from existing chunks")

        if entity_types:
            logger.info(f"Targeting specific entity types: {entity_types}")
        else:
            logger.info("Re-extracting all entity types")

        # Get all chunks or filtered chunks
        if chunk_limit:
            logger.info(f"Processing limited to {chunk_limit} chunks")
            all_chunks = self.collection.get(include=['documents', 'metadatas', 'ids'], limit=chunk_limit)
        else:
            all_chunks = self.collection.get(include=['documents', 'metadatas', 'ids'])

        chunks_to_process = all_chunks['ids']

        if not force_update:
            # Only process chunks that don't have entities or need updating
            chunks_to_process = []
            for i, metadata in enumerate(all_chunks['metadatas']):
                if not metadata or not metadata.get('has_entities'):
                    chunks_to_process.append(all_chunks['ids'][i])

            logger.info(f"Found {len(chunks_to_process)} chunks without entities")
        else:
            logger.info(f"Force update enabled - processing all {len(chunks_to_process)} chunks")

        # Reset statistics for this run
        extraction_stats = {
            "chunks_processed": 0,
            "chunks_with_new_entities": 0,
            "weapons_found": 0,
            "spells_found": 0,
            "ic_programs_found": 0,
            "total_entities_found": 0
        }

        start_time = time.time()

        for i, chunk_id in enumerate(chunks_to_process):
            # Get the corresponding document and metadata
            doc_index = all_chunks['ids'].index(chunk_id)
            document = all_chunks['documents'][doc_index]
            metadata = all_chunks['metadatas'][doc_index] or {}

            logger.info(f"Processing chunk {i + 1}/{len(chunks_to_process)}: {chunk_id}")

            # Create chunk object for entity extraction
            chunk = {
                "id": chunk_id,
                "text": document,
                "source": metadata.get('source', 'unknown'),
                "metadata": metadata
            }

            # Extract entities with updated patterns
            entities = self.entity_builder.extract_entities_from_chunk(chunk)

            # Store entities if any found
            if any(entities.values()):
                source = metadata.get('source', 'reextraction')
                self.entity_storage.store_entities(entities, chunk_id, source)

                # Update statistics
                entity_counts = {
                    "weapons": len(entities.get("weapons", [])),
                    "spells": len(entities.get("spells", [])),
                    "ic_programs": len(entities.get("ic_programs", []))
                }

                total_chunk_entities = sum(entity_counts.values())

                extraction_stats["chunks_with_new_entities"] += 1
                extraction_stats["weapons_found"] += entity_counts["weapons"]
                extraction_stats["spells_found"] += entity_counts["spells"]
                extraction_stats["ic_programs_found"] += entity_counts["ic_programs"]
                extraction_stats["total_entities_found"] += total_chunk_entities

                logger.info(f"  Found {total_chunk_entities} entities: {entity_counts}")

                # Update chunk metadata in ChromaDB (optional - preserves the has_entities flag)
                try:
                    updated_metadata = metadata.copy()
                    updated_metadata["has_entities"] = True
                    updated_metadata["entity_extraction_version"] = "2.0"  # Version tracking

                    # Update the metadata in ChromaDB
                    self.collection.update(
                        ids=[chunk_id],
                        metadatas=[self._clean_metadata_for_chroma(updated_metadata)]
                    )
                except Exception as e:
                    logger.warning(f"  Could not update chunk metadata: {e}")

            else:
                logger.info(f"  No entities found")

            extraction_stats["chunks_processed"] += 1

            # Progress reporting
            if (i + 1) % 25 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (len(chunks_to_process) - i - 1) / rate if rate > 0 else 0
                logger.info(f"Progress: {i + 1}/{len(chunks_to_process)} ({rate:.1f}/sec, ETA: {eta:.0f}s)")

        total_elapsed = time.time() - start_time
        final_registry_stats = self.entity_storage.get_registry_stats()

        logger.info(f"Entity re-extraction complete in {total_elapsed:.1f}s")
        logger.info(f"Results: {extraction_stats}")
        logger.info(f"Updated registry: {final_registry_stats}")

        return {
            "success": True,
            "extraction_stats": extraction_stats,
            "final_registry_stats": final_registry_stats,
            "processing_time": total_elapsed
        }
        """Generate a report on chunk quality focusing on table preservation and entity extraction."""
        try:
            # Sample some chunks to analyze
            sample = self.collection.peek(limit=50)

            if not sample or not sample.get('metadatas'):
                return {"error": "No chunks found for analysis"}

            # Analyze chunk quality
            total_chunks = len(sample['documents'])
            table_chunks = 0
            text_chunks = 0
            mixed_chunks = 0
            entity_chunks = 0

            token_counts = []

            for metadata in sample['metadatas']:
                if metadata:
                    if metadata.get('is_table_content') == 'true':
                        table_chunks += 1
                    elif metadata.get('chunk_type') == 'mixed':
                        mixed_chunks += 1
                    else:
                        text_chunks += 1

                    if metadata.get('has_entities') == 'true':
                        entity_chunks += 1

                    # Extract token count (stored as string in ChromaDB)
                    token_count = metadata.get('token_count')
                    if token_count:
                        try:
                            token_counts.append(int(token_count))
                        except ValueError:
                            pass

            avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0

            return {
                "sample_size": total_chunks,
                "chunk_distribution": {
                    "table_chunks": table_chunks,
                    "text_chunks": text_chunks,
                    "mixed_chunks": mixed_chunks,
                    "entity_chunks": entity_chunks
                },
                "token_statistics": {
                    "average_tokens": avg_tokens,
                    "min_tokens": min(token_counts) if token_counts else 0,
                    "max_tokens": max(token_counts) if token_counts else 0
                },
                "table_preservation_score": (table_chunks / total_chunks * 100) if total_chunks > 0 else 0,
                "entity_extraction_rate": (entity_chunks / total_chunks * 100) if total_chunks > 0 else 0,
                "collection_total": self.collection.count(),
                "registry_stats": self.entity_storage.get_registry_stats()
            }

        except Exception as e:
            logger.error(f"Error generating quality report: {e}")
            return {"error": str(e)}


# Factory function for backward compatibility
def create_enhanced_indexer(**kwargs):
    """Create enhanced indexer instance."""
    return EnhancedIncrementalIndexer(**kwargs)


# Alias for backward compatibility with original naming
def create_incremental_indexer(**kwargs):
    """Create incremental indexer instance (backward compatibility)."""
    return EnhancedIncrementalIndexer(**kwargs)